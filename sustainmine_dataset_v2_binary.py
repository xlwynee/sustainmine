from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


S2_RE = re.compile(r"^S2_(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)
S5_RE = re.compile(r"^S5P_(NO2|SO2|CO)_(?:Daily_)?(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)
S5_EXTENSIONS = ("*.tif", "*.TIF", "*.nc", "*.NC")


@dataclass
class SampleRecord:
    date: str
    s2_path: Path
    no2_path: Path
    so2_path: Path
    co_path: Path
    label: int
    sensor_current: np.ndarray
    sensor_forecast: np.ndarray


class SustainMineDatasetV2(Dataset):
    """
    Dataset that uses only days where Sentinel-2, Sentinel-5P, and sensor rows exist.

    Expected label setup for binary classification:
    - 0 = low
    - 1 = polluted

    If `binary_label` is missing but `pollution_label` exists, binary labels are created as:
    - 0 -> 0
    - 1 or 2 -> 1
    """

    def __init__(
        self,
        sensor_data_path: str | Path,
        sentinel2_path: str | Path = "data/sentinel_2",
        sentinel5_path: str | Path = "data/sentinel_5",
        img_size: int = 224,
        forecast_horizon: int = 3,
        sensor_feature_columns: Optional[Sequence[str]] = None,
        forecast_target_columns: Optional[Sequence[str]] = None,
        date_column: str = "Date",
        label_column: str = "binary_label",
    ) -> None:
        self.sensor_data_path = Path(sensor_data_path)
        self.sentinel2_path = Path(sentinel2_path)
        self.sentinel5_path = Path(sentinel5_path)
        self.img_size = img_size
        self.forecast_horizon = forecast_horizon
        self.date_column = date_column
        self.label_column = label_column

        self.sensor_df = pd.read_csv(self.sensor_data_path)

        if self.date_column not in self.sensor_df.columns:
            raise ValueError(f"Missing date column '{self.date_column}' in sensor CSV")

        self.sensor_df[self.date_column] = pd.to_datetime(
            self.sensor_df[self.date_column]
        ).dt.strftime("%Y-%m-%d")
        self.sensor_df = self.sensor_df.sort_values(self.date_column).reset_index(drop=True)

        self._ensure_binary_label_column()

        if sensor_feature_columns is None:
            sensor_feature_columns = self._infer_sensor_columns(self.sensor_df)
        if forecast_target_columns is None:
            forecast_target_columns = self._infer_forecast_columns(self.sensor_df)

        self.sensor_feature_columns = list(sensor_feature_columns)
        self.forecast_target_columns = list(forecast_target_columns)

        self.sensor_by_date = self.sensor_df.set_index(self.date_column)
        self.s2_index = self._index_s2_files(self.sentinel2_path)
        self.s5_index = self._index_s5_files(self.sentinel5_path)
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(
                "No valid samples were built. Check S2 filenames, S5 filenames, sensor dates, and labels."
            )

    def _ensure_binary_label_column(self) -> None:
        if self.label_column in self.sensor_df.columns:
            self.sensor_df[self.label_column] = self.sensor_df[self.label_column].astype(int)
        elif "pollution_label" in self.sensor_df.columns:
            self.sensor_df[self.label_column] = self.sensor_df["pollution_label"].apply(
                lambda x: 0 if int(x) == 0 else 1
            ).astype(int)
        else:
            raise ValueError(
                f"Missing label column '{self.label_column}'. "
                "Expected 'binary_label' or at least 'pollution_label' in the sensor CSV."
            )

        unique_values = set(self.sensor_df[self.label_column].dropna().astype(int).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Binary classification expects only labels {{0, 1}} in '{self.label_column}', "
                f"but found: {sorted(unique_values)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]

        image = self._load_s2_image(sample.s2_path)
        s5_vector = self._load_s5_vector(sample.no2_path, sample.so2_path, sample.co_path)

        return {
            "image": torch.from_numpy(image),
            "s5": torch.from_numpy(s5_vector),
            "sensor": torch.from_numpy(sample.sensor_current.astype(np.float32)),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "forecast": torch.from_numpy(sample.sensor_forecast.astype(np.float32)),
            "date": sample.date,
        }

    @staticmethod
    def _infer_sensor_columns(df: pd.DataFrame) -> List[str]:
        preferred = [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "Temperature", "Humidity", "Wind Speed",
        ]
        return [c for c in preferred if c in df.columns]

    @staticmethod
    def _infer_forecast_columns(df: pd.DataFrame) -> List[str]:
        preferred = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        found = [c for c in preferred if c in df.columns]
        if not found:
            raise ValueError("Could not infer forecast target columns from sensor CSV")
        return found

    @staticmethod
    def _extract_s2_date(path: Path) -> Optional[str]:
        match = S2_RE.match(path.stem)
        return match.group(1) if match else None

    @staticmethod
    def _extract_s5_info(path: Path) -> Optional[Tuple[str, str]]:
        match = S5_RE.match(path.stem)
        if not match:
            return None
        return match.group(1).upper(), match.group(2)

    def _index_s2_files(self, base: Path) -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        for tif in list(base.rglob("*.tif")) + list(base.rglob("*.TIF")):
            date = self._extract_s2_date(tif)
            if date:
                index[date] = tif
        return index

    def _index_s5_files(self, base: Path) -> Dict[str, Dict[str, Path]]:
        gases = {"NO2": {}, "SO2": {}, "CO": {}}
        all_files: List[Path] = []
        for pattern in S5_EXTENSIONS:
            all_files.extend(base.rglob(pattern))

        for file_path in all_files:
            info = self._extract_s5_info(file_path)
            if not info:
                continue
            gas, date = info
            if gas in gases:
                gases[gas][date] = file_path
        return gases

    def _build_samples(self) -> List[SampleRecord]:
        samples: List[SampleRecord] = []
        available_sensor_dates = set(self.sensor_by_date.index.astype(str))

        for date in sorted(self.s2_index.keys()):
            if date not in available_sensor_dates:
                continue
            if (
                date not in self.s5_index["NO2"]
                or date not in self.s5_index["SO2"]
                or date not in self.s5_index["CO"]
            ):
                continue

            row = self.sensor_by_date.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            future_rows = []
            current_ts = pd.Timestamp(date)
            valid_future = True

            for step in range(1, self.forecast_horizon + 1):
                future_date = (current_ts + pd.Timedelta(days=step)).strftime("%Y-%m-%d")
                if future_date not in available_sensor_dates:
                    valid_future = False
                    break

                future_row = self.sensor_by_date.loc[future_date]
                if isinstance(future_row, pd.DataFrame):
                    future_row = future_row.iloc[0]

                future_rows.append(
                    future_row[self.forecast_target_columns].to_numpy(dtype=np.float32)
                )

            if not valid_future:
                continue

            current_sensor = row[self.sensor_feature_columns].to_numpy(dtype=np.float32)
            label = int(row[self.label_column])
            forecast = np.stack(future_rows, axis=0).astype(np.float32)

            samples.append(
                SampleRecord(
                    date=date,
                    s2_path=self.s2_index[date],
                    no2_path=self.s5_index["NO2"][date],
                    so2_path=self.s5_index["SO2"][date],
                    co_path=self.s5_index["CO"][date],
                    label=label,
                    sensor_current=current_sensor,
                    sensor_forecast=forecast,
                )
            )

        return samples

    def get_label_counts(self) -> Dict[int, int]:
        counts = Counter(sample.label for sample in self.samples)
        return {int(label): int(count) for label, count in sorted(counts.items())}

    def summary(self) -> Dict[str, int | Dict[int, int]]:
        return {
            "s2_dates": len(self.s2_index),
            "no2_dates": len(self.s5_index["NO2"]),
            "so2_dates": len(self.s5_index["SO2"]),
            "co_dates": len(self.s5_index["CO"]),
            "sensor_dates": len(self.sensor_by_date),
            "valid_samples": len(self.samples),
            "label_counts": self.get_label_counts(),
        }

    def _load_s2_image(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        if image.shape[0] < 6:
            raise ValueError(f"Expected at least 6 S2 bands in {path.name}, got {image.shape[0]}")

        image = image[:6]
        image = self._resize_chw(image, self.img_size, self.img_size)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image / 10000.0, 0.0, 1.0).astype(np.float32)
        return image

    def _load_s5_numeric(self, path: Path) -> float:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata

        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)

        band = np.where(np.isfinite(band), band, np.nan)
        value = np.nanmean(band)
        if np.isnan(value):
            value = 0.0
        return float(value)

    def _load_s5_vector(self, no2_path: Path, so2_path: Path, co_path: Path) -> np.ndarray:
        return np.array(
            [
                self._load_s5_numeric(no2_path),
                self._load_s5_numeric(so2_path),
                self._load_s5_numeric(co_path),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _resize_chw(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).numpy()


if __name__ == "__main__":
    dataset = SustainMineDatasetV2(
        sensor_data_path="sensor_data_cleaned_with_binary_labels.csv",
        sentinel2_path="data/sentinel_2",
        sentinel5_path="data/sentinel_5",
        forecast_horizon=3,
        label_column="binary_label",
    )

    print("Dataset summary:")
    for key, value in dataset.summary().items():
        print(f"  {key}: {value}")

    first = dataset[0]
    print("\nFirst sample:")
    print(f"  date: {first['date']}")
    print(f"  image shape: {tuple(first['image'].shape)}")
    print(f"  s5 shape: {tuple(first['s5'].shape)}")
    print(f"  sensor shape: {tuple(first['sensor'].shape)}")
    print(f"  label: {first['label'].item()}")
    print(f"  forecast shape: {tuple(first['forecast'].shape)}")
