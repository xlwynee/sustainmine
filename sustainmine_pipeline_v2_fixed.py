"""
Data indexing helpers for SustainMine.
Builds aligned records from:
- sentinel_2/month/S2_YYYY-MM-DD.tif
- sentinel_5/GAS/month/S5P_GAS_YYYY-MM-DD.tif
- sentinel_5/GAS/month/S5P_GAS_Daily_YYYY-MM-DD.tif
- sentinel_5/GAS/month/S5P_GAS_YYYY-MM-DD.nc
- sentinel_5/GAS/month/S5P_GAS_Daily_YYYY-MM-DD.nc
"""

from pathlib import Path
import re
from typing import Dict, List
import json
import pandas as pd

S2_RE = re.compile(r"^S2_(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)
S5_RE = re.compile(r"^S5P_(NO2|SO2|CO)_(?:Daily_)?(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)
S5_EXTENSIONS = ("*.tif", "*.TIF", "*.nc", "*.NC")

# use your real folder names
MONTHS = ["July", "Aug", "Sep", "Oct", "Nov", "Dec"]
GASES = ["NO2", "SO2", "CO"]


class SustainMineDataPipelineV2:
    def __init__(
        self,
        sensor_data_path: str,
        sentinel2_path: str = "data/sentinel_2",
        sentinel5p_path: str = "data/sentinel_5",
    ):
        self.sensor_data_path = Path(sensor_data_path)
        self.sentinel2_path = Path(sentinel2_path)
        self.sentinel5p_path = Path(sentinel5p_path)

        self.sensor_df = pd.read_csv(self.sensor_data_path)

        if "Date" not in self.sensor_df.columns:
            raise ValueError("sensor_data_cleaned.csv must contain a 'Date' column")

        self.sensor_df["Date"] = pd.to_datetime(self.sensor_df["Date"]).dt.strftime("%Y-%m-%d")

    def _index_s2(self) -> Dict[str, str]:
        index: Dict[str, str] = {}

        for month in MONTHS:
            month_path = self.sentinel2_path / month
            if not month_path.exists():
                continue

            for file in list(month_path.glob("*.tif")) + list(month_path.glob("*.TIF")):
                match = S2_RE.match(file.stem)
                if match:
                    date = match.group(1)
                    index[date] = str(file)

        return index

    def _index_s5(self) -> Dict[str, Dict[str, str]]:
        index: Dict[str, Dict[str, str]] = {gas: {} for gas in GASES}

        for gas in GASES:
            for month in MONTHS:
                month_path = self.sentinel5p_path / gas / month
                if not month_path.exists():
                    continue

                files = []
                for pattern in S5_EXTENSIONS:
                    files.extend(month_path.glob(pattern))

                for file in files:
                    match = S5_RE.match(file.stem)
                    if match:
                        gas_name = match.group(1).upper()
                        date = match.group(2)
                        index[gas_name][date] = str(file)

        return index

    def build_aligned_samples(self) -> List[Dict]:
        s2_index = self._index_s2()
        s5_index = self._index_s5()
        sensor_dates = set(self.sensor_df["Date"])

        shared_dates = sorted(
            set(s2_index)
            & set(s5_index["NO2"])
            & set(s5_index["SO2"])
            & set(s5_index["CO"])
            & sensor_dates
        )

        samples: List[Dict] = []

        for date in shared_dates:
            sensor_match = self.sensor_df[self.sensor_df["Date"] == date]
            if sensor_match.empty:
                continue

            sensor_row = sensor_match.iloc[0].to_dict()

            samples.append(
                {
                    "date": date,
                    "s2_path": s2_index[date],
                    "no2_path": s5_index["NO2"][date],
                    "so2_path": s5_index["SO2"][date],
                    "co_path": s5_index["CO"][date],
                    "sensor_row": sensor_row,
                }
            )

        return samples

    def create_summary(self) -> Dict:
        samples = self.build_aligned_samples()

        return {
            "sensor_rows": int(len(self.sensor_df)),
            "aligned_samples": len(samples),
            "first_dates": [sample["date"] for sample in samples[:10]],
        }


if __name__ == "__main__":
    pipeline = SustainMineDataPipelineV2("sensor_data_cleaned.csv")
    summary = pipeline.create_summary()
    print(json.dumps(summary, indent=2))
