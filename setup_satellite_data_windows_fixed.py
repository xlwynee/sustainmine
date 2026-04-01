from pathlib import Path
import json
import re

MONTHS = ["July", "Aug", "Sep", "Oct", "Nov", "Dec"]
GASES = ["NO2", "SO2", "CO"]
S2_PATTERN = re.compile(r"^S2_(\d{4}-\d{2}-\d{2})\.tif$", re.IGNORECASE)
S5_PATTERN = re.compile(r"^S5P_(NO2|SO2|CO)_(?:Daily_)?(\d{4}-\d{2}-\d{2})\.(tif|nc)$",
             re.IGNORECASE
)

def _count_matching_files(path: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(files)


def _validate_s2_names(month_path: Path) -> tuple[int, list[str]]:
    tif_files = _count_matching_files(month_path, ("*.tif", "*.TIF"))
    invalid = [f.name for f in tif_files if not S2_PATTERN.match(f.name)]
    return len(tif_files), invalid[:5]


def _validate_s5_names(month_path: Path) -> tuple[int, int, list[str]]:
    tif_files = _count_matching_files(month_path, ("*.tif", "*.TIF"))
    nc_files = _count_matching_files(month_path, ("*.nc", "*.NC"))
    all_files = tif_files + nc_files
    invalid = [f.name for f in all_files if not S5_PATTERN.match(f.name)]
    return len(tif_files), len(nc_files), invalid[:5]


def verify_data_structure() -> bool:
    """
    Verify that satellite data is properly organized.

    Expected structure:

    data/
    ├── sentinel_2/
    │   ├── july/
    │   ├── aug/
    │   ├── sep/
    │   ├── oct/
    │   ├── nov/
    │   └── dec/
    └── sentinel_5/
        ├── NO2/
        │   ├── july/
        │   ├── aug/
        │   ├── sep/
        │   ├── oct/
        │   ├── nov/
        │   └── dec/
        ├── SO2/
        │   ├── july/
        │   ├── aug/
        │   ├── sep/
        │   ├── oct/
        │   ├── nov/
        │   └── dec/
        └── CO/
            ├── july/
            ├── aug/
            ├── sep/
            ├── oct/
            ├── nov/
            └── dec/
    """
    print("=" * 70)
    print("SustainMine Satellite Data Verification (Windows)")
    print("=" * 70)

    base_path = Path.cwd()
    data_path = base_path / "data"

    print(f"\nCurrent directory: {base_path}")
    print(f"Looking for data in: {data_path}")

    if not data_path.exists():
        print("\n❌ ERROR: 'data' folder not found!")
        return False

    print(f"\n✓ Found data folder: {data_path}")

    sentinel2_path = data_path / "sentinel_2"
    sentinel5_path = data_path / "sentinel_5"
    all_checks_passed = True

    print("\n--- Checking Sentinel-2 Data ---")
    if not sentinel2_path.exists():
        print(f"❌ Missing: {sentinel2_path}")
        all_checks_passed = False
    else:
        print(f"✓ Found: {sentinel2_path}")
        for month in MONTHS:
            month_path = sentinel2_path / month
            if month_path.exists():
                total_files = len(list(month_path.glob("*")))
                tif_count, invalid = _validate_s2_names(month_path)
                print(f"  ✓ {month:8s}: {tif_count} .tif files (total: {total_files} files)")
                if invalid:
                    print(f"      ⚠ invalid S2 names: {', '.join(invalid)}")
                    all_checks_passed = False
            else:
                print(f"  ⚠ {month:8s}: folder not found")
                all_checks_passed = False

    print("\n--- Checking Sentinel-5P Data ---")
    if not sentinel5_path.exists():
        print(f"❌ Missing: {sentinel5_path}")
        all_checks_passed = False
    else:
        print(f"✓ Found: {sentinel5_path}")
        for gas in GASES:
            gas_path = sentinel5_path / gas
            if not gas_path.exists():
                print(f"  ❌ {gas:5s}: folder not found")
                all_checks_passed = False
                continue

            print(f"  ✓ {gas:5s}:")
            for month in MONTHS:
                month_path = gas_path / month
                if month_path.exists():
                    total_files = len(list(month_path.glob("*")))
                    tif_count, nc_count, invalid = _validate_s5_names(month_path)
                    print(
                        f"      ✓ {month:8s}: {tif_count} .tif files, "
                        f"{nc_count} .nc files (total: {total_files})"
                    )
                    if invalid:
                        print(f"         ⚠ invalid S5 names: {', '.join(invalid)}")
                        all_checks_passed = False
                else:
                    print(f"      ⚠ {month:8s}: folder not found")
                    all_checks_passed = False

    metadata = {
        "data_location": str(data_path),
        "structure": {
            "sentinel_2": {
                "path": str(sentinel2_path),
                "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
                "months": MONTHS,
                "format": "GeoTIFF (.tif)",
                "filename_pattern": "S2_YYYY-MM-DD.tif",
                "expected_shape": "(6, H, W)"
            },
            "sentinel_5": {
                "path": str(sentinel5_path),
                "products": GASES,
                "months": MONTHS,
                "format": "GeoTIFF (.tif) or NetCDF (.nc)",
                "filename_pattern": "S5P_<GAS>_[Daily_]YYYY-MM-DD.(tif|nc)",
                "usage": "numerical features per gas"
            }
        }
    }

    metadata_file = base_path / "data_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata to: {metadata_file}")
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ DATA STRUCTURE VERIFIED!")
    else:
        print("⚠ SOME ISSUES FOUND - Please review above")
    print("=" * 70)
    return all_checks_passed


def create_sample_config() -> Path:
    config = {
        "paths": {
            "sensor_data": "sensor_data_cleaned.csv",
            "sentinel_2": "data/sentinel_2",
            "sentinel_5": "data/sentinel_5",
            "output_dir": "outputs",
            "checkpoints": "checkpoints",
            "reports": "reports"
        },
        "sentinel_2": {
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "target_size": [224, 224],
            "file_pattern": "S2_*.tif"
        },
        "sentinel_5": {
            "gases": GASES,
            "months": MONTHS,
            "file_pattern": "S5P_*_[Daily_]*.(tif|nc)",
            "representation": "numerical"
        },
        "model": {
            "img_size": 224,
            "patch_size": 16,
            "in_channels": 6,
            "s5_dim": 3,
            "sensor_dim": 9,
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 6,
            "num_classes": 3,
            "num_forecast_steps": 3,
            "num_pollutants": 6
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.0001,
            "num_epochs": 20,
            "device": "cuda"
        }
    }

    config_file = Path.cwd() / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Created sample configuration: {config_file}")
    return config_file


def list_available_dates() -> None:
    data_path = Path.cwd() / "data" / "sentinel_2"
    if not data_path.exists():
        print("\n⚠ Cannot list dates - sentinel_2 folder not found")
        return

    print("\n--- Available Sentinel-2 Dates ---")
    all_files: list[Path] = []
    for month in MONTHS:
        month_path = data_path / month
        if month_path.exists():
            tif_files = _count_matching_files(month_path, ("*.tif", "*.TIF"))
            if tif_files:
                print(f"\n{month.upper()}:")
                for file in sorted(tif_files)[:5]:
                    print(f"  - {file.name}")
                if len(tif_files) > 5:
                    print(f"  ... and {len(tif_files) - 5} more files")
                all_files.extend(tif_files)

    print(f"\n✓ Total Sentinel-2 files found: {len(all_files)}")


def check_dependencies() -> None:
    print("\n--- Checking Python Dependencies ---")
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "rasterio": "rasterio",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "pillow": "PIL"
    }
    missing: list[str] = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\n✓ All required packages installed!")


def main() -> None:
    print(
        """
╔═══════════════════════════════════════════════════════════════════╗
║                    SustainMine Data Setup                        ║
║                      Windows Version                             ║
╚═══════════════════════════════════════════════════════════════════╝
        """
    )

    check_dependencies()
    print("\n")
    data_ok = verify_data_structure()

    if data_ok:
        list_available_dates()
        create_sample_config()
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review data_metadata.json for your data structure")
        print("  2. Check config.json for model/training settings")
        print("  3. Run: python sustainmine_pipeline_v2_fixed.py")
        print("  4. Train: python train_sustainmine_v2_fixed.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("SETUP INCOMPLETE")
        print("=" * 70)
        print("\nPlease organize your data as shown above and run this script again.")
        print("=" * 70)


if __name__ == "__main__":
    main()
