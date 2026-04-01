from pathlib import Path
import re
import pandas as pd

MONTHS = ["July", "Aug", "Sep", "Oct", "Nov", "Dec"]
GASES = ["NO2", "SO2", "CO"]

def extract_s2_date(name: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None

def extract_s5_date(name: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None

base = Path.cwd()
s2_base = base / "data" / "sentinel_2"
s5_base = base / "data" / "sentinel_5"
sensor_csv = base / "sensor_data_cleaned.csv"

# sensor dates
sensor_df = pd.read_csv(sensor_csv)
sensor_df["Date"] = pd.to_datetime(sensor_df["Date"]).dt.strftime("%Y-%m-%d")
sensor_dates = set(sensor_df["Date"])

# s2 dates
s2_dates = set()
for month in MONTHS:
    month_path = s2_base / month
    if month_path.exists():
        for f in month_path.glob("*.tif"):
            d = extract_s2_date(f.stem)
            if d:
                s2_dates.add(d)

# s5 dates by gas
s5_dates = {}
for gas in GASES:
    gas_set = set()
    for month in MONTHS:
        month_path = s5_base / gas / month
        if month_path.exists():
            for f in list(month_path.glob("*.tif")) + list(month_path.glob("*.nc")):
                d = extract_s5_date(f.stem)
                if d:
                    gas_set.add(d)
    s5_dates[gas] = gas_set

shared = s2_dates & s5_dates["NO2"] & s5_dates["SO2"] & s5_dates["CO"] & sensor_dates

print("sensor rows:", len(sensor_df))
print("sensor dates:", len(sensor_dates))
print("s2 dates:", len(s2_dates))
print("no2 dates:", len(s5_dates["NO2"]))
print("so2 dates:", len(s5_dates["SO2"]))
print("co dates:", len(s5_dates["CO"]))
print("shared dates:", len(shared))
print("first shared:", sorted(shared)[:10])

print("\nexample s2 only:", sorted(s2_dates)[:5])
print("example no2 only:", sorted(s5_dates["NO2"])[:5])
print("example so2 only:", sorted(s5_dates["SO2"])[:5])
print("example co only:", sorted(s5_dates["CO"])[:5])
print("example sensor only:", sorted(sensor_dates)[:5])