# SustainMine Windows Guide (Rewritten)

This guide matches your **real project setup**.

## 1) Your real data layout

Your project folder should look like this:

```text
your_project_folder/
├── data/
│   ├── sentinel_2/
│   │   ├── july/
│   │   │   ├── S2_2025-07-01.tif
│   │   │   ├── S2_2025-07-02.tif
│   │   │   └── ...
│   │   ├── aug/
│   │   ├── sep/
│   │   ├── oct/
│   │   ├── nov/
│   │   └── dec/
│   └── sentinel_5/
│       ├── NO2/
│       │   ├── july/
│       │   │   ├── S5P_NO2_Daily_2025-07-01.tif
│       │   │   ├── S5P_NO2_Daily_2025-07-02.tif
│       │   │   └── ...
│       │   ├── aug/
│       │   ├── sep/
│       │   ├── oct/
│       │   ├── nov/
│       │   └── dec/
│       ├── SO2/
│       │   ├── july/
│       │   ├── aug/
│       │   ├── sep/
│       │   ├── oct/
│       │   ├── nov/
│       │   └── dec/
│       └── CO/
│           ├── july/
│           ├── aug/
│           ├── sep/
│           ├── oct/
│           ├── nov/
│           └── dec/
├── sensor_data_cleaned.csv
├── setup_satellite_data_windows_fixed.py
├── sustainmine_pipeline_v2_fixed.py
├── sustainmine_model_v2_fixed.py
├── train_sustainmine_v2_fixed.py
└── requirements.txt
```

## 2) Important clarification

### Sentinel-2
- Stored as **multi-band GeoTIFF** files.
- Example filename:

```text
S2_2025-08-02.tif
```

- These are used as the **image input** to the model.

### Sentinel-5
- Files are stored under:
  - `NO2/<month>/...`
  - `SO2/<month>/...`
  - `CO/<month>/...`
- Example filename:

```text
S5P_NO2_Daily_2025-12-01.tif
```

- In your fixed version, **Sentinel-5 is not treated as an image input**.
- It is converted into **numerical values**:
  - `NO2`
  - `SO2`
  - `CO`

So the model uses:
- **S2** → image input
- **S5** → numerical input
- **sensor data** → numerical input

## 3) Open PowerShell in your project folder

Example:

```powershell
cd C:\Users\leena\sustainmine
```

## 4) Create and activate a virtual environment

### Create it

```powershell
python -m venv venv
```

### Activate it in PowerShell

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks it, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
venv\Scripts\Activate.ps1
```

## 5) Install dependencies

If you have a `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

If not, install the important packages directly:

```powershell
pip install numpy pandas scipy matplotlib pillow rasterio
```

### Install PyTorch

For CPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU with compatible CUDA:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 6) Run the setup checker

Run:

```powershell
python setup_satellite_data_windows_fixed.py
```

This script will:
- verify `data/sentinel_2`
- verify `data/sentinel_5`
- verify month folders
- verify file naming patterns
- create metadata/config files

## 7) Run the data pipeline

Run:

```powershell
python sustainmine_pipeline_v2_fixed.py
```

This stage should:
- find matching dates across:
  - `S2`
  - `NO2`
  - `SO2`
  - `CO`
  - sensor CSV
- build samples for training

## 8) What the model expects

The fixed setup is based on this idea:

### Input 1: Sentinel-2 image
- multi-band image
- shape like:

```text
(6, H, W)
```

### Input 2: Sentinel-5 numerical vector
- one value per gas
- shape like:

```text
(3,)
```

meaning:

```text
[NO2, SO2, CO]
```

### Input 3: sensor features
- numerical vector from your CSV

## 9) Train the model

Run:

```powershell
python train_sustainmine_v2_fixed.py
```

If you later add command-line arguments, you can use things like:

```powershell
python train_sustainmine_v2_fixed.py --epochs 20 --batch_size 8
```

## 10) The main confusion that was fixed

Your old guide mixed up these things:

### Old assumption
- `sentinel_5/NO2/file.tif`

### Real structure
- `sentinel_5/NO2/july/file.tif`

Also:

### Old example names
- `NO2_20250701.tif`

### Your real names
- `S5P_NO2_Daily_2025-12-01.tif`

And:

### Old idea
- use S5 like image data

### Your fixed setup
- use S5 as numerical features

## 11) Recommended run order

Always run in this order:

```powershell
python setup_satellite_data_windows_fixed.py
python sustainmine_pipeline_v2_fixed.py
python train_sustainmine_v2_fixed.py
```

## 12) Quick checklist

Before running training, make sure:

- `data/sentinel_2` exists
- `data/sentinel_5` exists
- S5 has this structure:
  - `NO2/july/...`
  - `SO2/july/...`
  - `CO/july/...`
- S2 files look like:

```text
S2_2025-08-02.tif
```

- S5 files look like:

```text
S5P_NO2_Daily_2025-12-01.tif
```

- `sensor_data_cleaned.csv` exists
- virtual environment is activated
- required packages are installed

## 13) If something fails

### If activation fails in PowerShell
Use:

```powershell
venv\Scripts\Activate.ps1
```

not:

```bash
source venv/bin/activate
```

because that is for Linux/macOS.

### If `rasterio` fails to install
Try:

```powershell
pip install rasterio
```

and if it still fails, tell me the exact error.

### If training fails
The most common causes will be:
- date mismatch between S2 and S5
- missing sensor CSV columns
- shape mismatch between dataset and model

## 14) Simple summary

Your project now works like this:

- **Sentinel-2** gives the spatial image information.
- **Sentinel-5** gives numerical pollution features.
- **sensor data** gives ground measurements.
- the pipeline matches them by **date**.
- the model learns from all of them together.
