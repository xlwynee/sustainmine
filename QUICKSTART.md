# SustainMine Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.9+
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA 11.8+

### Step 1: Setup Environment (2 min)
```bash
# Download all files
# Extract to: ~/sustainmine/

cd ~/sustainmine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Satellite Data (Manual)
Visit these Google Drive links and download to `satellite_data/`:

**Sentinel-2** (Multispectral Imagery):
https://drive.google.com/drive/folders/15yN_kbDv615DukoMgJXPAVWreA5F-RwT

**Sentinel-5P** (Atmospheric Gases):
https://drive.google.com/drive/folders/1-X9de6Fkqw-2NNo3sSc2ObqTQ-yVzt8D

**OR use automated download:**
```bash
pip install gdown
gdown --folder 15yN_kbDv615DukoMgJXPAVWreA5F-RwT -O satellite_data/sentinel_2/
gdown --folder 1-X9de6Fkqw-2NNo3sSc2ObqTQ-yVzt8D -O satellite_data/sentinel_5/
```

### Step 3: Prepare Data (1 min)
```bash
# Setup directories
python setup_satellite_data.py

# Process sensor data
python sustainmine_pipeline.py
```

**Output:**
```
✓ Created directory structure at /home/claude/satellite_data
✓ Saved satellite metadata to satellite_metadata.json
✓ Data pipeline ready for model training
```

### Step 4: Train Model (30-60 min on GPU, 2-4 hours on CPU)
```bash
python train_sustainmine.py
```

**Monitor Progress:**
```
Epoch 1/20
----------------------------------------------------------
Train Loss: 1.2543 (Class: 1.0234, Forecast: 0.2309)
Val Loss: 1.1876 (Class: 0.9654, Forecast: 0.2222)
Val Accuracy: 0.6333
Val MAE: 1.4567
✓ Saved best model (loss: 1.1876)
```

**Checkpoints Saved:**
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/final_model.pth` - Final epoch
- `training_curves.png` - Visualization

### Step 5: Run Inference (< 5 sec)
```python
from inference_and_reporting import run_inference_pipeline

site_info = {
    'name': 'Al Murjan Mine',
    'location': 'Wadi Al-Dawasir, Riyadh Province'
}

predictions = run_inference_pipeline(
    satellite_image_path='satellite_data/sentinel_2/dec/20251205.tif',
    model_path='checkpoints/best_model.pth',
    site_info=site_info
)
```

**Output:**
```
============================================================
SustainMine Inference Pipeline
============================================================

1. Loading model...
2. Processing satellite imagery...
3. Running environmental analysis...
4. Generating automated report...

✓ Analysis complete!
✓ Report saved to reports/environmental_report_20251205_103000.json

============================================================
ENVIRONMENTAL STATUS SUMMARY
============================================================
Impact Level: Medium Impact
Confidence: 87.0%

3-Day Forecast Highlights:
  Day 1: 0 pollutant(s) may exceed limits
  Day 2: 0 pollutant(s) may exceed limits
  Day 3: 0 pollutant(s) may exceed limits
============================================================
```

## 📊 Understanding the Output

### Classification Result
```json
{
  "predicted_class": "Medium Impact",
  "confidence": 0.87,
  "probabilities": {
    "Low Impact": 0.08,
    "Medium Impact": 0.87,
    "High Impact": 0.05
  }
}
```

**Interpretation:**
- **87% confident** the site has **Medium Impact**
- Low probability of High Impact (5%)
- Classification based on multiple environmental indicators

### Forecast Result
```json
{
  "day_1": {
    "PM2.5": {
      "value": 8.3,
      "unit": "µg/m³",
      "ncec_limit": 35,
      "exceeds_limit": false
    },
    ...
  }
}
```

**Interpretation:**
- PM2.5 predicted at **8.3 µg/m³** for tomorrow
- **Well below** NCEC limit (35 µg/m³)
- ✓ **Compliant** with regulations

## 🎯 Common Use Cases

### Use Case 1: Daily Monitoring
```bash
# Run automatically every day at 6 AM
# (Add to crontab)
0 6 * * * cd ~/sustainmine && ./venv/bin/python inference_and_reporting.py
```

### Use Case 2: Alert System
```python
predictions = run_inference_pipeline(...)

if predictions['classification']['predicted_class'] == 'High Impact':
    send_alert_email(recipients=['environmental_manager@company.com'])
    
for day, forecasts in predictions['forecast'].items():
    for pollutant, data in forecasts.items():
        if data['exceeds_limit']:
            send_warning(f"{pollutant} may exceed limit on {day}")
```

### Use Case 3: Monthly Report
```python
# Generate monthly summary
reports = []
for date in get_month_dates('2025-12'):
    pred = run_inference_pipeline(f'data/{date}.tif', ...)
    reports.append(pred)

monthly_summary = generate_monthly_report(reports)
export_pdf(monthly_summary, 'monthly_report_dec2025.pdf')
```

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Problem: "CUDA out of memory"
**Solution:**
```python
# Edit train_sustainmine.py
batch_size = 4  # Reduce from 8
config['embed_dim'] = 384  # Reduce from 768
```

### Problem: "Satellite data not found"
**Solution:**
```bash
# Check directory structure
ls satellite_data/sentinel_2/july/  # Should show .tif files
ls satellite_data/sentinel_5/NO2/   # Should show gas data

# Verify file format
file satellite_data/sentinel_2/july/*.tif  # Should be GeoTIFF
```

### Problem: "Model accuracy is low"
**Solution:**
- Train for more epochs (50+ recommended)
- Ensure satellite data quality is good
- Check data preprocessing pipeline
- Verify sensor-satellite temporal alignment

## 📁 Project Files Overview

```
sustainmine/
├── 📊 DATA FILES
│   ├── sensor_data_cleaned.csv          # 158 days of measurements
│   ├── satellite_metadata.json          # Data alignment info
│   └── expected_data_format.json        # Format specifications
│
├── 🐍 PYTHON SCRIPTS
│   ├── sustainmine_pipeline.py          # Data preprocessing
│   ├── sustainmine_model.py             # Model architecture
│   ├── train_sustainmine.py             # Training script
│   ├── inference_and_reporting.py       # Inference engine
│   └── setup_satellite_data.py          # Data setup
│
├── 📚 DOCUMENTATION
│   ├── README.md                        # Complete guide
│   ├── QUICKSTART.md                    # This file
│   ├── COMPLETE_WORKFLOW_SUMMARY.md     # Technical details
│   └── DOWNLOAD_INSTRUCTIONS.txt        # Satellite data guide
│
└── ⚙️ CONFIGURATION
    └── requirements.txt                 # Dependencies
```

## ✅ Next Steps

1. **Customize for Your Site**
   - Update site coordinates in `site_info`
   - Adjust NCEC limits if different regulations apply
   - Modify forecast horizon (currently 3 days)

2. **Integrate with Existing Systems**
   - Connect to your GIS system
   - Export to regulatory reporting formats
   - Set up automated alerts

3. **Scale to Multiple Sites**
   - Batch process multiple mining locations
   - Compare sites side-by-side
   - Regional trend analysis

4. **Deploy to Production**
   - Set up REST API (FastAPI)
   - Create web dashboard (Dash/Plotly)
   - Schedule automated runs (cron/Airflow)

## 📞 Support

- **Questions?** Read the full README.md
- **Issues?** Check COMPLETE_WORKFLOW_SUMMARY.md
- **Contact:** gradproject016@gmail.com

---

**Congratulations! You now have a working AI environmental monitoring system.**

Next: Open README.md for advanced features and customization options.
