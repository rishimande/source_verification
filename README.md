# Source Verification Analysis

Analysis tool for solar device data verification with battery dip detection and sinusoidality checks.

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Process all files in device_data folder (default):
```bash
python source_verification.py
```

### Process a specific file:
```bash
python source_verification.py --input device_data/bquxjob_22a0183e_197ec64eb48\ copy.csv
```

### Process files from a custom directory:
```bash
python source_verification.py --input-dir custom_data_folder
```

### Custom output directory and time offset:
```bash
python source_verification.py --output-dir results --time-offset -6
```

## Arguments

- `--input, -i`: Single input CSV file path (optional)
- `--input-dir, -d`: Directory containing CSV files (default: `device_data`)
- `--output-dir, -o`: Output directory (default: `output`)
- `--time-offset, -t`: Local time offset in hours (default: -6)

## Output Structure

For each processed device file, outputs are organized by **device_id** (extracted from CSV data):

```
output/
  └── device_<device_id>/
      ├── daily_plot/
      │   └── observed_power_YYYY-MM-DD.png
      ├── seven_day_envelopes/
      └── 7_day_compressed_sin_fit/
          ├── compressed_fit_*.png
          └── daily_metrics.csv
```

### Device Identification

- **Output directories** are named using `device_<device_id>` where `device_id` is extracted from the CSV file's `device_id` column
- **daily_metrics.csv** includes:
  - `device_id`: Numeric device ID from CSV data
  - `device_file`: Original filename for traceability

This allows easy identification and comparison of results across different devices.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Features

- Automatic processing of multiple device data files
- Device identification based on CSV `device_id` column
- Daily power plots with battery dip detection
- 7-day envelope analysis with half-sine fitting
- Comprehensive diagnostics including sinusoidality checks
- Per-day metrics exported to CSV

