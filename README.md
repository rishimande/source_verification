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

## Algorithm Overview

The source verification algorithm analyzes solar device power generation data to verify that the observed power follows expected solar generation patterns. The core approach uses a **half-sine wave model** to represent ideal solar generation, then compares observed data against this model using multiple diagnostic tests.

### Core Mathematical Model

The algorithm uses a **half-wave sine function** to model solar power generation throughout the day:

```
P(t) = k₁ × sin((2π × t / 24) - k₂)
```

Where:
- `t` is the local time in hours (6:00 to 18:00, daylight hours)
- `k₁` is the amplitude parameter (peak power generation)
- `k₂` is the phase parameter (time offset for peak generation)

This model captures the expected sinusoidal pattern of solar generation: low at sunrise, peak around midday, and low at sunset.

### Processing Pipeline

#### 1. Data Preprocessing

- **Input**: CSV files with columns `heartbeat_ts`, `mv_in`, `ma_in`, `device_id`
- **Power Calculation**: `power_W = (mv_in / 1000) × (ma_in / 1000)`
- **Time Conversion**: UTC timestamps converted to local time using configurable offset
- **Daylight Filtering**: Data filtered to 6:00-18:00 local time (daylight hours)

#### 2. Daily Power Plotting

For each day with sufficient data:
- Plot observed power vs. local time
- Detect and mark battery full dip events (see Battery Dip Detection below)
- Generate visualization: `daily_plot/observed_power_YYYY-MM-DD.png`

#### 3. 7-Day Envelope Construction

The algorithm processes data in **7-day sliding windows**:

**Step 1: Time Binning**
- Divide daylight hours (6:00-18:00) into 0.25-hour bins (15-minute intervals)
- For each bin, find the **maximum power** across all 7 days
- This creates a "7-day envelope" representing the maximum observed power at each time of day

**Step 2: Peak Selection**
- Identify the global maximum in the envelope
- Find local maxima moving left and right from the global peak
- Chain maxima using **least-slope selection**: at each step, select the next maximum that minimizes the slope difference
- Filter peaks to remove low-power points (< 2W)

**Step 3: Model Fitting**
- Fit the half-sine model to the filtered peak points using **Custom Maximum Likelihood Estimation (MLE)**
- The MLE uses a two-sided, weighted approach:
  - **Positive residuals** (model > observed): Full weight in likelihood
  - **Negative residuals** (observed > model): Underweighted by factor of 0.2
  - This accounts for the fact that observed power should generally be below the envelope (which represents maximums)
- Optimization bounds: `k₁ ∈ [0, 500]`, `k₂ ∈ [0, 2π]`

#### 4. Battery Full Dip Detection

The algorithm detects when a battery becomes fully charged and stops accepting solar power, causing a sudden drop in observed generation.

**Detection Logic** (unified for both daily and envelope data):
1. Find the maximum power value in the dataset
2. Identify points after the peak where power drops below `70%` of maximum (`min_drop_ratio=0.7`)
3. For each candidate dip point:
   - Check if power remains low for at least `30 minutes` (`min_duration_minutes=30`)
   - Require that at least `80%` of points in the duration window are below the drop threshold plus tolerance (`min_flat_frac=0.8`, `tolerance_fraction=0.2`)
4. Return the first valid dip time, or `None` if no dip is detected

**Unified Implementation**: Works with both:
- **Datetime arrays** (daily plots): Uses actual time duration windows
- **Hour float arrays** (7-day envelope): Approximates duration using bin counts

#### 5. Diagnostic Tests

The algorithm performs multiple verification tests on each day:

##### Test 1: Sinusoidality Check (Window-Level)

**Purpose**: Verify that the 7-day envelope follows the expected sinusoidal pattern.

**Method**:
- Evaluate the fitted half-sine model at all envelope time points
- Calculate **Root Mean Square Error (RMSE)** between envelope and model
- Normalize by peak reference: `NRMSE = RMSE / peak_power`
- **Pass threshold**: `NRMSE ≤ 30%`

**Rationale**: The 30% threshold accommodates real-world deviations caused by battery dip events and other natural variations while still detecting significant non-sinusoidal patterns that might indicate data quality issues.

**Output**: 
- `sinusoidality_nrmse`: Normalized RMSE value
- `sinusoidality_pass_<=30pct`: Boolean pass/fail

##### Test 2: Negative Noise Check (Daily)

**Purpose**: Verify that observed daily power does not significantly exceed the model (accounting for measurement noise).

**Method**:
- For each daylight sample, calculate residual: `residual = observed - model`
- Define margin: `margin = 5% × peak_power`
- Count fraction of points where `residual > margin`
- **Pass threshold**: `fraction ≤ 5%`

**Rationale**: In normal operation, observed power should be at or below the model (which represents maximum envelope). Excessive positive residuals may indicate data quality issues or unexpected generation spikes.

**Output**:
- `frac_points_obs_gt_model_plus_5pct`: Fraction of violations
- `negative_noise_pass_<=5pct_violations`: Boolean pass/fail

##### Test 3: No Generation Outside Daylight (Daily)

**Purpose**: Verify that power generation is minimal outside daylight hours (5:00-19:00).

**Method**:
- Extract power samples outside daylight hours (before 5:00 or after 19:00)
- Define low-power threshold: `max(3% × daily_max, 50W)` (absolute floor of 50W)
- Calculate fraction of outside-daylight samples that are below threshold
- **Pass threshold**: `fraction ≥ 95%`

**Rationale**: Solar panels should generate minimal power outside daylight hours. High power at night may indicate data quality issues, incorrect timestamps, or non-solar power sources.

**Output**:
- `fraction_low_power_outside_daylight`: Fraction of low-power samples
- `no_generation_outside_daylight_pass_>=95pct`: Boolean pass/fail

##### Test 4: Lambda Parameter Estimation (Daily)

**Purpose**: Estimate the exponential decay parameter for positive residuals (model - observed).

**Method**:
- Calculate residuals: `residual = model - observed`
- Filter to positive residuals only (where model > observed)
- Estimate lambda: `λ = 1 / mean(positive_residuals)`

**Rationale**: This parameter characterizes the distribution of how observed power deviates below the model, useful for statistical modeling and anomaly detection.

**Output**:
- `lambda_day`: Estimated lambda parameter (NaN if no positive residuals)

### Output Metrics

Each day's diagnostics are saved to `daily_metrics.csv` with the following fields:

**Device Information**:
- `device_id`: Numeric device identifier
- `device_file`: Source CSV filename
- `date`: Date of analysis

**Window Information**:
- `window_start`: Start date of 7-day window
- `window_end`: End date of 7-day window
- `k1_window`: Fitted amplitude parameter (peak power)
- `k2_window`: Fitted phase parameter (time offset)

**Battery Dip Detection**:
- `detected_battery_dip`: Timestamp of detected dip (NaN if none)

**Sinusoidality Tests**:
- `sinusoidality_nrmse`: Normalized RMSE of envelope fit
- `sinusoidality_pass_<=30pct`: Pass/fail status
- `sinusoidality_env_nrmse`: Same as above (explicit envelope column)
- `sinusoidality_env_pass_<=30pct`: Same as above (explicit envelope column)

**Daily Diagnostic Tests**:
- `frac_points_obs_gt_model_plus_5pct`: Fraction of positive residual violations
- `negative_noise_pass_<=5pct_violations`: Negative noise test pass/fail
- `fraction_low_power_outside_daylight`: Fraction of low-power samples outside daylight
- `no_generation_outside_daylight_pass_>=95pct`: No-generation test pass/fail
- `lambda_day`: Estimated lambda parameter

### Algorithm Limitations and Considerations

1. **7-Day Window Requirement**: The algorithm requires at least 7 consecutive days of data to perform envelope analysis. Incomplete windows are skipped.

2. **Minimum Data Requirements**: 
   - Daily plots require at least 5 daylight samples
   - Envelope fitting requires at least 10 time bins and maximum power ≥ 5W
   - Peak filtering requires at least 3 valid peaks after filtering

3. **Time Zone Handling**: Local time offset is configurable (default: -6 hours). Ensure the offset matches your device's location.

4. **Battery Dip Tolerance**: The 30% sinusoidality threshold is intentionally lenient to accommodate real-world battery dip events, which are expected behavior when batteries reach full charge.

5. **Envelope vs. Daily Analysis**: 
   - Sinusoidality is evaluated at the **window level** (7-day envelope)
   - Other tests (negative noise, no-generation) are evaluated at the **daily level**
   - This hybrid approach balances robustness (envelope) with granularity (daily)

