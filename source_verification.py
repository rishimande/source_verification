import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for local execution
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import sys
from pathlib import Path
import argparse

# --- Half-sine model representing solar generation curve ---
def half_wave_sine(t, k1, k2):
    return k1 * np.sin((2 * np.pi * t / 24) - k2)

# --- Custom Maximum Likelihood Estimator (two-sided, weighted) ---
def custom_mle(params, t, p_obs, lambda_=1.0, underweight=0.2):
    k1, k2 = params
    p_model = half_wave_sine(t, k1, k2)
    residuals = p_model - p_obs
    pos_resid = residuals[residuals > 0]
    neg_resid = -residuals[residuals < 0]
    log_likelihood = 0
    if len(pos_resid) > 0:
        log_likelihood += np.sum(np.log(lambda_) - lambda_ * pos_resid)
    if len(neg_resid) > 0:
        log_likelihood += underweight * np.sum(np.log(lambda_) - lambda_ * neg_resid)
    return -log_likelihood

# --- Utility: Find local maxima moving left/right from a point ---
def find_local_maxima(series, start_idx, direction):
    indices = []
    i = start_idx
    while 0 <= i < len(series):
        window = series[max(i - 1, 0):min(i + 2, len(series))]
        if series[i] == window.max():
            indices.append(i)
            i += 1 if direction == "right" else -1
        else:
            i += 1 if direction == "right" else -1
    return indices

# --- Chain local maxima with smallest slope difference to form envelope ---
def least_slope_chain(series, time, start_idx, direction, maxima_indices):
    selected = [start_idx]
    current_idx = start_idx
    while True:
        candidates = [i for i in maxima_indices if (i < current_idx if direction == "left" else i > current_idx)]
        if not candidates:
            break
        slopes = [(abs(series[i] - series[current_idx]) / abs(time[i] - time[current_idx] + 1e-6), i) for i in candidates]
        next_idx = min(slopes, key=lambda x: x[0])[1]
        selected.append(next_idx)
        current_idx = next_idx
    return selected

# -------------------------------------------------------------------------
# UNIFIED: Battery full dip detection for both daily (datetime) and envelope (hour floats)
# -------------------------------------------------------------------------
def detect_battery_full_dip_unified(times, power, min_drop_ratio=0.7, min_duration_minutes=30,
                                    tolerance_fraction=0.2, min_flat_frac=0.8):
    """
    Detect a 'battery full' dip after the daily peak:
      - times: array-like of datetimes (daily) OR hour floats (7-day envelope)
      - power: array-like of power values (same length as times)
    Returns:
      - for datetime inputs: pandas.Timestamp of dip
      - for hour-floats inputs: float hour of dip
      - None if not detected
    """
    if power is None or len(power) == 0:
        return None

    times_arr = np.asarray(times)
    power_arr = np.asarray(power, dtype=float)

    max_idx = int(np.argmax(power_arr))
    max_power = float(power_arr[max_idx])
    if max_power < 10:
        return None

    # Determine mode: datetime or hour floats
    is_datetime_mode = np.issubdtype(times_arr.dtype, np.datetime64)

    drop_threshold = max_power * (1 - min_drop_ratio)
    post_max_power = power_arr[max_idx:]
    drop_rel_indices = np.where(post_max_power < drop_threshold)[0]
    if len(drop_rel_indices) == 0:
        return None

    tol = max_power * tolerance_fraction

    if is_datetime_mode:
        # datetime: use real-time duration window
        for rel_idx in drop_rel_indices:
            dip_idx = max_idx + rel_idx
            dip_time = pd.to_datetime(times_arr[dip_idx])
            end_time = dip_time + pd.Timedelta(minutes=min_duration_minutes)
            # Select segment between dip and end_time
            mask = (times_arr >= dip_time) & (times_arr <= end_time)
            segment = power_arr[mask]
            if len(segment) == 0:
                continue
            below_count = int(np.sum(segment < (drop_threshold + tol)))
            fraction_below = below_count / len(segment)
            if fraction_below >= min_flat_frac:
                return dip_time
        return None
    else:
        # hour floats: approximate by number of bins needed
        if len(times_arr) > 1:
            approx_step_h = float(np.median(np.diff(times_arr)))
        else:
            approx_step_h = 0.25  # default to 15 minutes

        required_bins = max(1, int(round((min_duration_minutes / 60.0) / max(approx_step_h, 1e-6))))
        for rel_idx in drop_rel_indices:
            dip_idx = max_idx + rel_idx
            end_idx = min(len(power_arr) - 1, dip_idx + required_bins)
            segment = power_arr[dip_idx:end_idx + 1]
            if len(segment) == 0:
                continue
            below_count = int(np.sum(segment < (drop_threshold + tol)))
            fraction_below = below_count / len(segment)
            if fraction_below >= min_flat_frac:
                return float(times_arr[dip_idx])
        return None

def process_device_file(input_path, output_base_dir, local_time_offset=-6):
    """
    Process a single device data file and generate all outputs.
    
    Args:
        input_path: Path to input CSV file
        output_base_dir: Base directory for outputs
        local_time_offset: Local time offset in hours
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # --- Load Data ---
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    
    # Validate required columns
    required_cols = ['heartbeat_ts', 'mv_in', 'ma_in', 'device_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        return
    
    # Extract device_id from data (use the most common value in case of mixed data)
    device_id = int(df['device_id'].mode()[0]) if len(df['device_id'].mode()) > 0 else int(df['device_id'].iloc[0])
    device_name = f"device_{device_id}"
    
    print(f"📱 Device ID: {device_id}")
    print(f"📁 Output directory: {device_name}")
    
    # Create device-specific output directory using device_id
    device_output_dir = output_base_dir / device_name
    
    daily_plot_dir = device_output_dir / "daily_plot"
    seven_day_dir = device_output_dir / "seven_day_envelopes"
    compressed_fit_dir = device_output_dir / "7_day_compressed_sin_fit"
    
    for d in [daily_plot_dir, seven_day_dir, compressed_fit_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    df['timestamp'] = pd.to_datetime(df['heartbeat_ts'].str.replace(" UTC", ""), utc=True)
    df.set_index('timestamp', inplace=True)
    df['voltage'] = df['mv_in'] / 1000
    df['current'] = df['ma_in'] / 1000
    df['power_W_raw'] = df['voltage'] * df['current']
    
    # --- Time Adjustments ---
    LOCAL_TIME_OFFSET = local_time_offset
    df['local_hour'] = (df.index.hour + df.index.minute / 60 + df.index.second / 3600 + LOCAL_TIME_OFFSET) % 24
    df['date'] = df.index.date
    
    # -------------------------------------------------------------------------
    # DAILY PLOTTING (Simplified): Only observed power vs local time (no daily sine fit)
    # -------------------------------------------------------------------------
    print(f"📊 Generating daily plots...")
    daily_count = 0
    for date, group in df.groupby(df.index.date):
        # Daylight subset for a clean plot; keeps figure uncluttered
        group_daylight = group[(group['local_hour'] >= 6) & (group['local_hour'] <= 18)].sort_index()
        if len(group_daylight) < 5:
            continue
        
        daily_count += 1
        
        # Detect battery dip on this day (daylight segment) using unified detector
        dip_time = detect_battery_full_dip_unified(group_daylight.index.values,
                                                   group_daylight['power_W_raw'].values,
                                                   min_drop_ratio=0.7, min_duration_minutes=30,
                                                   tolerance_fraction=0.2, min_flat_frac=0.8)
        
        # Plot observed power against LOCAL time (apply the configured offset)
        local_times = group_daylight.index + pd.Timedelta(hours=LOCAL_TIME_OFFSET)
        
        plt.figure(figsize=(10, 4))
        plt.plot(local_times, group_daylight['power_W_raw'].values, '.', alpha=0.6, label='Observed Power')
        if dip_time is not None:
            plt.axvline(dip_time + pd.Timedelta(hours=LOCAL_TIME_OFFSET), linestyle='--', label='Battery Dip')
        plt.title(f"Observed Power (Local Time) – {date}")
        plt.xlabel("Local Time")
        plt.ylabel("Power (W)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = daily_plot_dir / f"observed_power_{date}.png"
        plt.savefig(out_path)
        plt.close()
    
    print(f"✅ Generated {daily_count} daily plots")
    
    # --- 7-Day Envelope Plotting with Half-Sine Fitting & Per-Day Diagnostics ---
    print(f"📈 Generating 7-day envelope fits and per-day diagnostics...")
    bin_edges = np.arange(6, 18.01, 0.25)
    df_daylight = df[(df['local_hour'] >= 6) & (df['local_hour'] <= 18)].copy()
    # Create categorical once, and derive bin centers (labels)
    time_bins = pd.cut(df_daylight['local_hour'], bins=bin_edges)
    bin_labels = [(c.left + c.right) / 2 for c in time_bins.cat.categories]
    df_daylight['time_bin'] = time_bins
    df_daylight['date'] = df_daylight.index.date
    all_dates = sorted(df_daylight['date'].unique())
    window_size = 7
    
    print(f"📅 Found {len(all_dates)} unique dates with daylight data")
    if len(all_dates) > 0:
        print(f"   Date range: {all_dates[0]} to {all_dates[-1]}")
    
    # Collect per-day diagnostics across all windows
    daily_metrics = []
    
    windows_processed = 0
    for i in range(0, len(all_dates), window_size):
        window_dates = all_dates[i:i + window_size]
        if len(window_dates) < window_size:
            print(f"⚠️  Skipping incomplete window: {len(window_dates)} days (need {window_size} consecutive days)")
            if len(window_dates) > 0:
                print(f"   Window dates: {window_dates[0]} to {window_dates[-1]}")
            continue
        
        label = f"{window_dates[0]} to {window_dates[-1]}"
        window_df = df_daylight[df_daylight['date'].isin(window_dates)].copy()
        
        # Build 7-day envelope from bin maxima
        bin_max = window_df.groupby('time_bin')['power_W_raw'].max()
        # Align index to bin centers
        bin_max.index = bin_labels
        power_env = bin_max.values
        t_hours_env = np.array(bin_max.index, dtype=float)
        
        if len(power_env) < 10 or np.max(power_env) < 5:
            print(f"⚠️  Skipping window {label}: insufficient data (power_env len={len(power_env)}, max={np.max(power_env) if len(power_env) > 0 else 0:.2f})")
            continue
        
        windows_processed += 1
        
        # Peak selection (unchanged)
        global_max_idx = np.argmax(power_env)
        left_max = find_local_maxima(power_env, global_max_idx, "left")
        right_max = find_local_maxima(power_env, global_max_idx, "right")
        all_maxima = sorted(set(left_max + right_max + [global_max_idx]))
        left_chain = least_slope_chain(power_env, t_hours_env, global_max_idx, "left", all_maxima)
        right_chain = least_slope_chain(power_env, t_hours_env, global_max_idx, "right", all_maxima)
        final_indices = sorted(set(left_chain + right_chain))
        t_fit = t_hours_env[final_indices]
        p_fit = power_env[final_indices]
        mask = p_fit > 2
        t_fit = t_fit[mask]
        p_fit = p_fit[mask]
        if len(p_fit) < 3:
            print(f"⚠️  Skipping window {label}: insufficient filtered peaks (need 3+, found {len(p_fit)})")
            continue
        
        try:
            # Fit the 7-day "ideal" model on filtered peaks
            result = minimize(custom_mle, [p_fit.max(), np.pi / 2],
                              args=(t_fit, p_fit),
                              bounds=[(0, 500), (0, 2 * np.pi)])
            k1_fit, k2_fit = result.x
            
            # Dense curve for plotting and reference peak within daylight range
            t_dense = np.linspace(t_hours_env.min(), t_hours_env.max(), 500)
            ideal_dense = half_wave_sine(t_dense, k1_fit, k2_fit)
            peak_ref_env = float(np.max(ideal_dense))
            peak_ref_env = max(peak_ref_env, 1e-6)  # avoid divide-by-zero
            
            # ---------- NEW: Sinusoidality check on 7-day compressed envelope ----------
            # Compare envelope samples directly against the model evaluated at those times
            p_model_env = half_wave_sine(t_hours_env, k1_fit, k2_fit)
            rmse_env = float(np.sqrt(np.mean((power_env - p_model_env) ** 2)))
            nrmse_env = rmse_env / peak_ref_env
            # Threshold set to 30% to accommodate deviations caused by battery dip events,
            # which occur in real life when the battery gets completely charged and stops
            # accepting solar power, causing a drop in observed power generation
            sinusoidal_env_pass = nrmse_env <= 0.30
            
            # Detect and mark battery dip on the 7-day envelope
            dip_hour_env = detect_battery_full_dip_unified(t_hours_env, power_env,
                                                           min_drop_ratio=0.7, min_duration_minutes=30,
                                                           tolerance_fraction=0.2, min_flat_frac=0.8)
            
            # Plot the envelope + fitted sine
            plt.figure(figsize=(10, 4))
            plt.plot(t_hours_env, power_env, 'r-', marker='o', label='7-Day Max Envelope')
            plt.plot(t_fit, p_fit, 'go', label='Filtered Peaks')
            plt.plot(t_dense, ideal_dense, label='Half-Sine Fit (7-day)')
            if dip_hour_env is not None:
                plt.axvline(dip_hour_env, linestyle='--', label='Battery Dip')
            plt.title(f"7-Day Envelope + Fit: {label}\nSinusoidality NRMSE={nrmse_env:.3%} | Pass≤30%={sinusoidal_env_pass}")
            plt.xlabel("Local Time (Hour)")
            plt.ylabel("Power (W)")
            plt.grid(True)
            plt.xticks(np.arange(6, 19, 1))
            plt.ylim(bottom=0)
            plt.legend()
            plt.tight_layout()
            fname = compressed_fit_dir / f"compressed_fit_{window_dates[0]}_to_{window_dates[-1]}.png"
            plt.savefig(fname)
            plt.close()
            print(f"📸 Saved envelope + fit: {fname.name}")
            
            # ---------- Per-Day Diagnostics (sinusoidality fields now window-level) ----------
            for date in window_dates:
                day_mask = (df['date'] == date)
                day_df = df.loc[day_mask].copy()
                day_df_daylight = day_df[(day_df['local_hour'] >= 6) & (day_df['local_hour'] <= 18)]
                
                if len(day_df_daylight) < 5:
                    daily_metrics.append({
                        'device_id': device_id,
                        'device_file': input_path.name,
                        'date': str(date),
                        'window_start': str(window_dates[0]),
                        'window_end': str(window_dates[-1]),
                        'k1_window': float(k1_fit),
                        'k2_window': float(k2_fit),
                        'detected_battery_dip': np.nan,
                        # Window-level sinusoidality (requested change)
                        'sinusoidality_nrmse': float(nrmse_env),
                        'sinusoidality_pass_<=30pct': bool(sinusoidal_env_pass),
                        # Explicit env columns for clarity
                        'sinusoidality_env_nrmse': float(nrmse_env),
                        'sinusoidality_env_pass_<=30pct': bool(sinusoidal_env_pass),
                        # Daily-specific checks retained
                        'frac_points_obs_gt_model_plus_5pct': np.nan,
                        'negative_noise_pass_<=5pct_violations': False,
                        'fraction_low_power_outside_daylight': np.nan,
                        'no_generation_outside_daylight_pass_>=95pct': False,
                        'lambda_day': np.nan
                    })
                    continue
                
                t_day = day_df_daylight['local_hour'].values.astype(float)
                p_day = day_df_daylight['power_W_raw'].values.astype(float)
                p_model_day = half_wave_sine(t_day, k1_fit, k2_fit)
                
                # Negative-only noise check (allow 5% violations over +5% margin of ideal peak)
                margin = 0.05 * peak_ref_env
                resid_obs_minus_model = p_day - p_model_day
                frac_positive_resid = float(np.mean(resid_obs_minus_model > margin))
                negative_noise_pass = frac_positive_resid <= 0.05
                
                # No-generation outside daylight (per day)
                outside_samples = day_df[(day_df['local_hour'] < 5) | (day_df['local_hour'] > 19)]['power_W_raw'].values
                day_max = float(np.max(day_df['power_W_raw'])) if len(day_df) else 0.0
                low_thresh = max(0.03 * max(day_max, 1e-6), 50.0)  # absolute floor
                frac_low_outside = np.nan
                if len(outside_samples) > 0:
                    frac_low_outside = float(np.mean(outside_samples <= low_thresh))
                    no_generation_pass = (frac_low_outside >= 0.95) if not np.isnan(frac_low_outside) else False
                else:
                    # No measurements outside daylight => assume pass
                    no_generation_pass = True
                
                # Lambda estimate on (model - obs) positive residuals (one-sided)
                residuals_day = p_model_day - p_day
                if np.any(residuals_day > 0):
                    lambda_day = 1 / np.mean(residuals_day[residuals_day > 0])
                else:
                    lambda_day = np.nan
                
                # Daily battery dip detection (use daylight samples)
                dip_time_day = detect_battery_full_dip_unified(day_df_daylight.index.values,
                                                               day_df_daylight['power_W_raw'].values,
                                                               min_drop_ratio=0.7, min_duration_minutes=30,
                                                               tolerance_fraction=0.2, min_flat_frac=0.8)
                dip_time_day_str = str(dip_time_day) if dip_time_day is not None else np.nan
                
                daily_metrics.append({
                    'device_id': device_id,
                    'device_file': input_path.name,
                    'date': str(date),
                    'window_start': str(window_dates[0]),
                    'window_end': str(window_dates[-1]),
                    'k1_window': float(k1_fit),
                    'k2_window': float(k2_fit),
                    'detected_battery_dip': dip_time_day_str,
                    # Window-level sinusoidality (requested change)
                    'sinusoidality_nrmse': float(nrmse_env),
                    'sinusoidality_pass_<=30pct': bool(sinusoidal_env_pass),
                    # Explicit env columns for clarity
                    'sinusoidality_env_nrmse': float(nrmse_env),
                    'sinusoidality_env_pass_<=30pct': bool(sinusoidal_env_pass),
                    # Daily-specific checks retained
                    'frac_points_obs_gt_model_plus_5pct': float(frac_positive_resid),
                    'negative_noise_pass_<=5pct_violations': bool(negative_noise_pass),
                    'fraction_low_power_outside_daylight': float(frac_low_outside) if not np.isnan(frac_low_outside) else np.nan,
                    'no_generation_outside_daylight_pass_>=95pct': bool(no_generation_pass),
                    'lambda_day': float(lambda_day) if not np.isnan(lambda_day) else np.nan
                })
        
        except Exception as e:
            print(f"❌ 7-day fit/diagnostics error for window {label}: {e}")
    
    # --- Save per-day diagnostics CSV ---
    daily_metrics_path = compressed_fit_dir / "daily_metrics.csv"
    if len(daily_metrics) > 0:
        pd.DataFrame(daily_metrics).to_csv(daily_metrics_path, index=False)
        print(f"✅ Saved per-day diagnostics CSV: {daily_metrics_path}")
    else:
        print(f"ℹ️  No daily diagnostics to save (insufficient data/windows).")
        print(f"   Processed {windows_processed} valid 7-day window(s) out of {len(all_dates)} total days")
        if len(all_dates) < window_size:
            print(f"   Reason: Only {len(all_dates)} day(s) with data, need at least {window_size} consecutive days")
    
    print(f"✅ Completed processing: {device_name} (Device ID: {device_id})")

def main():
    """Main execution function for local runs."""
    parser = argparse.ArgumentParser(
        description='Source verification analysis for device data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files in device_data folder
  python source_verification.py
  
  # Process specific file
  python source_verification.py --input device_data/file.csv
  
  # Process all files in custom folder
  python source_verification.py --input-dir custom_data_folder
  
  # Custom output directory and time offset
  python source_verification.py --output-dir results --time-offset -6
        """
    )
    parser.add_argument('--input', '-i',
                       type=str,
                       help='Single input CSV file path (if not provided, processes all in --input-dir)')
    parser.add_argument('--input-dir', '-d',
                       type=str,
                       default='device_data',
                       help='Directory containing CSV files to process (default: device_data)')
    parser.add_argument('--output-dir', '-o',
                       type=str,
                       default='output',
                       help='Output directory for plots and results (default: output)')
    parser.add_argument('--time-offset', '-t',
                       type=int,
                       default=-6,
                       help='Local time offset in hours (default: -6)')
    args = parser.parse_args()
    
    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent.absolute()
    
    # Determine input files
    input_files = []
    if args.input:
        # Single file specified
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = script_dir / input_path
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {input_path}")
            sys.exit(1)
        input_files = [input_path]
    else:
        # Process all CSV files in input directory
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = script_dir / input_dir
        
        if not input_dir.exists():
            print(f"❌ Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Find all CSV files
        input_files = sorted(input_dir.glob("*.csv"))
        if len(input_files) == 0:
            print(f"❌ Error: No CSV files found in: {input_dir}")
            sys.exit(1)
        
        print(f"📁 Found {len(input_files)} CSV file(s) in {input_dir}")
    
    # Set output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for input_file in input_files:
        try:
            process_device_file(input_file, output_dir, args.time_offset)
        except Exception as e:
            print(f"❌ Error processing {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ All processing complete! Output saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
