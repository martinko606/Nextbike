import pandas as pd
import numpy as np
from scipy import stats
import os

# --- IMPORT EXISTING MODULES ---
# We import from your UNMODIFIED scripts to guarantee 100% consistency
import transport_analytics as ta
import representative_day as rd


# ==========================================
# 1. LOCAL STATISTICAL FUNCTION
# ==========================================
def calculate_p_value_locally(x, y, w):
    """
    Calculates Weighted Pearson r AND P-Value.
    Implemented locally to avoid altering existing modules.
    """
    x, y, w = np.asanyarray(x), np.asanyarray(y), np.asanyarray(w)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]

    if len(x) < 3: return np.nan, np.nan, 0
    sum_w = np.sum(w)
    if sum_w == 0: return np.nan, np.nan, 0

    mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
    xm, ym = x - mean_x, y - mean_y

    cov = np.sum(w * xm * ym) / sum_w
    var_x = np.sum(w * xm ** 2) / sum_w
    var_y = np.sum(w * ym ** 2) / sum_w

    if var_x <= 0 or var_y <= 0: return np.nan, np.nan, int(sum_w)

    r = cov / np.sqrt(var_x * var_y)
    r = max(min(r, 1.0), -1.0)

    n_eff = (sum_w ** 2) / np.sum(w ** 2)
    df = max(n_eff - 2, 1)

    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt(df / (1 - r ** 2))
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r, p, n_eff


# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================
def evaluate_representative_days():
    print("=== 📅 REPRESENTATIVE DAY SIGNIFICANCE CALCULATOR ===")
    print(f"Imported configuration from: {rd.__name__}.py\n")

    results = []
    ALPHA = 0.05

    # Iterate through the exact CITIES dictionary imported from representative_day
    for city_name, cfg in rd.CITIES.items():
        print(f"Processing: {city_name}...")

        # 1. Use the exact load function from representative_day
        df_t_raw, df_b_raw, df_w_raw = rd.load_data_via_ta(cfg)

        if df_t_raw is None:
            print("  [!] Data missing or corrupt. Skipping.\n")
            continue

        # 2. Standardize & Filter (Using ta)
        df_t = ta.standard_columns(df_t_raw)
        if 'station' in df_t.columns:
            df_t = df_t[df_t['station'].astype(str).str.contains(cfg['train_station'], case=False, na=False)]

        t_min = min(df_t['actual_arrival'].min(), df_b_raw['start_time'].min())
        t_max = max(df_t['actual_arrival'].max(), df_b_raw['end_time'].max())
        global_line = pd.date_range(t_min.floor('h'), t_max.ceil('h'), freq='1min')

        # 3. Generate Signals (Using ta)
        s_arr, s_dep = ta.generate_signals(df_t, global_line, cfg['params'])
        s_total_global = s_arr + s_dep

        # 4. Generate Bike Counts
        valid_st = [s.lower() for s in cfg['bike_stations']]
        starts = df_b_raw[df_b_raw['start_place'].str.lower().isin(valid_st)]
        ends = df_b_raw[df_b_raw['end_place'].str.lower().isin(valid_st)]

        c_rent = starts.set_index('start_time').resample('1min').size().reindex(global_line, fill_value=0)
        c_ret = ends.set_index('end_time').resample('1min').size().reindex(global_line, fill_value=0)
        c_total_global = c_rent + c_ret

        # 5. Weather Weights
        obs_weights = pd.Series(1.0, index=global_line)
        if df_w_raw is not None:
            try:
                df_w_raw.columns = [str(c).lower().strip() for c in df_w_raw.columns]
                t_col = next((c for c in df_w_raw.columns if c in ['datetime', 'date', 'cas']), None)
                p_col = next((c for c in df_w_raw.columns if c in ['precip', 'rain', 'srazky']), None)
                if t_col and p_col:
                    df_w_raw['ts'] = pd.to_datetime(df_w_raw[t_col]).dt.round('1min')
                    w_res = df_w_raw.set_index('ts').resample('1min').ffill().reindex(global_line, method='ffill')
                    mask_bad = (w_res[p_col] > 0.2)
                    obs_weights[mask_bad] = 0.5
            except:
                pass

        # 6. Global Target Correlation
        # Note: ta.weighted_pearson_corr does not return p-value, which is fine here
        target_corr = ta.weighted_pearson_corr(
            s_total_global.values,
            c_total_global.values,
            obs_weights.values
        )

        # 7. Exact Search Logic to Find Representative Day
        unique_days = df_t['actual_arrival'].dt.date.unique()
        unique_days = sorted([d for d in unique_days if pd.notnull(d)])

        best_day = None
        min_diff = float('inf')

        for d in unique_days:
            d_start = pd.Timestamp(d)
            d_end = d_start + pd.Timedelta(hours=23, minutes=59)

            day_mask = (global_line >= d_start) & (global_line <= d_end)
            if day_mask.sum() < 1000: continue

            s_slice = s_total_global[day_mask]
            c_slice = c_total_global[day_mask]
            w_slice = obs_weights[day_mask]

            if s_slice.sum() == 0 or c_slice.sum() == 0: continue

            day_corr = ta.weighted_pearson_corr(s_slice.values, c_slice.values, w_slice.values)
            if pd.isna(day_corr): continue

            diff = abs(day_corr - target_corr)
            if diff < min_diff:
                min_diff = diff
                best_day = d

        if best_day is None:
            print("  [!] Could not identify a valid representative day.\n")
            continue

        # 8. Calculate STATISTICAL SIGNIFICANCE specifically for the winning day
        d_start = pd.Timestamp(best_day)
        d_end = d_start + pd.Timedelta(hours=23, minutes=59)
        day_mask = (global_line >= d_start) & (global_line <= d_end)

        rep_r, rep_p, rep_neff = calculate_p_value_locally(
            s_total_global[day_mask].values,
            c_total_global[day_mask].values,
            obs_weights[day_mask].values
        )

        is_sig = "YES" if rep_p < ALPHA else "NO"

        print(f"  -> Selected Day: {best_day}")
        print(f"  -> Target Global r: {target_corr:.4f}")
        print(f"  -> Rep Day r: {rep_r:.4f}")
        print(f"  -> Rep Day p-value: {rep_p:.5f} -> Significant: {is_sig}\n")

        results.append({
            'City': city_name,
            'Representative_Date': best_day,
            'Target_Global_r': target_corr,
            'Rep_Day_r': rep_r,
            'Rep_Day_p_value': rep_p,
            'Significant': is_sig,
            'N_Effective': int(rep_neff)
        })

    # ==========================================
    # 3. EXPORT RESULTS
    # ==========================================
    if results:
        df_res = pd.DataFrame(results)

        # Ensure output directory exists
        out_dir = '../results'
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        out_path = os.path.join(out_dir, 'Representative_Day_Significance.csv')
        df_res.to_csv(out_path, index=False)

        print("=== FINAL REPORT ===")
        print(df_res[['City', 'Representative_Date', 'Rep_Day_r', 'Rep_Day_p_value', 'Significant']].to_string(
            index=False))
        print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    evaluate_representative_days()