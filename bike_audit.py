import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import re


class BikeStationAudit:
    def __init__(self, rides_path, logs_path, rebal_path, output_folder='audit_results'):
        self.rides_path = rides_path
        self.logs_path = logs_path
        self.rebal_path = rebal_path
        self.output_folder = output_folder
        self.date_format = '%d.%m.%Y %H:%M'

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run_analysis(self, station_name, sheets, start_date, end_date):
        print(f"\n>>> [Inventory Audit] Starting for: {station_name}")

        # 1. Load Data
        try:
            df_rides = pd.read_excel(self.rides_path, sheet_name=sheets['rides'])
            df_logs = pd.read_excel(self.logs_path, sheet_name=sheets['logs'])
            df_rebal = pd.read_excel(self.rebal_path, sheet_name=sheets['rebal'])
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load data. {e}")
            return

        # 2. Preprocess Dates
        analysis_start = datetime.strptime(start_date, self.date_format)
        analysis_end = datetime.strptime(end_date, self.date_format)

        df_rides['start_time'] = pd.to_datetime(df_rides['start_time'], format=self.date_format, errors='coerce')
        df_rides['end_time'] = pd.to_datetime(df_rides['end_time'], format=self.date_format, errors='coerce')

        time_col = 'Čas' if 'Čas' in df_logs.columns else df_logs.columns[0]
        df_logs[time_col] = pd.to_datetime(df_logs[time_col], format=self.date_format, errors='coerce')

        df_rebal['timestamp'] = pd.to_datetime(df_rebal['timestamp'], format=self.date_format, errors='coerce')

        # 3. Build Event Timeline
        events = []

        # Rides Out (Start at station)
        mask_out = (df_rides['start_place'] == station_name) & (
            df_rides['start_time'].between(analysis_start, analysis_end))
        for _, row in df_rides[mask_out].iterrows():
            events.append({'time': row['start_time'], 'change': -1})

        # Rides In (End at station)
        mask_in = (df_rides['end_place'] == station_name) & (df_rides['end_time'].between(analysis_start, analysis_end))
        for _, row in df_rides[mask_in].iterrows():
            events.append({'time': row['end_time'], 'change': 1})

        # Rebalancing
        mask_reb_out = (df_rebal['from'] == station_name) & (
            df_rebal['timestamp'].between(analysis_start, analysis_end))
        for _, row in df_rebal[mask_reb_out].iterrows():
            qty = row['number_of_bikes'] if pd.notnull(row['number_of_bikes']) else 0
            events.append({'time': row['timestamp'], 'change': -qty})

        mask_reb_in = (df_rebal['to'] == station_name) & (df_rebal['timestamp'].between(analysis_start, analysis_end))
        for _, row in df_rebal[mask_reb_in].iterrows():
            qty = row['number_of_bikes'] if pd.notnull(row['number_of_bikes']) else 0
            events.append({'time': row['timestamp'], 'change': qty})

        df_events = pd.DataFrame(events)
        if df_events.empty:
            print("   No movement events found.")
            return

        df_events = df_events.sort_values('time')

        # 4. Filter Logs & Calculate
        has_logs = False
        station_logs = pd.DataFrame()

        if station_name in df_logs.columns:
            station_logs = df_logs[[time_col, station_name]].dropna().sort_values(time_col)
            station_logs = station_logs[station_logs[time_col].between(analysis_start, analysis_end)]
            if not station_logs.empty:
                has_logs = True

        plot_segments = []
        discrepancy_table = []

        if has_logs:
            for i in range(len(station_logs) - 1):
                start_log, end_log = station_logs.iloc[i], station_logs.iloc[i + 1]
                seg, end_val = self._calculate_segment(df_events, start_log[time_col], start_log[station_name],
                                                       end_log[time_col])
                plot_segments.append(seg)

                diff = end_log[station_name] - end_val
                if diff != 0:
                    discrepancy_table.append({
                        'Log Timestamp': end_log[time_col], 'Expected': int(end_val),
                        'Actual': int(end_log[station_name]), 'Diff': int(diff),
                        'Note': 'Missing' if diff < 0 else 'Surplus'
                    })

            # Tail
            last_log = station_logs.iloc[-1]
            if last_log[time_col] < analysis_end:
                tail_seg, _ = self._calculate_segment(df_events, last_log[time_col], last_log[station_name],
                                                      analysis_end)
                plot_segments.append(tail_seg)
        else:
            # Auto-estimate
            df_events['cumulative'] = df_events['change'].cumsum()
            min_inventory = df_events['cumulative'].min()
            start_offset = abs(min_inventory) if min_inventory < 0 else 0
            df_events['inventory'] = df_events['cumulative'] + start_offset

            start_pt = pd.DataFrame({'time': [analysis_start], 'inventory': [start_offset]})
            middle_pts = df_events[['time', 'inventory']]
            end_pt = pd.DataFrame({'time': [analysis_end], 'inventory': [df_events.iloc[-1]['inventory']]})
            plot_segments.append(pd.concat([start_pt, middle_pts, end_pt]))

        self._save_results(station_name, discrepancy_table, plot_segments, station_logs, time_col)

    def _calculate_segment(self, all_events, start_time, start_count, end_time):
        interval_events = all_events[(all_events['time'] > start_time) & (all_events['time'] <= end_time)].copy()
        if interval_events.empty:
            return pd.DataFrame({'time': [start_time, end_time], 'inventory': [start_count, start_count]}), start_count

        interval_events = interval_events.sort_values('time')
        interval_events['cumulative_change'] = interval_events['change'].cumsum()
        interval_events['inventory'] = start_count + interval_events['cumulative_change']
        return pd.concat([
            pd.DataFrame({'time': [start_time], 'inventory': [start_count]}),
            interval_events[['time', 'inventory']],
            pd.DataFrame({'time': [end_time], 'inventory': [interval_events.iloc[-1]['inventory']]})
        ]), interval_events.iloc[-1]['inventory']

    def _sanitize_filename(self, name):
        name = re.sub(r'[^\w\s-]', '', name).strip().lower()
        return re.sub(r'[-\s]+', '_', name)

    def _save_results(self, station_name, discrepancy_table, segments, logs, time_col):
        clean_name = self._sanitize_filename(station_name)
        if discrepancy_table:
            pd.DataFrame(discrepancy_table).to_csv(
                os.path.join(self.output_folder, f"{clean_name}_inventory_report.csv"), index=False)

        plt.figure(figsize=(14, 7))
        for seg in segments:
            plt.step(seg['time'], seg['inventory'], where='post', color='#1f77b4', linewidth=2)
        if not logs.empty:
            plt.scatter(logs[time_col], logs[station_name], color='red', zorder=5)
        for adj in discrepancy_table:
            plt.vlines(x=adj['Log Timestamp'], ymin=adj['Expected'], ymax=adj['Actual'], color='orange', linestyle='--')

        plt.title(f"Inventory: {station_name}")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
        plt.gcf().autofmt_xdate()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f"{clean_name}_graph.png"))
        plt.close()
        print(f"   -> Inventory Graph saved.")