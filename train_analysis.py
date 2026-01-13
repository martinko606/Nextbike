import pandas as pd
from datetime import datetime, timedelta
import os
import re


class TrainImpactAnalyzer:
    def __init__(self, rides_path, trains_path, output_folder='audit_results'):
        self.rides_path = rides_path
        self.trains_path = trains_path
        self.output_folder = output_folder
        self.date_format = '%d.%m.%Y %H:%M'

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def run_train_analysis(self, station_name, sheets, start_date, end_date):
        print(f"\n>>> [Train Impact] Starting for: {station_name}")

        # 1. Load Data
        try:
            df_rides = pd.read_excel(self.rides_path, sheet_name=sheets['rides'])
            df_trains = pd.read_excel(self.trains_path, sheet_name=sheets['trains'])
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load data. {e}")
            return

        # 2. Preprocess Dates
        analysis_start = datetime.strptime(start_date, self.date_format)
        analysis_end = datetime.strptime(end_date, self.date_format)

        # Process Rides
        df_rides['start_time'] = pd.to_datetime(df_rides['start_time'], format=self.date_format, errors='coerce')
        df_rides['end_time'] = pd.to_datetime(df_rides['end_time'], format=self.date_format, errors='coerce')

        # Process Trains
        df_trains['time'] = pd.to_datetime(df_trains['time'], format=self.date_format, errors='coerce')
        # Filter trains to window
        df_trains = df_trains[df_trains['time'].between(analysis_start, analysis_end)].sort_values('time')

        if df_trains.empty:
            print("   -> No train events found in the specified timeframe.")
            return

        # 3. Analyze Impact
        impact_results = []

        # We need two subsets of rides:
        # A. People riding TO the station (Ending their ride there)
        rides_arriving = df_rides[df_rides['end_place'] == station_name].copy()

        # B. People riding FROM the station (Starting their ride there)
        rides_leaving = df_rides[df_rides['start_place'] == station_name].copy()

        for _, train in df_trains.iterrows():
            t_time = train['time']
            t_type = str(train['type']).lower().strip()  # "arrival" or "departure"
            t_cat = train['category']

            bike_count = 0
            window_desc = ""
            action = ""

            # SCENARIO 1: Train Departs (People arrive to catch it)
            if "departure" in t_type or "odjezd" in t_type:
                # Window: 10 mins BEFORE train leaves
                start_w = t_time - timedelta(minutes=10)
                end_w = t_time

                # Count matching rides
                count = len(rides_arriving[rides_arriving['end_time'].between(start_w, end_w)])

                bike_count = count
                window_desc = "10 min before"
                action = "Bikes Arriving"

            # SCENARIO 2: Train Arrives (People leave the station)
            elif "arrival" in t_type or "příjezd" in t_type:
                # Window: 5 mins AFTER train arrives
                start_w = t_time
                end_w = t_time + timedelta(minutes=5)

                # Count matching rides
                count = len(rides_leaving[rides_leaving['start_time'].between(start_w, end_w)])

                bike_count = count
                window_desc = "5 min after"
                action = "Bikes Leaving"

            impact_results.append({
                'Train Time': t_time,
                'Train Type': train['type'],
                'Category': t_cat,
                'Bike Action': action,
                'Count': bike_count,
                'Window': window_desc
            })

        # 4. Save Results
        if impact_results:
            clean_name = re.sub(r'[^\w\s-]', '', station_name).strip().lower().replace(' ', '_')
            df_impact = pd.DataFrame(impact_results)
            path = os.path.join(self.output_folder, f"{clean_name}_TRAIN_IMPACT.csv")
            df_impact.to_csv(path, index=False)
            print(f"   -> Train Analysis saved: {path}")
        else:
            print("   -> No results generated.")