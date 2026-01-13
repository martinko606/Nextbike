import auditor

from bike_audit import BikeStationAudit
from train_analysis import TrainImpactAnalyzer

# ============================
# 1. SETUP GLOBAL PATHS
# ============================
# These usually stay the same, or you can change them per instance if files differ
FILE_RIDES = r'C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik.xlsx'
FILE_LOGS = r'C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik.xlsx'
FILE_REBAL = r'C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik2.xlsx'

OUTPUT_DIR = 'Analysis_Results'

# Initialize Both Tools
inventory_audit = BikeStationAudit(FILE_RIDES, FILE_LOGS, FILE_REBAL, output_folder=OUTPUT_DIR)
train_audit = TrainImpactAnalyzer(FILE_RIDES, FILE_TRAINS, output_folder=OUTPUT_DIR)

# ============================
# 2. DEFINE YOUR INSTANCES
# ============================

# --- INSTANCE 1: Svinov Station ---
station_1_config = {
    'station_name': "SV-Svinov nádraží *(navíc 15min na odjezd)",
    'sheets': {
        'rides': 'Vypujcky_Ostrava',      # Change this if Svinov is on a different sheet
        'logs': 'Stanice_Ostrava',
        'rebal': 'Ostrava'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

# --- INSTANCE 2: Another Station (Example) ---
station_2_config = {
    'station_name': "MOAP-Tieto towers",
    'sheets': {
        'rides': 'Vypujcky_Ostrava',
        'logs': 'Stanice_Ostrava',
        'rebal': 'Ostrava'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_3_config = {
    'station_name': "MOAP-Hlavní nádraží",
    'sheets': {
        'rides': 'Vypujcky_Ostrava',
        'logs': 'Stanice_Ostrava',
        'rebal': 'Ostrava'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_4_config = {
    'station_name': "Hlavní nádraží - Hlavní vstup",
    'sheets': {
        'rides': 'Vypujcky_Brno',
        'logs': 'Stanice_Brno',
        'rebal': 'Brno'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_5_config = {
    'station_name': "Hlavní nádraží - pošta",
    'sheets': {
        'rides': 'Vypujcky_Brno',
        'logs': 'Stanice_Brno',
        'rebal': 'Brno'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_6_config = {
    'station_name': "Bajkazyl 666",
    'sheets': {
        'rides': 'Vypujcky_Brno',
        'logs': 'Stanice_Brno',
        'rebal': 'Brno'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_7_config = {
    'station_name': "Nádraží",
    'sheets': {
        'rides': 'Vypujcky_Prerov',
        'logs': 'Stanice_Prerov',
        'rebal': 'Prerov'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

station_8_config = {
    'station_name': "Vlakové nádraží Valašské Meziříčí (nové umístění)",
    'sheets': {
        'rides': 'Vypujcky_ValMez',
        'logs': 'Stanice_ValMez',
        'rebal': 'ValMez'
    },
    'start_date': "15.09.2025 0:00",
    'end_date': "19.10.2025 23:59"
}

# ============================
# 3. RUN THE ANALYSIS
# ============================

# Run Instance 1
auditor.run_analysis(
    station_name=station_1_config['station_name'],
    sheets=station_1_config['sheets'],
    start_date=station_1_config['start_date'],
    end_date=station_1_config['end_date']
)

auditor.run_analysis(
    station_name=station_2_config['station_name'],
    sheets=station_2_config['sheets'],
    start_date=station_2_config['start_date'],
    end_date=station_2_config['end_date']
)

auditor.run_analysis(
    station_name=station_3_config['station_name'],
    sheets=station_3_config['sheets'],
    start_date=station_3_config['start_date'],
    end_date=station_3_config['end_date']
)

auditor.run_analysis(
    station_name=station_4_config['station_name'],
    sheets=station_4_config['sheets'],
    start_date=station_4_config['start_date'],
    end_date=station_4_config['end_date']
)

auditor.run_analysis(
    station_name=station_5_config['station_name'],
    sheets=station_5_config['sheets'],
    start_date=station_5_config['start_date'],
    end_date=station_5_config['end_date']
)

auditor.run_analysis(
    station_name=station_6_config['station_name'],
    sheets=station_6_config['sheets'],
    start_date=station_6_config['start_date'],
    end_date=station_6_config['end_date']
)

auditor.run_analysis(
    station_name=station_7_config['station_name'],
    sheets=station_7_config['sheets'],
    start_date=station_7_config['start_date'],
    end_date=station_7_config['end_date']
)

auditor.run_analysis(
    station_name=station_8_config['station_name'],
    sheets=station_8_config['sheets'],
    start_date=station_8_config['start_date'],
    end_date=station_8_config['end_date']
)