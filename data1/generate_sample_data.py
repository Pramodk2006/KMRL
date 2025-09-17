import pandas as pd
from datetime import datetime, timedelta

# 1. trains.csv - Core train data with constraints
trains_data = {
    'train_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
    'fitness_valid_until': [
        '2025-09-10', '2025-09-04', '2025-08-31', '2025-09-12', '2025-09-08',
        '2025-09-15', '2025-09-06', '2025-08-30', '2025-09-20', '2025-09-11'
    ],
    'mileage_km': [15000, 27500, 32100, 8400, 22300, 18900, 29800, 12600, 35200, 19500],
    'branding_hours_left': [12.5, 2.0, 0.0, 20.0, 5.5, 8.2, 1.5, 15.3, 0.0, 10.8],
    'cleaning_slot_id': ['CS1', 'CS2', 'None', 'CS3', 'CS1', 'CS4', 'CS2', 'None', 'CS3', 'CS4'],
    'bay_geometry_score': [8, 5, 3, 9, 7, 6, 4, 3, 2, 7],
    'current_location': ['Depot_A', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_A']
}
trains_df = pd.DataFrame(trains_data)

# 2. job_cards.csv - Maintenance status
job_cards_data = {
    'train_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
    'job_card_status': ['closed', 'closed', 'open', 'closed', 'closed', 'open', 'closed', 'closed', 'closed', 'closed'],
    'last_maintenance_date': [
        '2025-08-15', '2025-08-20', '2025-08-25', '2025-09-01', '2025-08-18',
        '2025-08-22', '2025-08-28', '2025-08-30', '2025-08-12', '2025-08-26'
    ],
    'maintenance_priority': ['low', 'medium', 'high', 'low', 'medium', 'high', 'medium', 'low', 'high', 'low']
}
job_cards_df = pd.DataFrame(job_cards_data)

# 3. cleaning_slots.csv - Cleaning bay slots
cleaning_slots_data = {
    'cleaning_slot_id': ['CS1', 'CS2', 'CS3', 'CS4', 'CS5'],
    'available_bays': [2, 1, 2, 1, 0],
    'slot_type': ['deep_clean', 'basic_clean', 'deep_clean', 'basic_clean', 'maintenance'],
    'estimated_time_hours': [4, 2, 4, 2, 6]
}
cleaning_slots_df = pd.DataFrame(cleaning_slots_data)

# 4. bay_config.csv - Bay configuration
bay_config_data = {
    'bay_id': ['Bay1', 'Bay2', 'Bay3', 'Bay4', 'Bay5', 'Bay6'],
    'geometry_score': [9, 7, 5, 8, 6, 4],
    'max_capacity': [2, 2, 1, 2, 1, 1],
    'bay_type': ['service', 'service', 'maintenance', 'service', 'maintenance', 'storage'],
    'power_available': [True, True, False, True, False, False]
}
bay_config_df = pd.DataFrame(bay_config_data)

# 5. historical_outcomes.csv - For historical training data (sample)
historical_data = []
base_date = datetime(2025, 8, 1)
for i in range(50):
    date = base_date + timedelta(days=i)
    train_id = f"T{(i % 10) + 1:03d}"
    historical_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'train_id': train_id,
        'inducted': True if i % 3 != 0 else False,
        'actual_failure_occurred': True if i % 15 == 0 else False,
        'branding_sla_met': True if i % 5 != 0 else False,
        'energy_consumed_kwh': 150 + (i % 50),
        'passenger_complaints': i % 3
    })
historical_df = pd.DataFrame(historical_data)

# Save all dataframes to csv files
trains_df.to_csv('trains.csv', index=False)
job_cards_df.to_csv('job_cards.csv', index=False)
cleaning_slots_df.to_csv('cleaning_slots.csv', index=False)
bay_config_df.to_csv('bay_config.csv', index=False)
historical_df.to_csv('historical_outcomes.csv', index=False)

print("CSV files created successfully.")
