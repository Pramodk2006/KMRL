# Let me first recreate the sample data files so Layer 0 can load them

import pandas as pd
from datetime import datetime, timedelta
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import DataLoader

# Recreate the sample data
trains_data = {
    'train_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
    'fitness_valid_until': [
        '2025-09-10',  # Valid
        '2025-09-04',  # Expires tomorrow (edge case)  
        '2025-08-31',  # Already expired (CONFLICT)
        '2025-09-12',  # Valid
        '2025-09-08',  # Valid
        '2025-09-15',  # Valid
        '2025-09-06',  # Valid
        '2025-08-30',  # Expired (CONFLICT)
        '2025-09-20',  # Valid
        '2025-09-11'   # Valid
    ],
    'mileage_km': [15000, 27500, 32100, 8400, 22300, 18900, 29800, 12600, 35200, 19500],
    'branding_hours_left': [12.5, 2.0, 0.0, 20.0, 5.5, 8.2, 1.5, 15.3, 0.0, 10.8],
    'cleaning_slot_id': ['CS1', 'CS2', 'None', 'CS3', 'CS1', 'CS4', 'CS2', 'None', 'CS3', 'CS4'],
    'bay_geometry_score': [8, 5, 3, 9, 7, 6, 4, 8, 2, 7],
    'current_location': ['Depot_A', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_B', 'Depot_A', 'Depot_A']
}

job_cards_data = {
    'train_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
    'job_card_status': ['closed', 'closed', 'open', 'closed', 'closed', 'open', 'closed', 'closed', 'closed', 'closed'],
    'last_maintenance_date': [
        '2025-08-15', '2025-08-20', '2025-08-25', '2025-09-01', '2025-08-18',
        '2025-08-22', '2025-08-28', '2025-08-30', '2025-08-12', '2025-08-26'
    ],
    'maintenance_priority': ['low', 'medium', 'high', 'low', 'medium', 'high', 'medium', 'low', 'high', 'low']
}

cleaning_slots_data = {
    'cleaning_slot_id': ['CS1', 'CS2', 'CS3', 'CS4', 'CS5'],
    'available_bays': [2, 1, 2, 1, 0],
    'slot_type': ['deep_clean', 'basic_clean', 'deep_clean', 'basic_clean', 'maintenance'],
    'estimated_time_hours': [4, 2, 4, 2, 6]
}

bay_config_data = {
    'bay_id': ['Bay1', 'Bay2', 'Bay3', 'Bay4', 'Bay5', 'Bay6'],
    'geometry_score': [9, 7, 5, 8, 6, 4],
    'max_capacity': [2, 2, 1, 2, 1, 1],
    'bay_type': ['service', 'service', 'maintenance', 'service', 'maintenance', 'storage'],
    'power_available': [True, True, False, True, False, False]
}

# Create and save DataFrames
trains_df = pd.DataFrame(trains_data)
job_cards_df = pd.DataFrame(job_cards_data)
cleaning_slots_df = pd.DataFrame(cleaning_slots_data)
bay_config_df = pd.DataFrame(bay_config_data)

# Save to CSV files
trains_df.to_csv('trains.csv', index=False)
job_cards_df.to_csv('job_cards.csv', index=False)
cleaning_slots_df.to_csv('cleaning_slots.csv', index=False)
bay_config_df.to_csv('bay_config.csv', index=False)

# Also create the config file
constraints_config = {
    "hard_constraints": {
        "fitness_certificate": {
            "enabled": True,
            "description": "Train must have valid fitness certificate",
            "grace_period_hours": 0,
            "emergency_override_allowed": True
        },
        "job_card_status": {
            "enabled": True,
            "description": "All job cards must be closed",
            "emergency_override_allowed": True
        },
        "cleaning_slot": {
            "enabled": True,
            "description": "Train must have assigned cleaning slot",
            "allow_none": False,
            "emergency_override_allowed": True
        }
    },
    "validation_rules": {
        "max_trains_per_bay": 2,
        "min_service_trains": 3,
        "max_maintenance_trains": 2
    }
}

with open('constraints_config.json', 'w') as f:
    json.dump(constraints_config, f, indent=2)

print("✅ All sample data files recreated successfully!")
print("📁 Files created: trains.csv, job_cards.csv, cleaning_slots.csv, bay_config.csv, constraints_config.json")

# Now test the data loader again
print("\n=== Re-testing Layer 0 with Data Files ===")
loader = DataLoader()
data = loader.get_integrated_data()

print(f"\n🎉 Layer 0 Successfully Implemented!")
print(f"📊 Ready for Layer 1: Constraint Engine")
