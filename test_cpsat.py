import sys
sys.path.insert(0, 'src')

from src.cpsat_optimizer import solve_bay_assignment_cpsat
import pandas as pd

# Create test data
eligible_trains = [
    {'train_id': 'T001', 'bay_geometry_score': 0.8},
    {'train_id': 'T002', 'bay_geometry_score': 0.9}
]

service_bays_df = pd.DataFrame([
    {'bay_id': 'Bay1', 'max_capacity': 2, 'geometry_score': 0.85},
    {'bay_id': 'Bay2', 'max_capacity': 1, 'geometry_score': 0.75}
])

print("Testing CP-SAT solver...")
try:
    result = solve_bay_assignment_cpsat(eligible_trains, service_bays_df)
    print("CP-SAT solver completed successfully")
    print("Result:", result)
except Exception as e:
    print("CP-SAT solver failed:", e)
