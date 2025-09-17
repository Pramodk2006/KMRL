import sys
sys.path.insert(0, 'src')

from src.cpsat_optimizer import solve_bay_assignment_cpsat
import pandas as pd

# Create minimal test data
eligible_trains = [
    {'train_id': 'T001', 'bay_geometry_score': 0.8}
]

service_bays_df = pd.DataFrame([
    {'bay_id': 'Bay1', 'max_capacity': 1, 'geometry_score': 0.85}
])

print("Testing simple CP-SAT solver...")
try:
    result = solve_bay_assignment_cpsat(eligible_trains, service_bays_df)
    print("CP-SAT solver completed successfully")
    print("Result:", result)
except Exception as e:
    print("CP-SAT solver failed:", e)
    import traceback
    traceback.print_exc()
