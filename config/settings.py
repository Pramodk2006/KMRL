# KMRL IntelliFleet System Settings

SETTINGS = {
  "system": {
    "name": "KMRL IntelliFleet",
    "version": "1.0.0",
    "environment": "development"
  },
  "data": {
    "max_trains": 25,
    "max_bays": 6,
    "data_refresh_interval_minutes": 5,
    "stale_data_threshold_minutes": 30
  },
  "optimization": {
    "max_solve_time_seconds": 60,
    "solver_threads": 4,
    "enable_logging": true
  },
  "ui": {
    "refresh_rate_seconds": 10,
    "max_simulation_scenarios": 5
  },
  "safety": {
    "min_fitness_days_ahead": 1,
    "enable_emergency_override": true,
    "audit_trail_retention_days": 365
  }
}