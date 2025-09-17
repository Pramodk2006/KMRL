# KMRL IntelliFleet System Settings
SYSTEM_CONFIG = {
    'max_trains_per_night': 50,
    'optimization_timeout_minutes': 10,
    'simulation_speed_multiplier': 1.0,
    'enable_predictive_ai': True,
    'enable_real_time_monitoring': True
}
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
    "stale_data_threshold_minutes": 30,
    "required_trains": 21,
    "yard_topology_path": ""
  },
  "optimization": {
    "max_solve_time_seconds": 60,
    "solver_threads": 4,
    "enable_logging": true,
    "geometry_weight": 1.0,
    "shunting_weight": 1.0
  },
  "analytics": {
    "energy_kwh_per_train_service": 350.0,
    "punctuality_failure_threshold": 0.05,
    "cost_per_failure": 20000.0,
    "maintenance_cost_per_km": 12.0
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