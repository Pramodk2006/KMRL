import json

# Your configuration dictionaries (copy-paste your config code here)
settings_config = {
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
        "enable_logging": True
    },
    "ui": {
        "refresh_rate_seconds": 10,
        "max_simulation_scenarios": 5
    },
    "safety": {
        "min_fitness_days_ahead": 1,
        "enable_emergency_override": True,
        "audit_trail_retention_days": 365
    }
}

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
    "soft_constraints": {
        "mileage_balance": {
            "enabled": True,
            "target_deviation_percent": 10,
            "max_mileage_km": 30000
        },
        "branding_priority": {
            "enabled": True,
            "min_hours_threshold": 1.0
        },
        "bay_geometry": {
            "enabled": True,
            "prefer_high_score": True
        }
    },
    "validation_rules": {
        "max_trains_per_bay": 2,
        "min_service_trains": 3,
        "max_maintenance_trains": 2
    }
}

weights_config = {
    "default_weights": {
        "service_readiness": 0.25,
        "maintenance_penalty": 0.25,
        "branding_priority": 0.20,
        "mileage_balance": 0.15,
        "shunting_cost": 0.15
    },
    "seasonal_adjustments": {
        "monsoon": {
            "maintenance_penalty": 0.35,
            "service_readiness": 0.20
        },
        "festival": {
            "branding_priority": 0.30,
            "service_readiness": 0.30
        }
    },
    "emergency_weights": {
        "service_readiness": 0.50,
        "maintenance_penalty": 0.30,
        "branding_priority": 0.10,
        "mileage_balance": 0.05,
        "shunting_cost": 0.05
    }
}

# Writing settings.py as a Python file with dictionary assignment
with open('settings.py', 'w') as f:
    f.write("# KMRL IntelliFleet System Settings\n\n")
    f.write("SETTINGS = ")
    json.dump(settings_config, f, indent=2)

# Writing JSON config files
with open('constraints_config.json', 'w') as f:
    json.dump(constraints_config, f, indent=2)

with open('weights_config.json', 'w') as f:
    json.dump(weights_config, f, indent=2)

print("✅ All configuration files created successfully: settings.py, constraints_config.json, weights_config.json")
