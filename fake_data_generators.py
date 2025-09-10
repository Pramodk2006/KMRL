# fake_data_generators.py
"""
Generates realistic fake data for all missing KMRL IntelliFleet features.
This simulates real-time integration, mileage balancing, cleaning slots, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class FakeDataGenerators:
    """Generates convincing fake data for all problem statement requirements"""
    
    def __init__(self):
        self.current_time = datetime.now()
        self.train_ids = [f'T{i:03d}' for i in range(1, 26)]
        
    def generate_mileage_history(self):
        """Generate realistic mileage data for all 25 trains"""
        data = []
        for train_id in self.train_ids:
            base_mileage = random.randint(12000, 28000)
            for days_back in range(90, 0, -1):
                date = self.current_time - timedelta(days=days_back)
                daily_km = random.randint(150, 400)  # Realistic daily usage
                cumulative_km = base_mileage + (90 - days_back) * daily_km
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'train_id': train_id,
                    'daily_km': daily_km,
                    'cumulative_km': cumulative_km,
                    'bogie_wear_index': min(100, cumulative_km / 250),
                    'brake_pad_wear': min(100, cumulative_km / 200),
                    'hvac_usage_hours': daily_km * 0.8,
                    'maintenance_due_km': max(0, 25000 - cumulative_km)
                })
        
        return pd.DataFrame(data)
    
    def generate_cleaning_schedule(self):
        """Generate cleaning bay availability and staff scheduling"""
        cleaning_bays = ['CB1', 'CB2', 'CB3', 'CB4']
        shifts = ['Morning', 'Afternoon', 'Night']
        staff_names = ['Ravi Kumar', 'Priya S', 'Anand M', 'Lakshmi R', 'Suresh T', 'Maya P']
        
        data = []
        for days_back in range(7, 0, -1):
            date = self.current_time - timedelta(days=days_back)
            for bay in cleaning_bays:
                for shift in shifts:
                    occupied = random.choice([True, False, False])  # 33% occupied
                    staff_count = random.randint(2, 4)
                    assigned_staff = random.sample(staff_names, staff_count)
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'bay_id': bay,
                        'shift': shift,
                        'occupied': occupied,
                        'assigned_train': random.choice(self.train_ids) if occupied else None,
                        'staff_count': staff_count,
                        'assigned_staff': ', '.join(assigned_staff),
                        'cleaning_type': random.choice(['Deep Clean', 'Standard', 'Express']),
                        'estimated_duration': random.randint(2, 6),
                        'efficiency_rating': random.randint(85, 98)
                    })
        
        return pd.DataFrame(data)
    
    def generate_branding_contracts(self):
        """Generate advertiser SLA requirements and exposure tracking"""
        advertisers = [
            'Coca-Cola Kerala', 'LuLu Group', 'Federal Bank', 'Marico Kochi', 
            'Malabar Gold', 'Jos Alukkas', 'Kalyan Jewellers', 'Metro Shoes'
        ]
        
        data = []
        for advertiser in advertisers:
            trains_assigned = random.sample(self.train_ids, random.randint(2, 5))
            for train_id in trains_assigned:
                contract_start = self.current_time - timedelta(days=random.randint(30, 180))
                contract_end = contract_start + timedelta(days=random.randint(90, 365))
                
                data.append({
                    'advertiser': advertiser,
                    'train_id': train_id,
                    'contract_start': contract_start.strftime('%Y-%m-%d'),
                    'contract_end': contract_end.strftime('%Y-%m-%d'),
                    'min_daily_hours': random.randint(8, 16),
                    'min_monthly_km': random.randint(3000, 8000),
                    'penalty_per_hour': random.randint(500, 2000),
                    'current_exposure_hours': random.randint(5, 18),
                    'monthly_km_achieved': random.randint(2500, 9000),
                    'sla_compliance': random.choice([True, True, True, False]),  # 75% compliant
                    'revenue_per_day': random.randint(2000, 8000),
                    'penalty_risk': random.randint(0, 15000)
                })
        
        return pd.DataFrame(data)
    
    def generate_maximo_job_cards(self):
        """Generate fake IBM Maximo job card exports"""
        job_types = [
            'Safety Inspection', 'Brake System Check', 'HVAC Maintenance', 
            'Door System Service', 'Electrical Systems', 'Bogies Inspection',
            'Communication Equipment', 'Emergency Systems', 'Cleaning Systems'
        ]
        
        departments = ['Rolling-Stock', 'Signalling', 'Telecom', 'Electrical', 'Mechanical']
        statuses = ['Open', 'In Progress', 'Closed', 'Pending Approval', 'Overdue']
        priorities = ['Critical', 'High', 'Medium', 'Low']
        
        data = []
        for train_id in self.train_ids:
            num_jobs = random.randint(1, 4)  # 1-4 jobs per train
            for _ in range(num_jobs):
                created_date = self.current_time - timedelta(days=random.randint(1, 30))
                due_date = created_date + timedelta(days=random.randint(3, 14))
                
                status = random.choice(statuses)
                is_overdue = status in ['Open', 'In Progress'] and due_date < self.current_time
                
                data.append({
                    'job_card_id': f'JC{random.randint(100000, 999999)}',
                    'train_id': train_id,
                    'job_type': random.choice(job_types),
                    'department': random.choice(departments),
                    'priority': random.choice(priorities),
                    'status': 'Overdue' if is_overdue else status,
                    'created_date': created_date.strftime('%Y-%m-%d %H:%M'),
                    'due_date': due_date.strftime('%Y-%m-%d %H:%M'),
                    'assigned_technician': f'Tech{random.randint(1, 20):02d}',
                    'estimated_hours': random.randint(2, 24),
                    'actual_hours': random.randint(1, 30) if status == 'Closed' else None,
                    'parts_required': random.choice([True, False]),
                    'safety_clearance': random.choice([True, True, True, False]),  # 75% cleared
                    'fitness_impact': random.choice(['None', 'Minor', 'Major']),
                    'completion_percentage': random.randint(0, 100) if status != 'Open' else 0
                })
        
        return pd.DataFrame(data)
    
    def generate_iot_sensor_feeds(self):
        """Generate fake real-time IoT sensor data"""
        sensor_types = [
            'Door_Sensor', 'Brake_Pressure', 'HVAC_Temperature', 'Battery_Voltage',
            'Axle_Temperature', 'Vibration_Monitor', 'Pantograph_Status', 'Traction_Current'
        ]
        
        data = []
        for train_id in self.train_ids:
            for sensor_type in sensor_types:
                # Generate last 24 hours of data
                for hours_back in range(24, 0, -1):
                    timestamp = self.current_time - timedelta(hours=hours_back)
                    
                    # Generate realistic sensor values based on type
                    if sensor_type == 'Door_Sensor':
                        value = random.choice([0, 1])  # Open/Closed
                        status = 'Normal' if value == 0 else 'Door Open'
                    elif sensor_type == 'Brake_Pressure':
                        value = random.uniform(5.8, 6.2)  # Bar
                        status = 'Normal' if 5.9 <= value <= 6.1 else 'Warning'
                    elif sensor_type == 'HVAC_Temperature':
                        value = random.uniform(22, 28)  # Celsius
                        status = 'Normal' if 23 <= value <= 26 else 'Alert'
                    elif sensor_type == 'Battery_Voltage':
                        value = random.uniform(72, 76)  # Volts
                        status = 'Normal' if value >= 74 else 'Low Voltage'
                    else:
                        value = random.uniform(0, 100)
                        status = random.choice(['Normal', 'Normal', 'Normal', 'Warning'])
                    
                    data.append({
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'train_id': train_id,
                        'sensor_type': sensor_type,
                        'value': round(value, 2),
                        'unit': self._get_sensor_unit(sensor_type),
                        'status': status,
                        'location': f'Car_{random.randint(1, 4)}',
                        'last_calibration': (timestamp - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
                    })
        
        return pd.DataFrame(data)
    
    def generate_performance_history(self):
        """Generate historical KPI performance data"""
        data = []
        for days_back in range(90, 0, -1):
            date = self.current_time - timedelta(days=days_back)
            
            # Generate daily performance metrics
            total_services = random.randint(180, 220)
            delayed_services = random.randint(0, 8)
            punctuality = ((total_services - delayed_services) / total_services) * 100
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'total_services': total_services,
                'on_time_services': total_services - delayed_services,
                'delayed_services': delayed_services,
                'punctuality_percentage': round(punctuality, 2),
                'avg_delay_minutes': random.uniform(0.5, 3.2),
                'passenger_complaints': random.randint(0, 5),
                'revenue_per_service': random.randint(1800, 2200),
                'energy_consumption_kwh': random.randint(2800, 3400),
                'maintenance_alerts': random.randint(0, 3),
                'safety_incidents': random.randint(0, 1),
                'customer_satisfaction': random.uniform(4.2, 4.8),
                'cost_per_km': random.uniform(12.5, 18.2)
            })
        
        return pd.DataFrame(data)
    
    def _get_sensor_unit(self, sensor_type):
        """Get appropriate unit for sensor type"""
        units = {
            'Door_Sensor': 'status',
            'Brake_Pressure': 'bar',
            'HVAC_Temperature': '°C',
            'Battery_Voltage': 'V',
            'Axle_Temperature': '°C',
            'Vibration_Monitor': 'Hz',
            'Pantograph_Status': 'status',
            'Traction_Current': 'A'
        }
        return units.get(sensor_type, 'units')
    
    def generate_all_fake_data(self):
        """Generate all fake datasets and save to files"""
        datasets = {
            'mileage_history': self.generate_mileage_history(),
            'cleaning_schedule': self.generate_cleaning_schedule(),
            'branding_contracts': self.generate_branding_contracts(),
            'maximo_job_cards': self.generate_maximo_job_cards(),
            'iot_sensor_feeds': self.generate_iot_sensor_feeds(),
            'performance_history': self.generate_performance_history()
        }
        
        # Save all datasets
        for name, df in datasets.items():
            filename = f'data/fake_{name}.csv'
            df.to_csv(filename, index=False)
            print(f"✅ Generated {filename} with {len(df)} records")
        
        return datasets

# Configuration for the fake system
FAKE_SYSTEM_CONFIG = {
    'total_trains': 25,
    'service_bays': 6,
    'cleaning_bays': 4,
    'depot_capacity': 30,
    'operation_window': '21:00-23:00',
    'punctuality_target': 99.5,
    'mileage_balancing_enabled': True,
    'ml_learning_enabled': True,
    'multi_depot_enabled': True,
    'real_time_integration': True
}

if __name__ == "__main__":
    generator = FakeDataGenerators()
    datasets = generator.generate_all_fake_data()
    
    print(f"🎭 Fake data generation completed!")
    print(f"📊 Total datasets: {len(datasets)}")
    print(f"📈 Ready for comprehensive demonstration!")