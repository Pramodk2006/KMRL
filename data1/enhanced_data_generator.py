"""
Enhanced Data Generator for KMRL IntelliFleet Project
Generates comprehensive realistic fake data for all datasets to improve testing coverage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

class EnhancedKMRLDataGenerator:
    """Generates enhanced realistic data for all KMRL IntelliFleet components"""
    
    def __init__(self):
        self.current_time = datetime.now()
        # Scale up to 25 trains as per problem statement
        self.train_ids = [f'T{i:03d}' for i in range(1, 26)]
        self.depot_locations = ['Muttom_Depot', 'Ernakulam_South', 'Kadavanthra', 'Maharajas_College']
        self.cleaning_slot_ids = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS8']
        
        # Ensure data folder exists
        os.makedirs('data', exist_ok=True)
        
    def generate_enhanced_trains_data(self):
        """Generate enhanced trains.csv with 25 trains and realistic variations"""
        data = []
        
        for i, train_id in enumerate(self.train_ids):
            # Vary fitness validity - some expired, some valid, some expiring soon
            if i < 8:  # 8 trains with expired certificates (32%)
                fitness_date = self.current_time - timedelta(days=random.randint(1, 30))
            elif i < 15:  # 7 trains with valid certificates (28%)
                fitness_date = self.current_time + timedelta(days=random.randint(30, 180))
            else:  # 10 trains with certificates expiring soon (40%)
                fitness_date = self.current_time + timedelta(days=random.randint(1, 15))
            
            # Realistic mileage distribution
            if i < 5:  # Low mileage trains
                mileage = random.randint(5000, 15000)
            elif i < 15:  # Medium mileage trains
                mileage = random.randint(15000, 25000)
            else:  # High mileage trains
                mileage = random.randint(25000, 40000)
            
            # Branding hours distribution
            branding_hours = random.choice([
                0.0, 0.0, 0.0,  # No branding (30%)
                round(random.uniform(1.0, 5.0), 1),  # Low branding
                round(random.uniform(5.0, 12.0), 1),  # Medium branding
                round(random.uniform(12.0, 24.0), 1)  # High branding
            ])
            
            # Cleaning slot assignment (some None for testing)
            cleaning_slot = random.choice(self.cleaning_slot_ids + [None, None])
            
            # Bay geometry score
            bay_score = random.randint(2, 10)
            
            # Current location
            location = random.choice(self.depot_locations)
            
            data.append({
                'train_id': train_id,
                'fitness_valid_until': fitness_date.strftime('%Y-%m-%d'),
                'mileage_km': mileage,
                'branding_hours_left': branding_hours,
                'cleaning_slot_id': cleaning_slot,
                'bay_geometry_score': bay_score,
                'current_location': location
            })
        
        return pd.DataFrame(data)
    
    def generate_enhanced_job_cards_data(self):
        """Generate comprehensive job cards with multiple entries per train"""
        data = []
        
        for train_id in self.train_ids:
            # Generate 1-3 job cards per train
            num_jobs = random.randint(1, 3)
            
            for job_num in range(num_jobs):
                # Job card status distribution
                status = random.choices(
                    ['closed', 'open', 'pending', 'overdue'],
                    weights=[60, 25, 10, 5]  # 60% closed, 25% open, etc.
                )[0]
                
                # Last maintenance date
                maintenance_date = self.current_time - timedelta(days=random.randint(1, 60))
                
                # Priority distribution
                priority = random.choices(
                    ['low', 'medium', 'high', 'critical'],
                    weights=[40, 35, 20, 5]
                )[0]
                
                # Job type
                job_types = [
                    'Safety Inspection', 'Brake System', 'HVAC Maintenance',
                    'Door System', 'Electrical Check', 'Bogie Inspection',
                    'Communication Systems', 'Emergency Systems', 'Interior Cleaning'
                ]
                
                # Estimated completion time
                estimated_hours = random.randint(2, 24)
                
                data.append({
                    'train_id': train_id,
                    'job_card_id': f'JC{random.randint(100000, 999999)}',
                    'job_card_status': status,
                    'last_maintenance_date': maintenance_date.strftime('%Y-%m-%d'),
                    'maintenance_priority': priority,
                    'job_type': random.choice(job_types),
                    'estimated_hours': estimated_hours,
                    'assigned_technician': f'Tech{random.randint(1, 20):02d}',
                    'department': random.choice(['Rolling-Stock', 'Signalling', 'Telecom', 'Electrical'])
                })
        
        return pd.DataFrame(data)
    
    def generate_enhanced_cleaning_slots_data(self):
        """Generate enhanced cleaning slots with time-based availability"""
        data = []
        
        # Enhanced cleaning slots
        slots_config = [
            {'id': 'CS1', 'type': 'deep_clean', 'bays': 3, 'hours': 5},
            {'id': 'CS2', 'type': 'basic_clean', 'bays': 2, 'hours': 2},
            {'id': 'CS3', 'type': 'deep_clean', 'bays': 3, 'hours': 5},
            {'id': 'CS4', 'type': 'basic_clean', 'bays': 2, 'hours': 2},
            {'id': 'CS5', 'type': 'maintenance_clean', 'bays': 1, 'hours': 8},
            {'id': 'CS6', 'type': 'express_clean', 'bays': 4, 'hours': 1},
            {'id': 'CS7', 'type': 'deep_clean', 'bays': 2, 'hours': 6},
            {'id': 'CS8', 'type': 'inspection_clean', 'bays': 1, 'hours': 4}
        ]
        
        for slot in slots_config:
            # Simulate daily availability variations
            available_bays = max(0, slot['bays'] - random.randint(0, 2))
            
            data.append({
                'cleaning_slot_id': slot['id'],
                'available_bays': available_bays,
                'slot_type': slot['type'],
                'estimated_time_hours': slot['hours'],
                'staff_required': random.randint(2, 6),
                'equipment_available': random.choice([True, True, True, False])
            })
        
        return pd.DataFrame(data)
    
    def generate_enhanced_bay_config_data(self):
        """Generate enhanced bay configuration"""
        data = []
        
        # Service bays
        for i in range(1, 9):  # Bay1 to Bay8
            data.append({
                'bay_id': f'Bay{i}',
                'geometry_score': random.randint(6, 10),
                'max_capacity': random.choice([1, 2, 2, 3]),  # Mostly 2-capacity
                'bay_type': 'service',
                'power_available': True,
                'maintenance_equipment': random.choice([True, False]),
                'cleaning_access': random.choice([True, True, False])
            })
        
        # Maintenance bays
        for i in range(9, 12):  # Bay9 to Bay11
            data.append({
                'bay_id': f'Bay{i}',
                'geometry_score': random.randint(4, 8),
                'max_capacity': 1,
                'bay_type': 'maintenance',
                'power_available': random.choice([True, False]),
                'maintenance_equipment': True,
                'cleaning_access': True
            })
        
        # Storage bays
        for i in range(12, 15):  # Bay12 to Bay14
            data.append({
                'bay_id': f'Bay{i}',
                'geometry_score': random.randint(3, 6),
                'max_capacity': random.choice([1, 2]),
                'bay_type': 'storage',
                'power_available': False,
                'maintenance_equipment': False,
                'cleaning_access': False
            })
        
        return pd.DataFrame(data)
    
    def generate_enhanced_historical_outcomes(self):
        """Generate 6 months of historical outcomes data"""
        data = []
        
        # Generate data for last 180 days
        for days_back in range(180, 0, -1):
            date = self.current_time - timedelta(days=days_back)
            
            # Each day, each train has a record
            for train_id in self.train_ids:
                # Induction probability varies by train condition
                inducted = random.choices([True, False], weights=[70, 30])[0]
                
                # Failure probability (lower for inducted trains)
                failure_prob = 0.05 if inducted else 0.15
                actual_failure = random.random() < failure_prob
                
                # Branding SLA compliance
                branding_sla_met = random.choices([True, False], weights=[85, 15])[0]
                
                # Energy consumption (varies by usage)
                energy_consumed = random.randint(140, 200) if inducted else random.randint(50, 100)
                
                # Passenger complaints
                complaints = random.choices([0, 1, 2, 3], weights=[70, 20, 8, 2])[0]
                
                # Punctuality metrics
                on_time_percentage = random.uniform(95.0, 99.8)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'train_id': train_id,
                    'inducted': inducted,
                    'actual_failure_occurred': actual_failure,
                    'branding_sla_met': branding_sla_met,
                    'energy_consumed_kwh': energy_consumed,
                    'passenger_complaints': complaints,
                    'on_time_percentage': round(on_time_percentage, 2),
                    'total_km_covered': random.randint(200, 400) if inducted else 0,
                    'maintenance_cost_inr': random.randint(1000, 8000) if actual_failure else random.randint(0, 2000)
                })
        
        return pd.DataFrame(data)
    
    def generate_branding_contracts_data(self):
        """Generate branding contracts data"""
        advertisers = [
            'Coca-Cola Kerala', 'LuLu Group International', 'Federal Bank Ltd',
            'Marico Kochi Foods', 'Malabar Gold & Diamonds', 'Jos Alukkas Jewellery',
            'Kalyan Jewellers', 'Metro Shoes & Accessories', 'BigBazar Retail',
            'Reliance Digital', 'Airtel Kerala Circle', 'BSNL Kerala'
        ]
        
        data = []
        
        for advertiser in advertisers:
            # Each advertiser gets 2-4 trains
            num_trains = random.randint(2, 4)
            assigned_trains = random.sample(self.train_ids, num_trains)
            
            for train_id in assigned_trains:
                contract_start = self.current_time - timedelta(days=random.randint(30, 180))
                contract_duration = random.randint(90, 365)
                contract_end = contract_start + timedelta(days=contract_duration)
                
                data.append({
                    'advertiser': advertiser,
                    'train_id': train_id,
                    'contract_start': contract_start.strftime('%Y-%m-%d'),
                    'contract_end': contract_end.strftime('%Y-%m-%d'),
                    'min_daily_hours': random.randint(8, 16),
                    'min_monthly_km': random.randint(3000, 8000),
                    'penalty_per_hour_inr': random.randint(500, 2000),
                    'current_exposure_hours': random.randint(5, 18),
                    'monthly_km_achieved': random.randint(2500, 9000),
                    'sla_compliance_percentage': random.randint(75, 100),
                    'revenue_per_day_inr': random.randint(2000, 8000),
                    'penalty_risk_inr': random.randint(0, 15000)
                })
        
        return pd.DataFrame(data)
    
    def generate_iot_sensor_data(self):
        """Generate IoT sensor feeds data"""
        sensor_types = [
            'Door_Sensor', 'Brake_Pressure', 'HVAC_Temperature', 'Battery_Voltage',
            'Axle_Temperature', 'Vibration_Monitor', 'Pantograph_Status', 'Traction_Current',
            'Air_Compressor', 'Wheel_Wear_Sensor'
        ]
        
        data = []
        
        # Generate last 7 days of hourly data
        for hours_back in range(168, 0, -1):  # 7 days * 24 hours
            timestamp = self.current_time - timedelta(hours=hours_back)
            
            for train_id in self.train_ids:
                for sensor_type in sensor_types:
                    # Generate realistic sensor values
                    value, unit, status = self._generate_sensor_reading(sensor_type)
                    
                    data.append({
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'train_id': train_id,
                        'sensor_type': sensor_type,
                        'value': value,
                        'unit': unit,
                        'status': status,
                        'location': f'Car_{random.randint(1, 4)}',
                        'last_calibration': (timestamp - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
                    })
        
        return pd.DataFrame(data)
    
    def _generate_sensor_reading(self, sensor_type):
        """Generate realistic sensor reading based on type"""
        if sensor_type == 'Door_Sensor':
            value = random.choice([0, 1])
            status = 'Normal' if value == 0 else 'Door Open'
            unit = 'status'
        elif sensor_type == 'Brake_Pressure':
            value = round(random.uniform(5.8, 6.2), 2)
            status = 'Normal' if 5.9 <= value <= 6.1 else 'Warning'
            unit = 'bar'
        elif sensor_type == 'HVAC_Temperature':
            value = round(random.uniform(20, 30), 1)
            status = 'Normal' if 22 <= value <= 26 else 'Alert'
            unit = '°C'
        elif sensor_type == 'Battery_Voltage':
            value = round(random.uniform(70, 78), 1)
            status = 'Normal' if value >= 72 else 'Low Voltage'
            unit = 'V'
        elif sensor_type == 'Axle_Temperature':
            value = round(random.uniform(30, 80), 1)
            status = 'Normal' if value <= 70 else 'Overheating'
            unit = '°C'
        else:
            value = round(random.uniform(0, 100), 2)
            status = random.choices(['Normal', 'Warning', 'Alert'], weights=[80, 15, 5])[0]
            unit = 'units'
        
        return value, unit, status
    
    def generate_mileage_balancing_data(self):
        """Generate mileage balancing history for optimization"""
        data = []
        
        for train_id in self.train_ids:
            # Generate last 90 days of mileage data
            cumulative_km = random.randint(10000, 35000)
            
            for days_back in range(90, 0, -1):
                date = self.current_time - timedelta(days=days_back)
                daily_km = random.randint(0, 450)  # 0 for non-service days
                
                # Component wear calculations
                bogie_wear = min(100, cumulative_km / 300)
                brake_wear = min(100, cumulative_km / 250)
                hvac_wear = min(100, cumulative_km / 400)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'train_id': train_id,
                    'daily_km': daily_km,
                    'cumulative_km': cumulative_km,
                    'bogie_wear_percentage': round(bogie_wear, 2),
                    'brake_wear_percentage': round(brake_wear, 2),
                    'hvac_wear_percentage': round(hvac_wear, 2),
                    'next_maintenance_km': max(0, 30000 - cumulative_km),
                    'mileage_priority_score': round(random.uniform(0, 100), 2)
                })
                
                cumulative_km += daily_km
        
        return pd.DataFrame(data)
    
    def generate_all_enhanced_data(self):
        """Generate all enhanced datasets and save to CSV files"""
        print("🚀 Starting Enhanced KMRL IntelliFleet Data Generation...")
        
        datasets = {
            'trains': self.generate_enhanced_trains_data(),
            'job_cards': self.generate_enhanced_job_cards_data(),
            'cleaning_slots': self.generate_enhanced_cleaning_slots_data(),
            'bay_config': self.generate_enhanced_bay_config_data(),
            'historical_outcomes': self.generate_enhanced_historical_outcomes(),
            'branding_contracts': self.generate_branding_contracts_data(),
            'iot_sensor_feeds': self.generate_iot_sensor_data(),
            'mileage_balancing': self.generate_mileage_balancing_data()
        }
        
        # Save all datasets
        for name, df in datasets.items():
            filename = f'data/{name}.csv'
            df.to_csv(filename, index=False)
            print(f"✅ Generated {filename} with {len(df):,} records")
        
        # Generate summary statistics
        self._print_generation_summary(datasets)
        
        return datasets
    
    def _print_generation_summary(self, datasets):
        """Print comprehensive summary of generated data"""
        print("\n" + "="*60)
        print("📊 ENHANCED DATA GENERATION SUMMARY")
        print("="*60)
        
        trains_df = datasets['trains']
        job_cards_df = datasets['job_cards']
        historical_df = datasets['historical_outcomes']
        
        print(f"🚂 TRAINS DATA INSIGHTS:")
        print(f"   Total Trains: {len(trains_df)}")
        
        # Fitness certificate analysis
        current_date = datetime.now()
        expired_count = sum(1 for date_str in trains_df['fitness_valid_until'] 
                          if datetime.strptime(date_str, '%Y-%m-%d') < current_date)
        print(f"   Expired Fitness Certificates: {expired_count}/{len(trains_df)}")
        
        # Mileage distribution
        high_mileage = len(trains_df[trains_df['mileage_km'] > 25000])
        print(f"   High Mileage Trains (>25k km): {high_mileage}")
        
        # Branding coverage
        branded_trains = len(trains_df[trains_df['branding_hours_left'] > 0])
        print(f"   Trains with Active Branding: {branded_trains}")
        
        print(f"\n🔧 JOB CARDS ANALYSIS:")
        open_jobs = len(job_cards_df[job_cards_df['job_card_status'] == 'open'])
        critical_jobs = len(job_cards_df[job_cards_df['maintenance_priority'] == 'critical'])
        print(f"   Open Job Cards: {open_jobs}")
        print(f"   Critical Priority Jobs: {critical_jobs}")
        
        print(f"\n📈 HISTORICAL DATA COVERAGE:")
        print(f"   Days of Historical Data: {len(historical_df['date'].unique())}")
        print(f"   Total Historical Records: {len(historical_df):,}")
        
        induction_rate = historical_df['inducted'].mean() * 100
        failure_rate = historical_df['actual_failure_occurred'].mean() * 100
        print(f"   Average Induction Rate: {induction_rate:.1f}%")
        print(f"   Average Failure Rate: {failure_rate:.1f}%")
        
        print(f"\n🎯 DATA READY FOR:")
        print(f"   ✅ AI-Driven Optimization Testing")
        print(f"   ✅ Constraint Engine Validation")
        print(f"   ✅ Multi-Objective Ranking")
        print(f"   ✅ Historical ML Model Training")
        print(f"   ✅ Dashboard Visualization")
        print(f"   ✅ Real-time Simulation")
        
        print("="*60)


if __name__ == "__main__":
    # Initialize and run the enhanced data generator
    generator = EnhancedKMRLDataGenerator()
    datasets = generator.generate_all_enhanced_data()
    
    print(f"\n🎉 Enhanced data generation completed successfully!")
    print(f"📁 All files saved to 'data/' directory")
    print(f"🚀 Ready to run comprehensive KMRL IntelliFleet system testing!")