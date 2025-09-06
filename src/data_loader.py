import pandas as pd
import json
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Layer 0: Real-Time Data Integration Engine"""
    
    def __init__(self, config_path: str = "config/constraints_config.json"):
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.validation_errors = []
        self.data_quality_score = 0.0
        
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, using defaults")
            return {"hard_constraints": {"fitness_certificate": {"enabled": True}}}
    
    def load_trains_data(self, file_path: str = "data/trains.csv") -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.data_sources['trains'] = df
            required_columns = ['train_id', 'fitness_valid_until', 'mileage_km', 
                              'branding_hours_left', 'cleaning_slot_id', 'bay_geometry_score']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.validation_errors.append(f"Missing columns in trains.csv: {missing_cols}")
                return pd.DataFrame()
            
            df['fitness_valid_until'] = pd.to_datetime(df['fitness_valid_until'])
            df['mileage_km'] = pd.to_numeric(df['mileage_km'], errors='coerce')
            df['branding_hours_left'] = pd.to_numeric(df['branding_hours_left'], errors='coerce')
            df['bay_geometry_score'] = pd.to_numeric(df['bay_geometry_score'], errors='coerce')
            
            today = datetime.now().date()
            expired_count = len(df[df['fitness_valid_until'].dt.date < today])
            high_mileage_count = len(df[df['mileage_km'] > 30000])
            no_cleaning_count = len(df[df['cleaning_slot_id'] == 'None'])
            
            print(f"📊 Trains Data Loaded: {len(df)} trains")
            print(f"⚠️ Validation Results:")
            print(f"   - Expired fitness: {expired_count} trains")
            print(f"   - High mileage (>30k km): {high_mileage_count} trains")  
            print(f"   - No cleaning slot: {no_cleaning_count} trains")
            
            return df
        except Exception as e:
            self.validation_errors.append(f"Error loading trains data: {str(e)}")
            return pd.DataFrame()
    
    def load_job_cards_data(self, file_path: str = "data/job_cards.csv") -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.data_sources['job_cards'] = df
            required_columns = ['train_id', 'job_card_status']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.validation_errors.append(f"Missing columns in job_cards.csv: {missing_cols}")
                return pd.DataFrame()
            
            open_count = len(df[df['job_card_status'] == 'open'])
            print(f"📋 Job Cards Loaded: {len(df)} records")
            print(f"⚠️ Open job cards: {open_count} trains")
            return df
        except Exception as e:
            self.validation_errors.append(f"Error loading job cards data: {str(e)}")
            return pd.DataFrame()
    
    def load_cleaning_slots_data(self, file_path: str = "data/cleaning_slots.csv") -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.data_sources['cleaning_slots'] = df
            available_slots = df[df['available_bays'] > 0]
            total_capacity = df['available_bays'].sum()
            print(f"🧹 Cleaning Slots Loaded: {len(available_slots)} available slots")
            print(f"📊 Total cleaning capacity: {total_capacity} bays")
            return df
        except Exception as e:
            self.validation_errors.append(f"Error loading cleaning slots data: {str(e)}")
            return pd.DataFrame()
    
    def load_bay_config_data(self, file_path: str = "data/bay_config.csv") -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.data_sources['bay_config'] = df
            service_bays = df[df['bay_type'] == 'service']
            total_capacity = df['max_capacity'].sum()
            print(f"🏗️ Bay Config Loaded: {len(service_bays)} service bays")
            print(f"📊 Total bay capacity: {total_capacity} trains")
            return df
        except Exception as e:
            self.validation_errors.append(f"Error loading bay config data: {str(e)}")
            return pd.DataFrame()
    
    def get_integrated_data(self) -> Dict[str, pd.DataFrame]:
        print("🔄 Loading all data sources...")
        
        trains_df = self.load_trains_data()
        job_cards_df = self.load_job_cards_data()
        cleaning_slots_df = self.load_cleaning_slots_data()
        bay_config_df = self.load_bay_config_data()
        
        total_sources = 4
        loaded_sources = sum([1 for df in [trains_df, job_cards_df, cleaning_slots_df, bay_config_df] 
                             if not df.empty])
        self.data_quality_score = (loaded_sources / total_sources) * 100
        
        print(f"\n📊 Data Integration Summary:")
        print(f"   - Sources loaded: {loaded_sources}/{total_sources}")
        print(f"   - Data quality score: {self.data_quality_score:.1f}%")
        print(f"   - Validation errors: {len(self.validation_errors)}")
        
        if self.validation_errors:
            print("⚠️ Validation Errors:")
            for error in self.validation_errors:
                print(f"   - {error}")
        
        return {
            'trains': trains_df,
            'job_cards': job_cards_df,
            'cleaning_slots': cleaning_slots_df,
            'bay_config': bay_config_df
        }
