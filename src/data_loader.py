import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from .db import init_db, fetch_df, bootstrap_from_csv_if_empty
from .db import upsert_heartbeat

class DataLoader:
    """Layer 0: Real-Time Data Integration Engine"""
    
    def __init__(self, config_path: str = "config/constraints_config.json"):
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.validation_errors = []
        self.data_quality_score = 0.0
        self.use_db = os.environ.get('KMRL_USE_DB', '0') == '1'
        
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, using defaults")
            return {"hard_constraints": {"fitness_certificate": {"enabled": True}}}
    
    def load_trains_data(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            base_dir = getattr(self, 'DATA_PATH', getattr(DataLoader, 'DATA_PATH', None))
            if base_dir is None:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            file_path = os.path.join(base_dir, 'trains.csv')
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
    
    def load_job_cards_data(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            base_dir = getattr(self, 'DATA_PATH', getattr(DataLoader, 'DATA_PATH', None))
            if base_dir is None:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            file_path = os.path.join(base_dir, 'job_cards.csv')
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
    
    def load_cleaning_slots_data(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            base_dir = getattr(self, 'DATA_PATH', getattr(DataLoader, 'DATA_PATH', None))
            if base_dir is None:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            file_path = os.path.join(base_dir, 'cleaning_slots.csv')
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
    
    def load_bay_config_data(self, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            base_dir = getattr(self, 'DATA_PATH', getattr(DataLoader, 'DATA_PATH', None))
            if base_dir is None:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            file_path = os.path.join(base_dir, 'bay_config.csv')
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
        
        if self.use_db:
            # Ensure DB initialized and optionally bootstrap from CSVs
            init_db()
            base_dir = getattr(self, 'DATA_PATH', getattr(DataLoader, 'DATA_PATH', None))
            if base_dir:
                bootstrap_from_csv_if_empty({
                    'trains': os.path.join(base_dir, 'trains.csv'),
                    'job_cards': os.path.join(base_dir, 'job_cards.csv'),
                    'cleaning_slots': os.path.join(base_dir, 'cleaning_slots.csv'),
                    'bay_config': os.path.join(base_dir, 'bay_config.csv')
                })
            trains_df = fetch_df('trains')
            job_cards_df = fetch_df('job_cards')
            cleaning_slots_df = fetch_df('cleaning_slots')
            bay_config_df = fetch_df('bay_config')
        else:
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
        
        result = {
            'trains': self._ensure_depot(trains_df, default='DepotA'),
            'job_cards': job_cards_df,
            'cleaning_slots': cleaning_slots_df,
            'bay_config': self._ensure_depot(bay_config_df, default='DepotA')
        }
        # Heartbeats: fitness and data sources basic signals
        try:
            upsert_heartbeat('fitness_db', 'ok' if not trains_df.empty else 'stale', f"trains={len(trains_df)}")
            upsert_heartbeat('cleaning_slots', 'ok' if not cleaning_slots_df.empty else 'stale', f"slots={len(cleaning_slots_df)}")
            upsert_heartbeat('bay_config', 'ok' if not bay_config_df.empty else 'stale', f"bays={len(bay_config_df)}")
        except Exception:
            pass
        return result

    def load_yard_topology(self) -> Tuple[dict, dict, dict]:
        """Load yard graph and node maps if configured.

        Returns (yard_graph, train_start_nodes, bay_nodes). Any may be empty dicts if not configured.
        """
        # Determine path from env or settings
        path = os.environ.get('KMRL_YARD_TOPOLOGY')
        if not path:
            try:
                from config.settings import SETTINGS
                path = SETTINGS.get('data', {}).get('yard_topology_path')
            except Exception:
                path = None
        if not path:
            return {}, {}, {}
        try:
            with open(path, 'r') as f:
                topo = json.load(f)
            yard_graph = topo.get('yard_graph', {})
            train_start_nodes = topo.get('train_start_nodes', {})
            bay_nodes = topo.get('bay_nodes', {})
            return yard_graph, train_start_nodes, bay_nodes
        except Exception:
            return {}, {}, {}

    def _ensure_depot(self, df: pd.DataFrame, default: str = 'DepotA') -> pd.DataFrame:
        try:
            if df is None or df.empty:
                return df
            if 'depot_id' not in df.columns:
                # Add default depot for multi-depot readiness
                df = df.copy()
                df['depot_id'] = default
            else:
                df['depot_id'] = df['depot_id'].fillna(default).astype(str)
            return df
        except Exception:
            return df
