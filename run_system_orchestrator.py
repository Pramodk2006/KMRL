#!/usr/bin/env python3
"""
KMRL IntelliFleet System Orchestrator
Complete system initialization and coordination script that properly wires
all components with correct data contracts and field mappings.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Core imports
from src.data_loader import DataLoader
from src.constraint_engine import CustomConstraintEngine
from src.multi_objective_optimizer import MultiObjectiveOptimizer
from src.enhanced_optimizer import EnhancedMultiObjectiveOptimizer
from src.digital_twin_engine import DigitalTwinEngine
from src.ai_data_processor import AIDataProcessor
from src.monitoring_system import SystemMonitor
from src.api_gateway import APIGateway
from src.iot_sensor_system import IoTSensorSimulator, IoTWebSocketServer
from src.computer_vision_system import ComputerVisionSystem
from src.mobile_integration import MobileAPIServer
import os
from src.maximo_adapter import MaximoAdapter
from src.branding_sla import ingest_contracts_from_csv, record_exposure_for_inductions


# Dashboard imports
from combined_dashboard import CombinedKMRLDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KMRLIntelliFleetOrchestrator:
    """
    Complete system orchestrator that initializes and coordinates all components
    of the KMRL IntelliFleet system with proper data contracts.
    """
    
    def __init__(self, config_path: str = "config/settings.py"):
        self.config_path = config_path
        self.components = {}
        self.services = {}
        self.is_running = False
        
        # Initialize all components
        self._initialize_components()
        self._wire_components()
        
    def _initialize_components(self):
        """Initialize all system components with proper dependencies."""
        
        logger.info("🚀 Initializing KMRL IntelliFleet System...")
        
        # 1. Data Layer - Load and validate data
        logger.info("📊 Loading data...")
        # Enable DB-backed loader via env toggle
        os.environ.setdefault('KMRL_USE_DB', '1')
        self.components['data_loader'] = DataLoader()
        # Pre-run Maximo refresh if export path configured
        export_path = os.environ.get('KMRL_MAXIMO_EXPORT')
        if export_path:
            try:
                logger.info("🔄 Refreshing job-cards from Maximo export...")
                refreshed = MaximoAdapter(export_path).refresh()
                logger.info(f"✅ Maximo refresh complete: {refreshed} records")
            except Exception as e:
                logger.warning(f"Maximo refresh skipped due to error: {e}")
        # Optional: ingest branding contracts from CSV
        branding_csv = os.environ.get('KMRL_BRANDING_CONTRACTS')
        if branding_csv:
            try:
                cnt = ingest_contracts_from_csv(branding_csv)
                logger.info(f"📝 Branding contracts ingested: {cnt}")
            except Exception as e:
                logger.warning(f"Branding contracts ingest failed: {e}")
        data_dict = self.components['data_loader'].get_integrated_data()
        
        if not data_dict:
            raise RuntimeError("Failed to load required data files")
            
        # 2. Constraint Engine - Hard constraints and eligibility
        logger.info("⚖️ Initializing constraint engine...")
        self.components['constraint_engine'] = CustomConstraintEngine(data=data_dict)
        constraint_result = self.components['constraint_engine'].run_constraint_optimization()
        
        # 3. Multi-Objective Optimizer - Base optimization
        logger.info("🎯 Setting up optimization engines...")
        self.components['base_optimizer'] = MultiObjectiveOptimizer(
            constraint_result=constraint_result,
            data=data_dict
        )
        base_result = self.components['base_optimizer'].optimize_induction_ranking()
        self.components['base_optimizer'].optimized_result = base_result
        
        # 4. Enhanced AI Optimizer - Predictive optimization
        self.components['enhanced_optimizer'] = EnhancedMultiObjectiveOptimizer(
            constraint_result=constraint_result,
            data=data_dict
        )
        
        # Create historical data for predictive model
        historical_df = self._create_historical_data(data_dict['trains'])
        enhanced_result = self.components['enhanced_optimizer'].optimize_with_ai(historical_df)
        self.components['enhanced_optimizer'].optimized_result = enhanced_result
        
        # 5. AI Data Processor - UI data transformation (FIXED version)
        logger.info("🧠 Setting up AI data processor...")
        self.components['ai_data_processor'] = AIDataProcessor(
            optimizer=self.components['enhanced_optimizer'],
            constraint_engine=self.components['constraint_engine'],
            data_loader=self.components['data_loader']
        )
        
        # 6. Digital Twin Engine - Simulation and state management
        logger.info("🔮 Initializing digital twin...")
        self.components['digital_twin'] = self._initialize_digital_twin(
            data_dict['trains'], data_dict.get('bay_config', {})
        )
        
        # 7. IoT Sensor System - Real-time monitoring
        logger.info("📡 Setting up IoT systems...")
        train_ids = list(data_dict['trains']['train_id'].unique())
        self.components['iot_simulator'] = IoTSensorSimulator(train_ids=train_ids)
        self.services['iot_websocket'] = IoTWebSocketServer(
            self.components['iot_simulator']
        )
        
        # 8. Computer Vision System - Visual inspection
        logger.info("👁️ Initializing computer vision...")
        self.components['cv_system'] = ComputerVisionSystem()
        
        # 9. Monitoring System - Metrics and alerts
        logger.info("📊 Setting up monitoring...")
        self.components['monitoring'] = SystemMonitor(
            digital_twin_engine=self.components['digital_twin'],
            ai_optimizer=self.components['enhanced_optimizer']
        )
        
        logger.info("✅ All components initialized successfully!")
        
    def _create_historical_data(self, trains_df) -> Any:
        """Create synthetic historical data for predictive model training."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create 30 days of historical data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        historical_records = []
        for date in dates:
            for _, train in trains_df.iterrows():
                record = {
                    'date': date,
                    'train_id': train['train_id'],
                    'mileage_km': train['mileage_km'] - np.random.randint(0, 100),
                    'branding_hours_left': train['branding_hours_left'] + np.random.randint(0, 24),
                    'fitness_valid_until': train['fitness_valid_until'],
                    'cleaning_slot_id': train['cleaning_slot_id'],
                    'bay_geometry_score': train['bay_geometry_score'],
                    'last_maintenance_date': date - timedelta(days=np.random.randint(1, 30)),
                    'actual_failure_occurred': np.random.choice([0, 1], p=[0.95, 0.05]),
                    'temperature': 20 + np.random.normal(0, 5),
                    'humidity': 60 + np.random.normal(0, 10),
                    'season': self._get_season(date)
                }
                historical_records.append(record)
                
        return pd.DataFrame(historical_records)
    
    def _get_season(self, date):
        """Get season for a given date."""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'monsoon'
    
    def _initialize_digital_twin(self, trains_df, bay_config_df) -> DigitalTwinEngine:
        """Initialize digital twin with proper data contracts."""
        
        # Convert trains DataFrame to digital twin format
        trains_dict = {}
        for _, train in trains_df.iterrows():
            trains_dict[train['train_id']] = {
                'location': f"Platform_{train['train_id'][-1]}",
                'status': 'idle',
                'mileage_km': float(train['mileage_km']),
                'branding_hours_left': int(train['branding_hours_left']),
                'fitness_valid_until': str(train['fitness_valid_until']),
                'cleaning_slot_id': str(train['cleaning_slot_id']),
                'bay_geometry_score': float(train['bay_geometry_score']),
                'failure_probability': 0.05,  # Default low risk
                'depot_id': str(train.get('depot_id', 'DepotA'))
            }
        
        # Convert bay config to digital twin format
        bays_dict = {}
        if not bay_config_df.empty:
            for _, bay in bay_config_df.iterrows():
                bays_dict[bay['bay_id']] = {
                    'bay_type': bay['bay_type'],
                    'max_capacity': int(bay['max_capacity']),
                    'geometry_score': float(bay['geometry_score']),
                    'power_available': True,
                    'status': 'available',
                    'depot_id': str(bay.get('depot_id', 'DepotA'))
                }
        else:
            # Create default bay configuration
            default_bays = [
                {'bay_id': 'SB001', 'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 0.95},
                {'bay_id': 'SB002', 'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 0.90},
                {'bay_id': 'SB003', 'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 0.85},
                {'bay_id': 'MB001', 'bay_type': 'maintenance', 'max_capacity': 1, 'geometry_score': 0.80},
                {'bay_id': 'MB002', 'bay_type': 'maintenance', 'max_capacity': 1, 'geometry_score': 0.75},
                {'bay_id': 'STB001', 'bay_type': 'storage', 'max_capacity': 4, 'geometry_score': 0.70}
            ]
            
            for bay in default_bays:
                bays_dict[bay['bay_id']] = {
                    'bay_type': bay['bay_type'],
                    'max_capacity': bay['max_capacity'],
                    'geometry_score': bay['geometry_score'],
                    'power_available': True,
                    'status': 'available'
                }
        
        # Initialize digital twin with initial data contract
        initial_data = {
            'trains': trains_dict,
            'bay_config': bays_dict
        }
        twin = DigitalTwinEngine(initial_data)
        return twin
    
    def _wire_components(self):
        """Wire components together with proper data contracts."""
        
        logger.info("🔗 Wiring components...")
        
        # Wire digital twin with monitoring
        self.components['monitoring'].digital_twin_engine = self.components['digital_twin']
        
        # Wire AI data processor with all dependencies
        ai_processor = self.components['ai_data_processor']
        ai_processor.optimizer = self.components['enhanced_optimizer']
        ai_processor.constraint_engine = self.components['constraint_engine']
        ai_processor.data_loader = self.components['data_loader']
        
        # Execute induction plan in digital twin
        if hasattr(self.components['enhanced_optimizer'], 'optimized_result'):
            inducted_trains = self.components['enhanced_optimizer'].optimized_result.get('inducted_trains', [])
            if inducted_trains:
                plan_items = [
                    {
                        'train_id': train['train_id'],
                        'assigned_bay': train.get('assigned_bay', 'SB001'),
                        'estimated_duration': 120  # 2 hours default
                    }
                    for train in inducted_trains[:3]  # Limit to first 3
                ]
                induction_plan = { 'inducted_trains': plan_items }
                self.components['digital_twin'].execute_induction_plan(induction_plan)
                # Record branding exposure for tonight's inducted trains
                try:
                    record_exposure_for_inductions(inducted_trains, hours=8.0)
                except Exception as e:
                    logger.warning(f"Branding exposure logging failed: {e}")
        
        logger.info("✅ Components wired successfully!")
    
    def start_services(self):
        """Start all background services."""
        
        logger.info("🚀 Starting background services...")
        
        # Start IoT simulation
        if 'iot_simulator' in self.components:
            self.components['iot_simulator'].start_simulation()
            
        # Start IoT WebSocket server
        if 'iot_websocket' in self.services:
            ws_thread = threading.Thread(
                target=self.services['iot_websocket'].start_server,
                daemon=True
            )
            ws_thread.start()
            
        # Start digital twin simulation
        if 'digital_twin' in self.components:
            self.components['digital_twin'].start_simulation(time_multiplier=1.0)
            
        # Start monitoring system
        if 'monitoring' in self.components:
            monitor_thread = threading.Thread(
                target=self.components['monitoring'].start_monitoring,
                daemon=True
            )
            monitor_thread.start()
            
        logger.info("✅ Background services started!")
        
    def start_data_management_dashboard(self, host='127.0.0.1', port=8051):
        """Start the data management dashboard."""
        
        logger.info(f"📊 Starting data management dashboard at http://{host}:{port}")
        
        try:
            from src.data_management_dashboard import DataManagementDashboard
            dashboard = DataManagementDashboard()
            dashboard.run(host=host, port=port, debug=False)
        except Exception as e:
            logger.error(f"Failed to start data management dashboard: {e}")
    
    def start_web_dashboard(self, host='127.0.0.1', port=8050, debug=False):
        """Start the combined Dash web dashboard."""
        
        logger.info(f"🖥️ Starting web dashboard at http://{host}:{port}")
        
        # Create combined dashboard with all components
        dashboard = CombinedKMRLDashboard(
            self.components['digital_twin'],
            self.components.get('monitoring'),
            self.components.get('iot_simulator'),
            self.components.get('cv_system'),
            ai_optimizer=self.components.get('enhanced_optimizer'),
            constraint_engine=self.components.get('constraint_engine'),
            ai_data_processor=self.components.get('ai_data_processor')
        )
        
        # Run dashboard
        dashboard.run(host=host, port=port, debug=debug)
        
    def start_api_gateway(self, host='127.0.0.1', port=8000):
        """Start the API gateway server."""
        
        logger.info(f"🔌 Starting API gateway at http://{host}:{port}")
        
        # Create API gateway
        api_gateway = APIGateway(
            digital_twin_engine=self.components['digital_twin'],
            ai_optimizer=self.components.get('enhanced_optimizer')
        )
        
        # Start server
        api_gateway.run_server(host=host, port=port)
        
    def start_mobile_server(self, host='127.0.0.1', port=8080):
        """Start the mobile integration server."""
        
        logger.info(f"📱 Starting mobile server at http://{host}:{port}")
        
        # Create mobile server
        mobile_server = MobileAPIServer(
            digital_twin=self.components['digital_twin'],
            iot_simulator=self.components.get('iot_simulator'),
            cv_system=self.components.get('cv_system'),
            port=port
        )
        
        # Start server
        mobile_server.start_server()
        
    def start_complete_system(self, 
                            dashboard_port=8050, 
                            api_port=8000, 
                            mobile_port=8080,
                            host='127.0.0.1'):
        """Start the complete system with all services."""
        
        logger.info("🎯 Starting complete KMRL IntelliFleet system...")
        
        # Start background services first
        self.start_services()
        
        # Give services time to initialize
        time.sleep(2)
        
        # Start API gateway in separate thread
        api_thread = threading.Thread(
            target=self.start_api_gateway,
            args=(host, api_port),
            daemon=True
        )
        api_thread.start()
        
        # Start mobile server in separate thread
        mobile_thread = threading.Thread(
            target=self.start_mobile_server,
            args=(host, mobile_port),
            daemon=True
        )
        mobile_thread.start()
        
        # Give servers time to start
        time.sleep(3)
        
        logger.info("🌟 System Status:")
        logger.info(f"📊 Web Dashboard: http://{host}:{dashboard_port}")
        logger.info(f"📈 Data Management: http://{host}:8051")
        logger.info(f"🔌 API Gateway: http://{host}:{api_port}")
        logger.info(f"📱 Mobile Server: http://{host}:{mobile_port}")
        logger.info(f"📡 IoT WebSocket: ws://{host}:8765")
        logger.info("🎉 KMRL IntelliFleet System is fully operational!")
        
        # Start data management dashboard in separate thread
        data_mgmt_thread = threading.Thread(
            target=self.start_data_management_dashboard,
            args=(host, 8051),
            daemon=True
        )
        data_mgmt_thread.start()
        
        # Start web dashboard (blocking - main thread)
        self.start_web_dashboard(host=host, port=dashboard_port, debug=False)
        
    def stop_system(self):
        """Gracefully stop all system components."""
        
        logger.info("🛑 Stopping KMRL IntelliFleet system...")
        
        # Stop digital twin
        if 'digital_twin' in self.components:
            self.components['digital_twin'].stop_simulation()
            
        # Stop IoT simulation
        if 'iot_simulator' in self.components:
            self.components['iot_simulator'].stop_simulation()
            
        # Stop monitoring
        if 'monitoring' in self.components:
            self.components['monitoring'].stop_monitoring()
            
        self.is_running = False
        logger.info("✅ System stopped successfully!")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health."""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_operational': True,
            'components': {}
        }
        
        # Check each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'is_running'):
                    status['components'][name] = {
                        'status': 'running' if component.is_running else 'stopped',
                        'healthy': True
                    }
                else:
                    status['components'][name] = {
                        'status': 'active',
                        'healthy': True
                    }
            except Exception as e:
                status['components'][name] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                
        # Get digital twin summary
        if 'digital_twin' in self.components:
            try:
                twin_state = self.components['digital_twin'].get_current_state()
                status['fleet_summary'] = twin_state.get('summary', {})
            except Exception as e:
                logger.error(f"Error getting digital twin state: {e}")
                
        return status


def main():
    """Main entry point for the KMRL IntelliFleet system."""
    
    print("=" * 60)
    print("🚄 KMRL IntelliFleet System Orchestrator")
    print("   Comprehensive Train Induction Optimization Platform")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = KMRLIntelliFleetOrchestrator()
        
        # Start complete system
        orchestrator.start_complete_system(
            dashboard_port=8050,
            api_port=8000,
            mobile_port=8080,
            host='127.0.0.1'
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
        if 'orchestrator' in locals():
            orchestrator.stop_system()
        print("👋 KMRL IntelliFleet system stopped. Goodbye!")
        
    except Exception as e:
        print(f"❌ System startup failed: {e}")
        logger.error(f"System startup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()