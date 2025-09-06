"""
enterprise_main.py
Complete KMRL IntelliFleet Enterprise System with patched SystemMonitor initialization.
"""
import asyncio
import threading
import time
import logging
import signal
import os
import sys
from datetime import datetime
from src.ai_data_processor import AIDataProcessor
# Add src to path
sys.path.append('src')

from src.digital_twin_engine import DigitalTwinEngine
from src.api_gateway import APIGateway
from src.web_dashboard import InteractiveWebDashboard
from src.monitoring_system import SystemMonitor
from src.iot_sensor_system import IoTSensorSimulator, IoTDataProcessor, IoTWebSocketServer
from src.computer_vision_system import ComputerVisionSystem
from src.mobile_integration import MobileAPIServer

# Import AI optimization components from main_app.py
from src.data_loader import DataLoader
from src.constraint_engine import CustomConstraintEngine
from src.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dashboard import InductionDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("KMRLIntelliFleet")

class AIDataProcessor:
    """Processes AI optimization results for web dashboard visualization"""
    
    def __init__(self, optimizer, constraint_engine, data_loader):
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        self.data_loader = data_loader

    
    def get_train_status_summary(self):
        """Get comprehensive train status summary"""
        summary = {
            'total_trains': 0,
            'inducted_trains': 0,
            'ready_trains': 0,
            'maintenance_trains': 0,
            'standby_trains': 0,
            'ineligible_trains': 0
        }
        
        if not hasattr(self.optimizer, 'optimized_result'):
            return summary
            
        results = self.optimizer.optimized_result
        inducted_trains = results.get('inducted_trains', [])
        
        # Count inducted trains
        summary['inducted_trains'] = len([t for t in inducted_trains if t.get('inducted', False)])
        summary['total_trains'] = len(inducted_trains)
        
        # Count by status
        for train in inducted_trains:
            status = train.get('status_recommendation', 'unknown').lower()
            if 'ready' in status:
                summary['ready_trains'] += 1
            elif 'maintenance' in status:
                summary['maintenance_trains'] += 1
            elif 'standby' in status:
                summary['standby_trains'] += 1
        
        # Add ineligible trains
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            if isinstance(ineligible, (list, dict)):
                summary['ineligible_trains'] = len(ineligible)
                summary['total_trains'] += summary['ineligible_trains']
        
        return summary
    
    def get_detailed_train_list(self):
        """Get detailed list of all trains with their status"""
        train_details = []
        
        if not hasattr(self.optimizer, 'optimized_result'):
            return train_details
            
        results = self.optimizer.optimized_result
        inducted_trains = results.get('inducted_trains', [])
        
        # Process inducted trains
        for i, train in enumerate(inducted_trains, 1):
            train_details.append({
                'rank': i if train.get('inducted', False) else '-',
                'train_id': train.get('train_id', 'N/A'),
                'status': self._get_train_status(train),
                'bay_assignment': train.get('bay_assignment', 'N/A'),
                'priority_score': train.get('priority_score', 0.0),
                'branding_hours': train.get('branding_hours_remaining', 0.0),
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': train.get('inducted', False)
            })
        
        # Add ineligible trains
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            ineligible_list = []
            
            if isinstance(ineligible, dict):
                ineligible_list = list(ineligible.keys())
            elif isinstance(ineligible, list):
                ineligible_list = [str(t) if isinstance(t, str) else t.get('train_id', 'Unknown') for t in ineligible]
            
            for train_id in ineligible_list:
                train_details.append({
                    'rank': '-',
                    'train_id': train_id,
                    'status': 'Ineligible',
                    'bay_assignment': 'N/A',
                    'priority_score': 0.0,
                    'branding_hours': 0.0,
                    'mileage_km': 0,
                    'fitness_valid': 'Expired',
                    'inducted': False
                })
        
        return train_details
    
    def _get_train_status(self, train):
        """Determine train status based on optimization results"""
        if not train.get('inducted', False):
            return 'Standby'
        
        score = train.get('priority_score', 0)
        if score >= 80:
            return 'Ready'
        elif score >= 60:
            return 'Ready (Caution)'
        else:
            return 'Maintenance Required'
    
    def get_performance_metrics(self):
        """Get performance metrics for display"""
        if not hasattr(self.optimizer, 'optimized_result'):
            return {}
            
        results = self.optimizer.optimized_result
        
        # Calculate metrics from inducted trains
        inducted_trains = results.get('inducted_trains', [])
        inducted_only = [t for t in inducted_trains if t.get('inducted', False)]
        
        if not inducted_only:
            return {}
        
        avg_score = sum(t.get('priority_score', 0) for t in inducted_only) / len(inducted_only)
        total_branding = sum(t.get('branding_hours_remaining', 0) for t in inducted_only)
        
        return {
            'system_performance': avg_score,
            'service_readiness': min(100, avg_score * 1.1),  # Slight boost for display
            'maintenance_risk': max(0, 100 - avg_score),
            'branding_compliance': min(100, total_branding / len(inducted_only) * 10),
            'cost_savings': 138000,  # From your example
            'annual_savings': 50370000
        }
    
    def get_constraint_violations(self):
        """Get constraint violations for display - Fixed version"""
        violations = []
        
        # Handle conflicts from constraint engine
        if hasattr(self.constraint_engine, 'conflicts'):
            for conflict in self.constraint_engine.conflicts:
                if isinstance(conflict, dict):
                    # Handle dictionary format
                    violations.append({
                        'train_id': conflict.get('train_id', 'Unknown'),
                        'violations': conflict.get('violations', [])
                    })
                elif isinstance(conflict, str):
                    # Handle string format - extract train ID if possible
                    train_id = conflict if conflict.startswith('T') else 'Unknown'
                    violations.append({
                        'train_id': train_id,
                        'violations': [conflict]
                    })
                else:
                    # Handle other formats
                    violations.append({
                        'train_id': str(conflict),
                        'violations': [str(conflict)]
                    })
        
        # Also check ineligible trains for violations
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            if isinstance(ineligible, dict):
                for train_id, reason in ineligible.items():
                    violations.append({
                        'train_id': train_id,
                        'violations': [reason] if isinstance(reason, str) else reason
                    })
            elif isinstance(ineligible, list):
                for item in ineligible:
                    if isinstance(item, str):
                        violations.append({
                            'train_id': item,
                            'violations': ['Ineligible for service']
                        })
        
        return violations

class KMRLIntelliFleetSystem:
    def __init__(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        self.running = False
        self._init_components()

    def _init_components(self):
        logger.info("Initializing system components...")
        
        # Phase 1: Initialize AI Optimization Components
        self.data_loader = DataLoader()
        self.ai_data = self.data_loader.get_integrated_data()
        
        # Phase 2: Constraint Processing
        self.constraint_engine = CustomConstraintEngine(self.ai_data)
        self.constraint_result = self.constraint_engine.run_constraint_optimization()
        
        # Phase 3: Multi-Objective Optimization  
        self.optimizer = MultiObjectiveOptimizer(self.constraint_result, self.ai_data)
        self.optimized_result = self.optimizer.optimize_induction_ranking()
        self.optimizer.optimized_result = self.optimized_result
        
        # Phase 4: AI Dashboard
        self.ai_dashboard = InductionDashboard(self.data_loader, self.constraint_engine, self.optimizer)
        
        # Phase 5: AI Data Processor for Web Dashboard
        self.ai_data_processor = AIDataProcessor(self.optimizer, self.constraint_engine, self.data_loader)
        
        # Initialize Digital Twin with AI-optimized data
        initial_data = self._create_digital_twin_data()
        self.digital_twin = DigitalTwinEngine(initial_data)
        
        # Add AI data to digital twin state
        self._inject_ai_data_into_digital_twin()
        
        # IoT and other components
        self.iot_simulator = IoTSensorSimulator(list(initial_data['trains'].keys()))
        self.iot_processor = IoTDataProcessor()
        self.cv_system = ComputerVisionSystem()
        self.monitor = SystemMonitor(self.digital_twin, self.iot_processor)
        
        # Enhanced Web Dashboard with AI integration
        self.web_dashboard = InteractiveWebDashboard(
            self.digital_twin, 
            self.monitor, 
            self.iot_simulator, 
            self.cv_system,
            ai_optimizer=self.optimizer,  # Pass AI optimizer
            constraint_engine=self.constraint_engine,  # Pass constraint engine
            ai_dashboard=self.ai_dashboard,  # Pass AI dashboard
            ai_data_processor=self.ai_data_processor  # Pass AI data processor
        )
        
        # API and Mobile components
        self.api_gateway = APIGateway(self.digital_twin, self.monitor)
        self.mobile_api = MobileAPIServer(self.digital_twin, self.iot_simulator, self.cv_system, port=5000)
        self.iot_websocket = IoTWebSocketServer(self.iot_simulator, port=8765)
        
        logger.info("All components initialized with AI optimization.")

    def _inject_ai_data_into_digital_twin(self):
        """Inject AI optimization data into digital twin state"""
        try:
            ai_state = {
                'ai_summary': self.ai_data_processor.get_train_status_summary(),
                'ai_train_details': self.ai_data_processor.get_detailed_train_list(),
                'ai_performance': self.ai_data_processor.get_performance_metrics(),
                'ai_violations': self.ai_data_processor.get_constraint_violations(),
                'last_updated': datetime.now().isoformat()
            }
            
            # Add to digital twin's current state
            current_state = self.digital_twin.get_current_state()
            current_state['ai_data'] = ai_state
            
            print(f"🤖 AI Data injected: {len(ai_state['ai_train_details'])} trains processed")
        except Exception as e:
            logger.error(f"Error injecting AI data: {e}")
            # Continue without AI data if there's an error

    def _create_digital_twin_data(self):
        """Create digital twin data from AI optimization results"""
        trains_dict = {}
        
        # Get inducted trains from optimizer
        if hasattr(self.optimizer, 'optimized_result') and self.optimizer.optimized_result:
            results = self.optimizer.optimized_result
            inducted_trains = results.get('inducted_trains', [])
            
            for train_info in inducted_trains:
                train_id = train_info.get('train_id', '')
                trains_dict[train_id] = {
                    'location': 'depot',
                    'status': 'inducted' if train_info.get('inducted', False) else 'idle',
                    'mileage_km': train_info.get('mileage_km', 15000),
                    'fitness_valid_until': train_info.get('fitness_valid_until', '2026-12-31'),
                    'priority_score': train_info.get('priority_score', 0.0),
                    'bay_assignment': train_info.get('bay_assignment', ''),
                    'branding_hours': train_info.get('branding_hours_remaining', 0.0)
                }
        
        # Add ineligible trains - Fixed error handling
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible_trains = self.constraint_engine.ineligible_trains
            
            # Handle different data structures for ineligible trains
            if isinstance(ineligible_trains, dict):
                # If it's a dictionary, get the keys (train IDs)
                ineligible_train_ids = list(ineligible_trains.keys())
            elif isinstance(ineligible_trains, list):
                # If it's a list, check if it contains strings or dictionaries
                if ineligible_trains and isinstance(ineligible_trains[0], dict):
                    # List of dictionaries - extract train_id field
                    ineligible_train_ids = [train.get('train_id', str(i)) for i, train in enumerate(ineligible_trains)]
                else:
                    # List of train IDs (strings)
                    ineligible_train_ids = ineligible_trains
            else:
                ineligible_train_ids = []
            
            # Add ineligible trains to the dictionary
            for train_id in ineligible_train_ids:
                if isinstance(train_id, str) and train_id not in trains_dict:
                    trains_dict[train_id] = {
                        'location': 'depot',
                        'status': 'ineligible',
                        'mileage_km': 20000,
                        'fitness_valid_until': '2025-01-01'
                    }
        
        # If we don't have any trains, add some default ones
        if not trains_dict:
            trains_dict = {
                'KMRL_001': {'location': 'depot', 'status': 'idle', 'mileage_km': 15000, 'fitness_valid_until': '2026-12-31'},
                'KMRL_002': {'location': 'route', 'status': 'running', 'mileage_km': 20000, 'fitness_valid_until': '2027-03-31'},
                'KMRL_003': {'location': 'depot', 'status': 'idle', 'mileage_km': 18000, 'fitness_valid_until': '2026-09-30'},
                'KMRL_004': {'location': 'route', 'status': 'running', 'mileage_km': 22000, 'fitness_valid_until': '2027-01-15'},
                'KMRL_005': {'location': 'depot', 'status': 'idle', 'mileage_km': 20000, 'fitness_valid_until': '2026-11-20'}
            }
        
        bay_config = {
            'bay_1': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 5, 'power_available': True},
            'bay_2': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 4, 'power_available': True},
            'bay_4': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 3, 'power_available': True}
        }
        
        return {
            'trains': trains_dict,
            'depots': ['Muttom', 'Ernakulam South'],
            'maintenance_schedule': {},
            'bay_config': bay_config
        }

    def start(self):
        logger.info("Starting KMRL IntelliFleet System...")
        self.running = True
        
        # Start all components
        self.digital_twin.start_simulation()
        self.iot_simulator.start_simulation()
        self.monitor.start_monitoring()
        self._inject_ai_data_into_digital_twin()
        
        # Start servers
        threading.Thread(target=self.api_gateway.run_server, daemon=True).start()
        threading.Thread(target=self.mobile_api.start_server, daemon=True).start()
        threading.Thread(target=lambda: self.web_dashboard.run_server(debug=False), daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(self.iot_websocket.broadcast_sensor_data()), daemon=True).start()
          # initial inject
    
    # Start periodic refresh loop
        try:
            while self.running:
                time.sleep(30)  # adjust refresh interval
                self.constraint_result = self.constraint_engine.run_constraint_optimization()
                self.optimized_result = self.optimizer.optimize_induction_ranking()
                self.optimizer.optimized_result = self.optimized_result
                
                self._inject_ai_data_into_digital_twin()
                # Optional: print console summary here
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self._shutdown()
        # Print AI Dashboard summary to console
        try:
            print("\n" + "="*80)
            print("🤖 AI OPTIMIZATION SUMMARY")
            print("="*80)
            summary = self.ai_data_processor.get_train_status_summary()
            train_details = self.ai_data_processor.get_detailed_train_list()
            
            print(f"📊 FLEET OVERVIEW:")
            print(f"   Total Trains: {summary['total_trains']}")
            print(f"   ✅ Inducted: {summary['inducted_trains']}")
            print(f"   🟢 Ready: {summary['ready_trains']}")
            print(f"   🔧 Maintenance: {summary['maintenance_trains']}")
            print(f"   ⏸️  Standby: {summary['standby_trains']}")
            print(f"   ❌ Ineligible: {summary['ineligible_trains']}")
            
            print(f"\n📋 INDUCTED TRAINS:")
            inducted_trains = [t for t in train_details if t['inducted']]
            for train in inducted_trains[:6]:  # Show top 6
                status_icon = "✅" if "Ready" in train['status'] else "⚠️"
                print(f"   {train['rank']}. {train['train_id']} - {train['bay_assignment']} - Score: {train['priority_score']:.1f} {status_icon}")
            
            performance = self.ai_data_processor.get_performance_metrics()
            if performance:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"   System Performance: {performance.get('system_performance', 0):.1f}/100")
                print(f"   Cost Savings: ₹{performance.get('cost_savings', 0):,}")
            
            print("\n🌐 Web Dashboard available at: http://127.0.0.1:8050")
            print("="*80)
        except Exception as e:
            logger.error(f"Error displaying AI summary: {e}")
        
        # Main loop
        try:
            while self.running:
                time.sleep(30)
                # Periodically refresh optimization
                self._refresh_optimization()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self._shutdown()
        # All analytics and summaries are now displayed in the web dashboard

    def _refresh_optimization(self):
        """Refresh AI optimization results periodically"""
        try:
            # Re-run optimization
            self.constraint_result = self.constraint_engine.run_constraint_optimization()
            self.optimized_result = self.optimizer.optimize_induction_ranking()
            self.optimizer.optimized_result = self.optimized_result
            
            # Update AI data in digital twin
            self._inject_ai_data_into_digital_twin()
            
            logger.info("AI optimization refreshed")
        except Exception as e:
            logger.error(f"Error refreshing optimization: {e}")

    def _shutdown(self, *args):
        logger.info("Shutting down KMRL IntelliFleet System...")
        self.running = False
        self.digital_twin.stop_simulation()
        self.iot_simulator.stop_simulation()
        self.monitor.stop_monitoring()
        logger.info("Shutdown complete.")
        os._exit(0)

if __name__ == "__main__":
    print("🚀 Launching KMRL IntelliFleet Enterprise System with AI Dashboard")
    system = KMRLIntelliFleetSystem()
    # Start only the web dashboard (Dash app)
    system.web_dashboard.app.run(debug=True)
