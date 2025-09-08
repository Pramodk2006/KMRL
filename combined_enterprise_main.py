"""
KMRL IntelliFleet Combined Enterprise Main
Complete system with Combined Dashboard featuring both Animated Simulation and Classical Analytics UI
"""

import asyncio
import threading
import time
import logging
import signal
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.digital_twin_engine import DigitalTwinEngine
from src.api_gateway import APIGateway
from src.monitoring_system import SystemMonitor
from src.iot_sensor_system import IoTSensorSimulator, IoTDataProcessor, IoTWebSocketServer
from src.computer_vision_system import ComputerVisionSystem
from src.mobile_integration import MobileAPIServer

# Import AI optimization components
from src.data_loader import DataLoader
from src.constraint_engine import CustomConstraintEngine
from src.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dashboard import InductionDashboard

# Import the new combined dashboard
from combined_dashboard import CombinedKMRLDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("KMRLIntelliFleetCombined")

class AIDataProcessor:
    """FIXED: Processes AI optimization results for web dashboard visualization"""
    
    # Cost Calculation Constants
    BASE_SAVINGS_PER_TRAIN = 23000  # Base operational savings per optimally inducted train
    ANNUAL_OPERATIONAL_FACTOR = 0.6  # 60% of year achieves similar optimization
    MAINTENANCE_SAVINGS_PER_RISK_POINT = 500  # Savings per maintenance risk point reduced

    def __init__(self, optimizer, constraint_engine, data_loader):
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        self.data_loader = data_loader

    def get_train_status_summary(self):
        """FIXED: Get comprehensive train status summary"""
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
        
        # FIXED: All trains in inducted_trains list are considered inducted
        inducted_trains = results.get('inducted_trains', [])
        summary['inducted_trains'] = len(inducted_trains)  # All trains in this list are inducted
        
        # Count by status for inducted trains
        for train in inducted_trains:
            priority_score = train.get('priority_score', 0)
            if priority_score >= 80:
                summary['ready_trains'] += 1
            elif priority_score >= 60:
                summary['ready_trains'] += 1  # Ready (Caution) still counts as ready
            else:
                summary['maintenance_trains'] += 1
        
        # Add standby trains (these are in standby_trains list)
        standby_trains = results.get('standby_trains', [])
        summary['standby_trains'] = len(standby_trains)
        
        # Add ineligible trains from constraint engine
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            if isinstance(ineligible, (list, dict)):
                summary['ineligible_trains'] = len(ineligible)
        
        # Calculate total trains
        summary['total_trains'] = (summary['inducted_trains'] + 
                                 summary['standby_trains'] + 
                                 summary['ineligible_trains'])
        
        return summary

    def get_detailed_train_list(self):
        """FIXED: Get detailed list of all trains with their status"""
        train_details = []
        
        if not hasattr(self.optimizer, 'optimized_result'):
            return train_details

        results = self.optimizer.optimized_result
        
        # FIXED: Process inducted trains (these are ALL inducted)
        inducted_trains = results.get('inducted_trains', [])
        for i, train in enumerate(inducted_trains, 1):
            train_details.append({
                'rank': i,  # All trains in inducted_trains list get a rank
                'train_id': train.get('train_id', 'N/A'),
                'status': self._get_train_status(train),
                'bay_assignment': train.get('assigned_bay', 'N/A'),  # FIXED: use 'assigned_bay'
                'priority_score': train.get('priority_score', 0.0),
                'branding_hours': train.get('branding_hours_left', 0.0),  # FIXED: use correct field name
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': True  # FIXED: All trains in this list are inducted
            })
        
        # Process standby trains
        standby_trains = results.get('standby_trains', [])
        for train in standby_trains:
            train_details.append({
                'rank': '-',
                'train_id': train.get('train_id', 'N/A'),
                'status': 'Standby',
                'bay_assignment': 'N/A',
                'priority_score': train.get('priority_score', 0.0),
                'branding_hours': train.get('branding_hours_left', 0.0),
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': False
            })
        
        # Add ineligible trains
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            if isinstance(ineligible, list):
                for train in ineligible:
                    if isinstance(train, dict):
                        train_id = train.get('train_id', 'Unknown')
                    else:
                        train_id = str(train)
                    
                    train_details.append({
                        'rank': '-',
                        'train_id': train_id,
                        'status': 'Ineligible',
                        'bay_assignment': 'N/A',
                        'priority_score': 0.0,
                        'branding_hours': 0.0,
                        'mileage_km': 0,
                        'fitness_valid': 'Expired/Invalid',
                        'inducted': False
                    })
        
        return train_details

    def _get_train_status(self, train):
        """FIXED: Determine train status based on optimization results"""
        score = train.get('priority_score', 0)
        if score >= 80:
            return 'Ready'
        elif score >= 60:
            return 'Ready (Caution)'
        else:
            return 'Maintenance Required'

    def get_performance_metrics(self):
        """FIXED: Get performance metrics for display - with authentic cost calculations"""
        if not hasattr(self.optimizer, 'optimized_result'):
            return {
                'system_performance': 0,
                'service_readiness': 0,
                'maintenance_risk': 100,
                'branding_compliance': 0,
                'cost_savings': 0,
                'annual_savings': 0,
                'trains_processed': 0,
                'calculation_basis': 'No AI optimization data available'
            }

        results = self.optimizer.optimized_result
        inducted_trains = results.get('inducted_trains', [])
        
        if not inducted_trains:
            return {
                'system_performance': 0,
                'service_readiness': 0,
                'maintenance_risk': 100,
                'branding_compliance': 0,
                'cost_savings': 0,
                'annual_savings': 0,
                'trains_processed': 0,
                'calculation_basis': 'No trains inducted for service'
            }

        # Calculate authentic metrics based on actual train processing
        avg_score = sum(t.get('priority_score', 0) for t in inducted_trains) / len(inducted_trains)
        total_branding = sum(t.get('branding_hours_left', 0) for t in inducted_trains)
        
        # AUTHENTIC COST SAVINGS CALCULATION
        # Performance multiplier based on actual train scores
        performance_multiplier = avg_score / 100  # 0.0 to 1.0 based on actual performance
        
        # Branding compliance bonus (reduces rebranding costs)
        branding_factor = 1.0
        if total_branding > 0:
            avg_branding_hours = total_branding / len(inducted_trains)
            if avg_branding_hours >= 8:
                branding_factor = 1.3  # 30% bonus for good branding compliance
            elif avg_branding_hours >= 4:
                branding_factor = 1.15  # 15% bonus for acceptable branding
        
        # Calculate tonight's savings
        trains_processed = len(inducted_trains)
        tonight_cost_savings = (
            self.BASE_SAVINGS_PER_TRAIN * 
            trains_processed * 
            performance_multiplier * 
            branding_factor
        )
        
        # Annual projection (conservative estimate)
        annual_operational_days = 365 * self.ANNUAL_OPERATIONAL_FACTOR  # 219 operational days per year
        annual_savings = tonight_cost_savings * annual_operational_days
        
        # Maintenance cost reduction calculation
        maintenance_risk_reduction = max(0, 100 - avg_score)
        maintenance_savings = trains_processed * maintenance_risk_reduction * self.MAINTENANCE_SAVINGS_PER_RISK_POINT
        
        return {
            'system_performance': avg_score,
            'service_readiness': min(100, avg_score * 1.1),
            'maintenance_risk': max(0, 100 - avg_score),
            'branding_compliance': min(100, (total_branding / len(inducted_trains)) * 10) if len(inducted_trains) > 0 else 0,
            'cost_savings': int(tonight_cost_savings + maintenance_savings),
            'annual_savings': int(annual_savings),
            'trains_processed': trains_processed,
            'performance_multiplier': performance_multiplier,
            'branding_factor': branding_factor,
            'calculation_basis': f'Based on {trains_processed} inducted trains with avg score {avg_score:.1f}'
        }

    def get_constraint_violations(self):
        """FIXED: Get constraint violations for display"""
        violations = []
        
        # Handle conflicts from constraint engine
        if hasattr(self.constraint_engine, 'conflicts'):
            current_train_id = None
            current_violations = []
            
            for conflict in self.constraint_engine.conflicts:
                if isinstance(conflict, str):
                    # Parse "T002: Expired fitness certificate (2025-09-04)" format
                    if ':' in conflict:
                        train_id, violation = conflict.split(':', 1)
                        train_id = train_id.strip()
                        violation = violation.strip()
                        
                        if train_id != current_train_id:
                            # New train, save previous and start new
                            if current_train_id and current_violations:
                                violations.append({
                                    'train_id': current_train_id,
                                    'violations': current_violations
                                })
                            current_train_id = train_id
                            current_violations = [violation]
                        else:
                            # Same train, add violation
                            current_violations.append(violation)
                    else:
                        # Generic violation
                        violations.append({
                            'train_id': 'Unknown',
                            'violations': [conflict]
                        })
            
            # Don't forget the last train
            if current_train_id and current_violations:
                violations.append({
                    'train_id': current_train_id,
                    'violations': current_violations
                })
        
        # Also check ineligible trains for additional context
        if hasattr(self.constraint_engine, 'ineligible_trains'):
            ineligible = self.constraint_engine.ineligible_trains
            if isinstance(ineligible, list):
                for train in ineligible:
                    train_id = train.get('train_id', 'Unknown') if isinstance(train, dict) else str(train)
                    train_conflicts = train.get('conflicts', []) if isinstance(train, dict) else []
                    if train_conflicts:
                        violations.append({
                            'train_id': train_id,
                            'violations': train_conflicts
                        })
        
        return violations

class KMRLCombinedIntelliFleetSystem:
    def __init__(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        self.running = False
        self._init_components()

    def _init_components(self):
        logger.info("Initializing combined system components...")

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

        # Phase 5: FIXED AI Data Processor for Web Dashboard
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

        # MAIN CHANGE: Use Combined Dashboard instead of just classical
        self.combined_dashboard = CombinedKMRLDashboard(
            self.digital_twin,
            self.monitor,
            self.iot_simulator,
            self.cv_system,
            ai_optimizer=self.optimizer,
            constraint_engine=self.constraint_engine,
            ai_dashboard=self.ai_dashboard,
            ai_data_processor=self.ai_data_processor
        )

        # API and Mobile components (using different ports to avoid conflicts)
        self.api_gateway = APIGateway(self.digital_twin, self.monitor)
        self.mobile_api = MobileAPIServer(self.digital_twin, self.iot_simulator, self.cv_system, port=5001)
        self.iot_websocket = IoTWebSocketServer(self.iot_simulator, port=8766)

        logger.info("All components initialized with Combined Dashboard.")

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

            # FIXED: Print debug info to verify data injection
            summary = ai_state['ai_summary']
            print(f"🤖 AI Data injected successfully:")
            print(f" - Total trains: {summary['total_trains']}")
            print(f" - Inducted trains: {summary['inducted_trains']}")
            print(f" - Train details count: {len(ai_state['ai_train_details'])}")
            print(f" - Performance data: {ai_state['ai_performance']['trains_processed']} processed")

        except Exception as e:
            logger.error(f"Error injecting AI data: {e}")

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
                    'location': 'muttom_depot',  # Start all trains at depot
                    'status': 'inducted',  # FIXED: All trains in inducted_trains are inducted
                    'mileage_km': train_info.get('mileage_km', 15000),
                    'fitness_valid_until': train_info.get('fitness_valid_until', '2026-12-31'),
                    'priority_score': train_info.get('priority_score', 0.0),
                    'assigned_bay': train_info.get('assigned_bay', ''),
                    'branding_hours': train_info.get('branding_hours_left', 0.0)
                }

            # Add standby trains
            standby_trains = results.get('standby_trains', [])
            for train_info in standby_trains:
                train_id = train_info.get('train_id', '')
                trains_dict[train_id] = {
                    'location': 'muttom_depot',
                    'status': 'standby',
                    'mileage_km': train_info.get('mileage_km', 15000),
                    'fitness_valid_until': train_info.get('fitness_valid_until', '2026-12-31'),
                    'priority_score': train_info.get('priority_score', 0.0),
                    'branding_hours': train_info.get('branding_hours_left', 0.0)
                }

            # Add ineligible trains - FIXED error handling
            if hasattr(self.constraint_engine, 'ineligible_trains'):
                ineligible_trains = self.constraint_engine.ineligible_trains
                if isinstance(ineligible_trains, list):
                    for train in ineligible_trains:
                        if isinstance(train, dict):
                            train_id = train.get('train_id', 'Unknown')
                            trains_dict[train_id] = {
                                'location': 'muttom_depot',
                                'status': 'ineligible',
                                'mileage_km': train.get('mileage_km', 20000),
                                'fitness_valid_until': train.get('fitness_valid_until', '2025-01-01')
                            }
                        else:
                            train_id = str(train)
                            trains_dict[train_id] = {
                                'location': 'muttom_depot',
                                'status': 'ineligible',
                                'mileage_km': 20000,
                                'fitness_valid_until': '2025-01-01'
                            }

        # If we don't have any trains, add some default ones
        if not trains_dict:
            trains_dict = {
                'KMRL_001': {'location': 'muttom_depot', 'status': 'idle', 'mileage_km': 15000, 'fitness_valid_until': '2026-12-31'},
                'KMRL_002': {'location': 'ernakulam_south', 'status': 'running', 'mileage_km': 20000, 'fitness_valid_until': '2027-03-31'},
                'KMRL_003': {'location': 'kadavanthra', 'status': 'idle', 'mileage_km': 18000, 'fitness_valid_until': '2026-09-30'},
                'KMRL_004': {'location': 'maharajas_college', 'status': 'running', 'mileage_km': 22000, 'fitness_valid_until': '2027-01-15'},
                'KMRL_005': {'location': 'muttom_depot', 'status': 'idle', 'mileage_km': 20000, 'fitness_valid_until': '2026-11-20'}
            }

        bay_config = {
            'Bay1': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 9, 'power_available': True},
            'Bay2': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 7, 'power_available': True},
            'Bay4': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 8, 'power_available': True}
        }

        return {
            'trains': trains_dict,
            'depots': ['Muttom', 'Ernakulam South'],
            'maintenance_schedule': {},
            'bay_config': bay_config
        }

    def start(self):
        logger.info("Starting KMRL Combined IntelliFleet System...")
        self.running = True

        # Start all components
        self.digital_twin.start_simulation(time_multiplier=2.0)  # 2x speed for demo
        self.iot_simulator.start_simulation()
        self.monitor.start_monitoring()
        self._inject_ai_data_into_digital_twin()

        # Start background services with different ports
        def run_api_gateway():
            try:
                for port in [8002, 8003, 8004, 8005]:
                    try:
                        self.api_gateway.run_server(host="127.0.0.1", port=port)
                        break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Could not start API Gateway: {e}")

        threading.Thread(target=run_api_gateway, daemon=True).start()
        threading.Thread(target=self.mobile_api.start_server, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(self.iot_websocket.broadcast_sensor_data()), daemon=True).start()

        # Print AI Dashboard summary to console FIRST
        self._print_console_summary()

        # MAIN CHANGE: Start the COMBINED dashboard (this will block)
        print("\n🎬 Starting Combined Dashboard with Navigation Tabs...")
        print("🗺️ Switch to 'Live Simulation' tab for animated train tracking")
        print("📊 Switch to 'Analytics Dashboard' tab for AI insights and performance metrics")
        print("="*80 + "\n")

        # Start the combined dashboard server (blocking)
        self.combined_dashboard.run(debug=False)

    def _print_console_summary(self):
        """Print AI Dashboard summary to console - same as main_app.py output"""
        try:
            print("\n" + "="*80)
            print("🤖 AI OPTIMIZATION SUMMARY")
            print("="*80)

            summary = self.ai_data_processor.get_train_status_summary()
            train_details = self.ai_data_processor.get_detailed_train_list()
            performance = self.ai_data_processor.get_performance_metrics()
            violations = self.ai_data_processor.get_constraint_violations()

            print(f"📊 FLEET OVERVIEW:")
            print(f" Total Trains: {summary['total_trains']}")
            print(f" ✅ Inducted: {summary['inducted_trains']}")
            print(f" 🟢 Ready: {summary['ready_trains']}")
            print(f" 🔧 Maintenance: {summary['maintenance_trains']}")
            print(f" ⏸️ Standby: {summary['standby_trains']}")
            print(f" ❌ Ineligible: {summary['ineligible_trains']}")

            print(f"\n📋 INDUCTED TRAINS:")
            inducted_trains = [t for t in train_details if t['inducted']]
            for train in inducted_trains[:6]:  # Show top 6
                status_icon = "✅" if "Ready" in train['status'] else "⚠️"
                print(f" {train['rank']}. {train['train_id']} - {train['bay_assignment']} - Score: {train['priority_score']:.1f} {status_icon}")

            if performance and performance['trains_processed'] > 0:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f" System Performance: {performance.get('system_performance', 0):.1f}/100")
                print(f" Service Readiness: {performance.get('service_readiness', 0):.1f}/100")
                print(f" Maintenance Risk: {performance.get('maintenance_risk', 0):.1f}/100")
                print(f" Cost Savings: ₹{performance.get('cost_savings', 0):,}")
                print(f" Annual Savings: ₹{performance.get('annual_savings', 0):,}")

            print(f"\n⚠️ CONSTRAINT VIOLATIONS:")
            for violation in violations[:6]:  # Show first 6
                print(f" ❌ {violation['train_id']}: {', '.join(violation['violations'][:2])}")  # Show first 2 violations per train

            print("\n🌐 Combined Dashboard available at: http://127.0.0.1:8050")
            print("🎬 Features: Animated Simulation + Classical Analytics in one interface")
            print("="*80)

        except Exception as e:
            logger.error(f"Error displaying AI summary: {e}")

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
        logger.info("Shutting down KMRL Combined IntelliFleet System...")
        self.running = False
        if hasattr(self, 'digital_twin'):
            self.digital_twin.stop_simulation()
        if hasattr(self, 'iot_simulator'):
            self.iot_simulator.stop_simulation()
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        logger.info("Shutdown complete.")
        os._exit(0)

if __name__ == "__main__":
    print("🚀 Launching KMRL IntelliFleet Combined Enterprise System")
    print("🎯 Features: Animated Simulation + Classical Analytics Dashboard")
    print("📱 Navigation: Use tabs to switch between Live Simulation and Analytics")
    print("🗺️ Live Simulation: Real-time animated train tracking with GPS coordinates")
    print("📊 Analytics Dashboard: AI insights, performance metrics, and system monitoring")
    print("\n" + "="*60)

    system = KMRLCombinedIntelliFleetSystem()
    # Start the complete system with combined dashboard
    system.start()