"""
Modern KMRL IntelliFleet Enterprise System
Features enhanced live simulation layout with professional design and improved UX
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

# Import the modern combined dashboard
from modern_combined_dashboard import ModernCombinedKMRLDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("ModernKMRLIntelliFleet")

class AIDataProcessor:
    """Enhanced AI Data Processor for Modern Dashboard Integration"""

    # Cost Calculation Constants
    BASE_SAVINGS_PER_TRAIN = 23000 # Base operational savings per optimally inducted train
    ANNUAL_OPERATIONAL_FACTOR = 0.6 # 60% of year achieves similar optimization
    MAINTENANCE_SAVINGS_PER_RISK_POINT = 500 # Savings per maintenance risk point reduced

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

        # All trains in inducted_trains list are considered inducted
        inducted_trains = results.get('inducted_trains', [])
        summary['inducted_trains'] = len(inducted_trains)

        # Count by status for inducted trains
        for train in inducted_trains:
            priority_score = train.get('priority_score', 0)
            if priority_score >= 80:
                summary['ready_trains'] += 1
            elif priority_score >= 60:
                summary['ready_trains'] += 1 # Ready (Caution) still counts as ready
            else:
                summary['maintenance_trains'] += 1

        # Add standby trains
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
        """Get detailed list of all trains with their status"""
        train_details = []

        if not hasattr(self.optimizer, 'optimized_result'):
            return train_details

        results = self.optimizer.optimized_result

        # Process inducted trains (these are ALL inducted)
        inducted_trains = results.get('inducted_trains', [])
        for i, train in enumerate(inducted_trains, 1):
            train_details.append({
                'rank': i,
                'train_id': train.get('train_id', 'N/A'),
                'status': self._get_train_status(train),
                'bay_assignment': train.get('assigned_bay', 'N/A'),
                'priority_score': train.get('priority_score', 0.0),
                'branding_hours': train.get('branding_hours_left', 0.0),
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': True
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
        """Determine train status based on optimization results"""
        score = train.get('priority_score', 0)
        if score >= 80:
            return 'Ready'
        elif score >= 60:
            return 'Ready (Caution)'
        else:
            return 'Maintenance Required'

    def get_performance_metrics(self):
        """Get performance metrics with authentic cost calculations"""
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

        # Performance multiplier based on actual train scores
        performance_multiplier = avg_score / 100

        # Branding compliance bonus
        branding_factor = 1.0
        if total_branding > 0:
            avg_branding_hours = total_branding / len(inducted_trains)
            if avg_branding_hours >= 8:
                branding_factor = 1.3 # 30% bonus for good branding compliance
            elif avg_branding_hours >= 4:
                branding_factor = 1.15 # 15% bonus for acceptable branding

        # Calculate tonight's savings
        trains_processed = len(inducted_trains)
        tonight_cost_savings = (
            self.BASE_SAVINGS_PER_TRAIN *
            trains_processed *
            performance_multiplier *
            branding_factor
        )

        # Annual projection
        annual_operational_days = 365 * self.ANNUAL_OPERATIONAL_FACTOR
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
        """Get constraint violations for display"""
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

        return violations

class ModernKMRLIntelliFleetSystem:
    """Modern KMRL IntelliFleet System with Enhanced Live Simulation"""

    def __init__(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        self.running = False
        self._init_components()

    def _init_components(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing modern system components...")

        # Phase 1: Initialize AI Optimization Components
        logger.info("⚡ Loading AI optimization components...")
        self.data_loader = DataLoader()
        self.ai_data = self.data_loader.get_integrated_data()

        # Phase 2: Constraint Processing
        logger.info("🔍 Running constraint processing...")
        self.constraint_engine = CustomConstraintEngine(self.ai_data)
        self.constraint_result = self.constraint_engine.run_constraint_optimization()

        # Phase 3: Multi-Objective Optimization
        logger.info("🎯 Executing multi-objective optimization...")
        self.optimizer = MultiObjectiveOptimizer(self.constraint_result, self.ai_data)
        self.optimized_result = self.optimizer.optimize_induction_ranking()
        self.optimizer.optimized_result = self.optimized_result

        # Phase 4: AI Dashboard
        logger.info("🤖 Setting up AI dashboard...")
        self.ai_dashboard = InductionDashboard(self.data_loader, self.constraint_engine, self.optimizer)

        # Phase 5: AI Data Processor
        logger.info("📊 Configuring AI data processor...")
        self.ai_data_processor = AIDataProcessor(self.optimizer, self.constraint_engine, self.data_loader)

        # Phase 6: Digital Twin Engine
        logger.info("🔄 Initializing digital twin engine...")
        initial_data = self._create_digital_twin_data()
        self.digital_twin = DigitalTwinEngine(initial_data)
        self._inject_ai_data_into_digital_twin()

        # Phase 7: IoT and Monitoring
        logger.info("📡 Setting up IoT sensors and monitoring...")
        self.iot_simulator = IoTSensorSimulator(list(initial_data['trains'].keys()))
        self.iot_processor = IoTDataProcessor()
        self.cv_system = ComputerVisionSystem()
        self.monitor = SystemMonitor(self.digital_twin, self.iot_processor)

        # Phase 8: Modern Combined Dashboard
        logger.info("🎨 Loading modern combined dashboard...")
        self.modern_dashboard = ModernCombinedKMRLDashboard(
            self.digital_twin,
            self.monitor,
            self.iot_simulator,
            self.cv_system,
            ai_optimizer=self.optimizer,
            constraint_engine=self.constraint_engine,
            ai_dashboard=self.ai_dashboard,
            ai_data_processor=self.ai_data_processor
        )

        # Phase 9: API Services
        logger.info("🌐 Configuring API services...")
        self.api_gateway = APIGateway(self.digital_twin, self.monitor)
        self.mobile_api = MobileAPIServer(self.digital_twin, self.iot_simulator, self.cv_system, port=5001)
        self.iot_websocket = IoTWebSocketServer(self.iot_simulator, port=8766)

        logger.info("✅ All components initialized successfully!")

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

            # Log injection success
            summary = ai_state['ai_summary']
            logger.info(f"🤖 AI data injected successfully:")
            logger.info(f"   📊 Total trains: {summary['total_trains']}")
            logger.info(f"   ✅ Inducted trains: {summary['inducted_trains']}")
            logger.info(f"   📋 Train details: {len(ai_state['ai_train_details'])}")
            logger.info(f"   💼 Performance data: {ai_state['ai_performance']['trains_processed']} processed")

        except Exception as e:
            logger.error(f"❌ Error injecting AI data: {e}")

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
                    'location': 'muttom_depot',
                    'status': 'inducted',
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

            # Add ineligible trains
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

        # Fallback data if no trains available
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
        """Start the modern KMRL IntelliFleet system"""
        logger.info("🚀 Starting Modern KMRL IntelliFleet System...")
        self.running = True

        # Start core components
        logger.info("▶️ Starting digital twin simulation...")
        self.digital_twin.start_simulation(time_multiplier=2.0)
        
        logger.info("📡 Starting IoT simulation...")
        self.iot_simulator.start_simulation()
        
        logger.info("🔍 Starting system monitoring...")
        self.monitor.start_monitoring()
        
        # Refresh AI data
        self._inject_ai_data_into_digital_twin()

        # Start background services
        logger.info("🌐 Starting background API services...")
        
        def run_api_gateway():
            try:
                for port in [8002, 8003, 8004, 8005]:
                    try:
                        self.api_gateway.run_server(host="127.0.0.1", port=port)
                        break
                    except:
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Could not start API Gateway: {e}")

        threading.Thread(target=run_api_gateway, daemon=True).start()
        threading.Thread(target=self.mobile_api.start_server, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(self.iot_websocket.broadcast_sensor_data()), daemon=True).start()

        # Print console summary
        self._print_modern_console_summary()

        # Start the modern dashboard
        logger.info("🎨 Launching Modern Dashboard...")
        self.modern_dashboard.run(debug=False)

    def _print_modern_console_summary(self):
        """Print modern console summary with enhanced formatting"""
        try:
            print("\n" + "═" * 80)
            print("🤖 AI OPTIMIZATION SUMMARY - MODERN KMRL INTELLIFLEET")
            print("═" * 80)
            
            summary = self.ai_data_processor.get_train_status_summary()
            train_details = self.ai_data_processor.get_detailed_train_list()
            performance = self.ai_data_processor.get_performance_metrics()
            violations = self.ai_data_processor.get_constraint_violations()

            # Fleet Overview
            print(f"📊 FLEET OVERVIEW:")
            print(f"   🚂 Total Trains: {summary['total_trains']}")
            print(f"   ✅ Inducted: {summary['inducted_trains']}")
            print(f"   🟢 Ready: {summary['ready_trains']}")
            print(f"   🔧 Maintenance: {summary['maintenance_trains']}")
            print(f"   ⏸️ Standby: {summary['standby_trains']}")
            print(f"   ❌ Ineligible: {summary['ineligible_trains']}")

            # Inducted Trains
            print(f"\n📋 INDUCTED TRAINS:")
            inducted_trains = [t for t in train_details if t['inducted']]
            for train in inducted_trains[:6]:  # Show top 6
                status_icon = "✅" if "Ready" in train['status'] else "⚠️"
                print(f"   {train['rank']}. {train['train_id']} - {train['bay_assignment']} - Score: {train['priority_score']:.1f} {status_icon}")

            # Performance Metrics
            if performance and performance['trains_processed'] > 0:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"   🎯 System Performance: {performance.get('system_performance', 0):.1f}/100")
                print(f"   🔄 Service Readiness: {performance.get('service_readiness', 0):.1f}/100")
                print(f"   ⚠️ Maintenance Risk: {performance.get('maintenance_risk', 0):.1f}/100")
                print(f"   💰 Cost Savings: ₹{performance.get('cost_savings', 0):,}")
                print(f"   📊 Annual Savings: ₹{performance.get('annual_savings', 0):,}")

            # Constraint Violations
            print(f"\n⚠️ CONSTRAINT VIOLATIONS:")
            for violation in violations[:5]:  # Show first 5
                violations_text = ', '.join(violation['violations'][:2])  # Show first 2 violations per train
                print(f"   ❌ {violation['train_id']}: {violations_text}")

            print(f"\n🌐 MODERN DASHBOARD ACCESS:")
            print(f"   🎨 Main URL: http://127.0.0.1:8050")
            print(f"   📱 Mobile: Fully responsive design")
            print(f"   🔧 API: http://127.0.0.1:8002")

            print(f"\n✨ ENHANCED FEATURES:")
            print(f"   🎬 Enhanced Live Simulation - 3-column modern layout")
            print(f"   📊 Analytics Dashboard - AI insights and performance metrics")
            print(f"   🎨 Material Design 3.0 - Professional styling")
            print(f"   📱 Fully Responsive - Works on all devices")
            print(f"   ⚡ Real-time Updates - 1.5s refresh rate")

            print("═" * 80)

        except Exception as e:
            logger.error(f"❌ Error displaying modern summary: {e}")

    def _shutdown(self, *args):
        """Gracefully shutdown the system"""
        logger.info("🛑 Shutting down Modern KMRL IntelliFleet System...")
        self.running = False

        if hasattr(self, 'digital_twin'):
            self.digital_twin.stop_simulation()

        if hasattr(self, 'iot_simulator'):
            self.iot_simulator.stop_simulation()

        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()

        logger.info("✅ Shutdown complete.")
        os._exit(0)

if __name__ == "__main__":
    print("\n" + "🚀" * 20 + " MODERN KMRL INTELLIFLEET " + "🚀" * 20)
    print("🎨 Next-Generation Railway Management with Enhanced Live Simulation")
    print("📱 Features: Modern Design • Responsive Layout • Enhanced UX")
    print("🎬 Enhanced Live Simulation: 3-column layout with real-time tracking")
    print("📊 Analytics Dashboard: AI insights and performance monitoring")
    print("⚡ Real-time Updates: Professional animations and interactions")
    print("=" * 80)

    system = ModernKMRLIntelliFleetSystem()
    system.start()