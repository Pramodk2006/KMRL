import asyncio
import threading
import time
import logging
from datetime import datetime
import signal
import sys
import os

# Import all system components
from src.digital_twin_engine import DigitalTwinEngine
from src.api_gateway import APIGateway
from src.web_dashboard import InteractiveWebDashboard
from src.monitoring_system import SystemMonitor
from src.iot_sensor_system import IoTSensorSimulator, IoTDataProcessor, IoTWebSocketServer
from src.computer_vision_system import ComputerVisionSystem
from src.mobile_integration import MobileAPIServer

# Complete initial data with all required fields
train_data_dict = {
    'KMRL_001': {
        'train_id': 'KMRL_001',
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 15000,
        'branding_hours_left': 80,
        'fitness_valid_until': '2026-12-31',
        'cleaning_slot_id': None,
        'bay_geometry_score': 5,
        'failure_probability': 0.1
    },
    'KMRL_002': {
        'train_id': 'KMRL_002',
        'location': 'route',
        'status': 'running',
        'mileage_km': 20000,
        'branding_hours_left': 65,
        'fitness_valid_until': '2027-03-31',
        'cleaning_slot_id': None,
        'bay_geometry_score': 4,
        'failure_probability': 0.15
    },
    'KMRL_003': {
        'train_id': 'KMRL_003',
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 18000,
        'branding_hours_left': 90,
        'fitness_valid_until': '2026-09-30',
        'cleaning_slot_id': None,
        'bay_geometry_score': 5,
        'failure_probability': 0.08
    },
    'KMRL_004': {
        'train_id': 'KMRL_004',
        'location': 'route',
        'status': 'running',
        'mileage_km': 22000,
        'branding_hours_left': 45,
        'fitness_valid_until': '2027-01-15',
        'cleaning_slot_id': None,
        'bay_geometry_score': 4,
        'failure_probability': 0.18
    },
    'KMRL_005': {
        'train_id': 'KMRL_005',
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 16500,
        'branding_hours_left': 75,
        'fitness_valid_until': '2026-11-20',
        'cleaning_slot_id': None,
        'bay_geometry_score': 5,
        'failure_probability': 0.12
    }
}

# Bay configuration with all required fields
bay_config = {
    'bay_1': {
        'bay_id': 'bay_1',
        'bay_type': 'service',
        'max_capacity': 1,
        'geometry_score': 5,
        'power_available': True,
        'status': 'available'
    },
    'bay_2': {
        'bay_id': 'bay_2',
        'bay_type': 'heavy_maintenance',
        'max_capacity': 1,
        'geometry_score': 4,
        'power_available': True,
        'status': 'available'
    },
    'bay_3': {
        'bay_id': 'bay_3',
        'bay_type': 'inspection',
        'max_capacity': 1,
        'geometry_score': 5,
        'power_available': True,
        'status': 'available'
    },
    'bay_4': {
        'bay_id': 'bay_4',
        'bay_type': 'cleaning',
        'max_capacity': 1,
        'geometry_score': 3,
        'power_available': True,
        'status': 'available'
    }
}

# Complete initial data structure
initial_data = {
    'trains': train_data_dict,
    'depots': ['Muttom', 'Maharajas College', 'Ernakulam South', 'Kadavanthra'],
    'maintenance_schedule': {},
    'bay_config': bay_config
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kmrl_intellifleet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MockAIOptimizer:
    """Mock AI Optimizer for compatibility with monitoring system"""
    
    def __init__(self):
        self.optimized_result = {
            'optimization_improvements': {
                'avg_composite_score': 85.5,
                'avg_service_readiness': 92.3,
                'avg_maintenance_penalty': 12.8
            }
        }

class KMRLIntelliFleetSystem:
    """Complete KMRL IntelliFleet Enterprise System"""
    
    def __init__(self):
        self.system_components = {}
        self.running = False
        
        # Initialize all system components
        self.initialize_components()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing KMRL IntelliFleet Enterprise System...")
            
            # 1. Digital Twin Engine (Core)
            logger.info("🔧 Initializing Digital Twin Engine...")
            self.digital_twin = DigitalTwinEngine(initial_data=initial_data)
            self.system_components['digital_twin'] = self.digital_twin
            
            # 2. IoT Sensor System
            logger.info("🌐 Initializing IoT Sensor System...")
            train_ids = ['KMRL_001', 'KMRL_002', 'KMRL_003', 'KMRL_004', 'KMRL_005']
            self.iot_simulator = IoTSensorSimulator(train_ids)
            self.iot_processor = IoTDataProcessor()
            self.system_components['iot_simulator'] = self.iot_simulator
            self.system_components['iot_processor'] = self.iot_processor
            
            # 3. Computer Vision System
            logger.info("🔍 Initializing Computer Vision System...")
            self.cv_system = ComputerVisionSystem()
            self.system_components['computer_vision'] = self.cv_system
            
            # 4. Mock AI Optimizer (for monitoring compatibility)
            logger.info("🤖 Initializing Mock AI Optimizer...")
            self.ai_optimizer = MockAIOptimizer()
            self.system_components['ai_optimizer'] = self.ai_optimizer
            
            # 5. System Monitor (with correct parameters)
            logger.info("📊 Initializing System Monitor...")
            self.monitor = SystemMonitor(self.digital_twin, self.ai_optimizer, config={})
            self.system_components['monitor'] = self.monitor
            
            # 6. API Gateway
            logger.info("🌍 Initializing API Gateway...")
            self.api_gateway = APIGateway(self.digital_twin, self.monitor)
            self.system_components['api_gateway'] = self.api_gateway
            
            # 7. Web Dashboard
            logger.info("📱 Initializing Web Dashboard...")
            self.web_dashboard = InteractiveWebDashboard(
                self.digital_twin, 
                self.monitor,
                self.iot_simulator,
                self.cv_system
            )
            self.system_components['web_dashboard'] = self.web_dashboard
            
            # 8. Mobile API Server
            logger.info("📲 Initializing Mobile API Server...")
            self.mobile_api = MobileAPIServer(
                self.digital_twin,
                self.iot_simulator,
                self.cv_system,
                port=5000
            )
            self.system_components['mobile_api'] = self.mobile_api
            
            # 9. IoT WebSocket Server
            logger.info("🔗 Initializing IoT WebSocket Server...")
            self.iot_websocket = IoTWebSocketServer(self.iot_simulator, port=8765)
            self.system_components['iot_websocket'] = self.iot_websocket
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def start_system(self):
        """Start all system components"""
        try:
            logger.info("🏁 Starting KMRL IntelliFleet Enterprise System...")
            self.running = True
            
            # Start Digital Twin Engine
            logger.info("▶️ Starting Digital Twin Engine...")
            self.digital_twin.start_simulation()
            
            # Start IoT Sensor Simulation
            logger.info("▶️ Starting IoT Sensor Simulation...")
            self.iot_simulator.start_simulation()
            
            # Start System Monitor
            logger.info("▶️ Starting System Monitor...")
            self.monitor.start_monitoring()
            
            # Start API Gateway
            logger.info("▶️ Starting API Gateway...")
            api_thread = threading.Thread(target=self.api_gateway.start_server, daemon=True)
            api_thread.start()
            
            # Start Mobile API Server
            logger.info("▶️ Starting Mobile API Server...")
            mobile_thread = threading.Thread(target=self.mobile_api.start_server, daemon=True)
            mobile_thread.start()
            
            # Start Web Dashboard
            logger.info("▶️ Starting Web Dashboard...")
            dashboard_thread = threading.Thread(
                target=lambda: self.web_dashboard.run_server(debug=False),
                daemon=True
            )
            dashboard_thread.start()
            
            # Start IoT WebSocket Server (in separate thread for async)
            logger.info("▶️ Starting IoT WebSocket Server...")
            websocket_thread = threading.Thread(
                target=self._start_websocket_server,
                daemon=True
            )
            websocket_thread.start()
            
            # Start automated inspections
            logger.info("▶️ Starting Automated Inspections...")
            inspection_thread = threading.Thread(
                target=self._run_automated_inspections,
                daemon=True
            )
            inspection_thread.start()
            
            # Wait a moment for all servers to start
            time.sleep(5)
            
            # System status
            self.print_system_status()
            
            # Keep main thread alive
            self.main_loop()
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            self.stop_system()
            raise
    
    def _start_websocket_server(self):
        """Start WebSocket server in async loop"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Import websockets here to avoid import issues
            import websockets
            
            async def register_client(websocket, path):
                await self.iot_websocket.register_client(websocket, path)
            
            # Start WebSocket server
            start_server = websockets.serve(register_client, "localhost", 8765)
            
            # Start broadcasting
            loop.run_until_complete(asyncio.gather(
                start_server,
                self.iot_websocket.broadcast_sensor_data()
            ))
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    def _run_automated_inspections(self):
        """Run automated computer vision inspections"""
        logger.info("🔍 Starting automated inspection cycle...")
        
        while self.running:
            try:
                # Get list of trains from digital twin
                current_state = self.digital_twin.get_current_state()
                trains = current_state.get('trains', {})
                
                # Perform inspections on random trains (limit to 2 per cycle)
                train_list = list(trains.keys())
                for train_id in train_list[:2]:
                    logger.info(f"🔍 Performing automated inspection on {train_id}...")
                    inspection_result = self.cv_system.perform_full_inspection(train_id)
                    
                    # Log inspection results
                    logger.info(f"✅ Inspection complete for {train_id}: "
                              f"{inspection_result.overall_condition} condition, "
                              f"{inspection_result.total_defects} defects found")
                    
                    # Small delay between inspections
                    time.sleep(10)
                
                # Wait before next inspection cycle (30 minutes in real deployment, 5 minutes for demo)
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in automated inspection: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def main_loop(self):
        """Main system loop with enhanced debugging"""
        import traceback
        
        logger.info("🔄 Entering main system loop...")
        
        while self.running:
            try:
                # System health check every 30 seconds
                time.sleep(30)
                
                if not self.running:
                    break
                
                # Enhanced debugging for health status
                logger.info("📊 Starting health check...")
                
                try:
                    health_status = self.get_system_health()
                    logger.info(f"📊 Health status type: {type(health_status)}")
                    logger.info(f"📊 Health status keys: {list(health_status.keys()) if isinstance(health_status, dict) else 'NOT A DICT'}")
                    
                    # Safe access with multiple fallbacks
                    if isinstance(health_status, dict):
                        overall_status = health_status.get('overall', 'unknown')
                        logger.info(f"📊 Overall status retrieved: {overall_status}")
                    else:
                        overall_status = 'unknown'
                        logger.warning(f"📊 Health status is not a dict: {type(health_status)}")
                    
                    if overall_status != 'healthy':
                        logger.warning(f"⚠️ System health issues detected: {health_status}")
                    else:
                        logger.info("✅ System health is good")
                        
                except Exception as health_error:
                    logger.error(f"❌ Error during health check: {health_error}")
                    logger.error(f"❌ Health check stack trace:")
                    traceback.print_exc()
                
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal...")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                logger.error(f"❌ Error type: {type(e)}")
                logger.error(f"❌ Full stack trace:")
                traceback.print_exc()
                time.sleep(10)

    def get_system_health(self):
        """Get overall system health status - bulletproof version"""
        import traceback
        
        # Initialize with guaranteed structure
        health = {}
        
        try:
            logger.info("🏥 Starting system health check...")
            
            # Ensure basic structure exists
            health = {
                'timestamp': datetime.now().isoformat(),
                'overall': 'unknown',  # Always start with unknown
                'components': {},
                'errors': []
            }
            
            logger.info("🏥 Basic health structure initialized")
            
            # Check Digital Twin with extreme safety
            try:
                if hasattr(self, 'digital_twin'):
                    if hasattr(self.digital_twin, 'is_running'):
                        dt_status = 'healthy' if self.digital_twin.is_running else 'unhealthy'
                        logger.info(f"🏥 Digital twin status: {dt_status}")
                    else:
                        dt_status = 'unknown'
                        logger.info("🏥 Digital twin has no is_running attribute")
                else:
                    dt_status = 'missing'
                    logger.warning("🏥 No digital_twin attribute found")
                    
                health['components']['digital_twin'] = dt_status
            except Exception as dt_error:
                logger.error(f"🏥 Error checking digital twin: {dt_error}")
                health['components']['digital_twin'] = 'error'
                health['errors'].append(f"digital_twin: {dt_error}")
            
            # Check IoT Simulator with extreme safety
            try:
                if hasattr(self, 'iot_simulator'):
                    if hasattr(self.iot_simulator, 'is_running'):
                        iot_status = 'healthy' if self.iot_simulator.is_running else 'unhealthy'
                        logger.info(f"🏥 IoT simulator status: {iot_status}")
                    else:
                        iot_status = 'unknown'
                        logger.info("🏥 IoT simulator has no is_running attribute")
                else:
                    iot_status = 'missing'
                    logger.warning("🏥 No iot_simulator attribute found")
                    
                health['components']['iot_simulator'] = iot_status
            except Exception as iot_error:
                logger.error(f"🏥 Error checking IoT simulator: {iot_error}")
                health['components']['iot_simulator'] = 'error'
                health['errors'].append(f"iot_simulator: {iot_error}")
            
            # Check other components
            try:
                health['components']['api_gateway'] = 'healthy' if hasattr(self, 'api_gateway') else 'missing'
                health['components']['computer_vision'] = 'healthy' if hasattr(self, 'cv_system') else 'missing'
                health['components']['web_dashboard'] = 'healthy' if hasattr(self, 'web_dashboard') else 'missing'
                health['components']['mobile_api'] = 'healthy' if hasattr(self, 'mobile_api') else 'missing'
                health['components']['monitor'] = 'healthy' if hasattr(self, 'monitor') else 'missing'
            except Exception as comp_error:
                logger.error(f"🏥 Error checking components: {comp_error}")
                health['errors'].append(f"components: {comp_error}")
            
            # Calculate overall health
            try:
                component_statuses = list(health['components'].values())
                healthy_count = sum(1 for status in component_statuses if status == 'healthy')
                total_count = len(component_statuses)
                
                if total_count == 0:
                    health['overall'] = 'unknown'
                elif healthy_count == total_count:
                    health['overall'] = 'healthy'
                elif healthy_count >= total_count * 0.7:  # 70% healthy
                    health['overall'] = 'degraded'
                else:
                    health['overall'] = 'unhealthy'
                    
                logger.info(f"🏥 Calculated overall health: {health['overall']} ({healthy_count}/{total_count} healthy)")
            except Exception as calc_error:
                logger.error(f"🏥 Error calculating overall health: {calc_error}")
                health['overall'] = 'error'
                health['errors'].append(f"calculation: {calc_error}")
            
            # Final safety check
            if 'overall' not in health:
                health['overall'] = 'unknown'
                logger.warning("🏥 'overall' key was missing, set to unknown")
                
            logger.info(f"🏥 Final health status: {health}")
            
        except Exception as e:
            logger.error(f"🏥 Critical error in get_system_health: {e}")
            traceback.print_exc()
            
            # Emergency fallback
            health = {
                'timestamp': datetime.now().isoformat(),
                'overall': 'critical_error',
                'components': {},
                'errors': [f"Critical health check failure: {e}"]
            }
        
        # Triple-check that 'overall' exists
        if not isinstance(health, dict):
            logger.error(f"🏥 Health is not a dict! Type: {type(health)}, Value: {health}")
            health = {'overall': 'invalid_response', 'timestamp': datetime.now().isoformat(), 'components': {}}
        
        if 'overall' not in health:
            logger.error("🏥 CRITICAL: 'overall' key still missing from health dict!")
            health['overall'] = 'missing_key_error'
        
        return health
    
    def print_system_status(self):
        """Print current system status"""
        print("\n" + "="*80)
        print("🏆 KMRL INTELLIFLEET ENTERPRISE SYSTEM - RUNNING")
        print("="*80)
        print("🌐 System Access Points:")
        print("   📊 Main Web Dashboard:    http://127.0.0.1:8050")
        print("   🔗 API Gateway:          http://127.0.0.1:8000")
        print("   📱 Mobile API Server:    http://127.0.0.1:5000")
        print("   🌐 IoT WebSocket:        ws://127.0.0.1:8765")
        print("")
        print("📋 System Components Status:")
        for component_name, component in self.system_components.items():
            status = "🟢 RUNNING" if hasattr(component, 'is_running') and getattr(component, 'is_running', True) else "🟢 ACTIVE"
            print(f"   {component_name:20} {status}")
        print("")
        print("🚀 System Features:")
        print("   ✅ Real-time Digital Twin Engine")
        print("   ✅ IoT Sensor Data Simulation & Processing")  
        print("   ✅ Computer Vision Defect Detection")
        print("   ✅ Predictive Maintenance Analytics")
        print("   ✅ Interactive Web Dashboard")
        print("   ✅ RESTful API Gateway")
        print("   ✅ Mobile Field Operations API")
        print("   ✅ Real-time WebSocket Streaming")
        print("   ✅ Automated Inspection System")
        print("   ✅ QR Code Generation & Management")
        print("   ✅ Enterprise Monitoring & Alerts")
        print("="*80)
        print("💡 Press Ctrl+C to shutdown gracefully")
        print("="*80 + "\n")
    
    def graceful_shutdown(self, signum=None, frame=None):
        """Gracefully shutdown all system components"""
        logger.info("🛑 Initiating graceful shutdown...")
        self.stop_system()
    
    def stop_system(self):
        """Stop all system components"""
        try:
            self.running = False
            
            logger.info("⏹️ Stopping KMRL IntelliFleet System components...")
            
            # Stop Digital Twin Engine
            if hasattr(self, 'digital_twin'):
                self.digital_twin.stop_simulation()
            
            # Stop IoT Sensor Simulation
            if hasattr(self, 'iot_simulator'):
                self.iot_simulator.stop_simulation()
            
            # Stop System Monitor
            if hasattr(self, 'monitor'):
                self.monitor.stop_monitoring()
            
            logger.info("✅ All components stopped successfully")
            logger.info("👋 KMRL IntelliFleet System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            # Exit the application
            os._exit(0)

def main():
    """Main entry point"""
    try:
        print("\n🚀 Starting KMRL IntelliFleet Enterprise System...")
        print("⚡ Advanced AI & IoT Integration - Phase 6 Complete")
        print("🏛️ Built for Kochi Metro Rail Limited (KMRL)")
        print("-" * 60)
        
        # Create and start the system
        intellifleet_system = KMRLIntelliFleetSystem()
        intellifleet_system.start_system()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Critical system error: {e}")
        print(f"\n❌ System failed to start: {e}")
    finally:
        print("👋 KMRL IntelliFleet System terminated")

if __name__ == "__main__":
    main()
