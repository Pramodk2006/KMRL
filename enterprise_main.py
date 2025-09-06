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

from src.digital_twin_engine import DigitalTwinEngine
from src.api_gateway import APIGateway
from src.web_dashboard import InteractiveWebDashboard
from src.monitoring_system import SystemMonitor
from src.iot_sensor_system import IoTSensorSimulator, IoTDataProcessor, IoTWebSocketServer
from src.computer_vision_system import ComputerVisionSystem
from src.mobile_integration import MobileAPIServer

# Sample initial data as pure dicts
train_data_dict = {
    'KMRL_001': {
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 15000,
        'fitness_valid_until': '2026-12-31'
    },
    'KMRL_002': {
        'location': 'route',
        'status': 'running',
        'mileage_km': 20000,
        'fitness_valid_until': '2027-03-31'
    },
    'KMRL_003': {
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 18000,
        'fitness_valid_until': '2026-09-30'
    },
    'KMRL_004': {
        'location': 'route',
        'status': 'running',
        'mileage_km': 22000,
        'fitness_valid_until': '2027-01-15'
    },
    'KMRL_005': {
        'location': 'depot',
        'status': 'idle',
        'mileage_km': 20000,
        'fitness_valid_until': '2026-11-20'
    }
}

bay_config = {
    'bay_1': {'bay_type': 'service', 'max_capacity': 1, 'geometry_score': 5, 'power_available': True},
    'bay_2': {'bay_type': 'maintenance', 'max_capacity': 1, 'geometry_score': 4, 'power_available': True},
    'bay_3': {'bay_type': 'inspection', 'max_capacity': 1, 'geometry_score': 5, 'power_available': True},
    'bay_4': {'bay_type': 'cleaning', 'max_capacity': 1, 'geometry_score': 3, 'power_available': True}
}

initial_data = {
    'trains': train_data_dict,
    'depots': ['Muttom', 'Ernakulam South'],
    'maintenance_schedule': {},
    'bay_config': bay_config
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('kmrl_intellifleet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KMRLIntelliFleet")


class KMRLIntelliFleetSystem:
    def __init__(self):
        # Handle shutdown signals
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        self.running = False
        self._init_components()

    def _init_components(self):
        logger.info("Initializing system components...")

        # Digital Twin Engine
        self.digital_twin = DigitalTwinEngine(initial_data)

        # IoT Sensor System
        self.iot_simulator = IoTSensorSimulator(list(initial_data['trains'].keys()))
        self.iot_processor = IoTDataProcessor()

        # Computer Vision System
        self.cv_system = ComputerVisionSystem()

        # System Monitor (requires digital twin and AI optimizer)
        self.monitor = SystemMonitor(self.digital_twin, self.iot_processor)

        # API Gateway
        self.api_gateway = APIGateway(self.digital_twin, self.monitor)

        # Web Dashboard
        self.web_dashboard = InteractiveWebDashboard(self.digital_twin)


        # Mobile API Server
        self.mobile_api = MobileAPIServer(
            self.digital_twin,
            self.iot_simulator,
            self.cv_system,
            port=5000
        )

        # IoT WebSocket Server
        self.iot_websocket = IoTWebSocketServer(self.iot_simulator, port=8765)

        logger.info("All components initialized.")

    def start(self):
        logger.info("Starting KMRL IntelliFleet System...")
        self.running = True

        # Start Digital Twin simulation
        self.digital_twin.start_simulation()

        # Start IoT Sensor simulation
        self.iot_simulator.start_simulation()

        # Start System Monitor
        self.monitor.start_monitoring()

        # Start API Gateway
        threading.Thread(target=self.api_gateway.run_server, daemon=True).start()

        # Start Mobile API Server
        threading.Thread(target=self.mobile_api.start_server, daemon=True).start()

        # Start Web Dashboard
        threading.Thread(
            target=lambda: self.web_dashboard.run_server(debug=False),
            daemon=True
        ).start()

        # Start IoT WebSocket broadcasting
        threading.Thread(
            target=lambda: asyncio.run(self.iot_websocket.broadcast_sensor_data()),
            daemon=True
        ).start()

        # Main loop to keep alive
        try:
            while self.running:
                time.sleep(30)
                dashboard = self.monitor.get_monitoring_dashboard_data()
                system_health = dashboard.get('system_health', {})
                # Check if digital twin is running
                if not system_health.get('digital_twin_running', False):
                    logger.warning("System health degraded: Digital twin simulation is not running!")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self._shutdown()

    def _shutdown(self, *args):
        logger.info("Shutting down KMRL IntelliFleet System...")
        self.running = False
        self.digital_twin.stop_simulation()
        self.iot_simulator.stop_simulation()
        self.monitor.stop_monitoring()
        logger.info("Shutdown complete.")
        os._exit(0)


if __name__ == "__main__":
    print("🚀 Launching KMRL IntelliFleet Enterprise System")
    system = KMRLIntelliFleetSystem()
    system.start()
