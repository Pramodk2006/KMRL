"""
enterprise_main.py
Wires up all components with the patched DigitalTwinEngine.
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

# Initial data as pure Python dicts
train_data_dict = {
    'KMRL_001': {
        'location': 'depot', 'status': 'idle', 'mileage_km': 15000,
        'fitness_valid_until': '2026-12-31'
    },
    'KMRL_002': {
        'location': 'route', 'status': 'running', 'mileage_km': 20000,
        'fitness_valid_until': '2027-03-31'
    }
}

bay_config = {
    'bay_1': {'bay_type':'service','max_capacity':1,'geometry_score':5,'power_available':True},
    'bay_2': {'bay_type':'maintenance','max_capacity':1,'geometry_score':4,'power_available':True},
    'bay_3': {'bay_type':'inspection','max_capacity':1,'geometry_score':5,'power_available':True},
}

initial_data = {
    'trains': train_data_dict,
    'depots': ['Muttom','Ernakulam South'],
    'maintenance_schedule': {},
    'bay_config': bay_config
}

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('kmrl.log')])
logger = logging.getLogger()

class KMRLIntelliFleetSystem:
    def __init__(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        self.running = False
        self._init_components()

    def _init_components(self):
        logger.info("Initializing components...")
        self.digital_twin = DigitalTwinEngine(initial_data)
        self.iot_sim = IoTSensorSimulator(list(initial_data['trains'].keys()))
        self.iot_proc = IoTDataProcessor()
        self.cv = ComputerVisionSystem()
        self.monitor = SystemMonitor(self.digital_twin, self.iot_proc, config={})
        self.api = APIGateway(self.digital_twin, self.monitor)
        self.dashboard = InteractiveWebDashboard(self.digital_twin, self.monitor, self.iot_sim, self.cv)
        self.mobile = MobileAPIServer(self.digital_twin, self.iot_sim, self.cv)
        self.ws = IoTWebSocketServer(self.iot_sim)

    def start(self):
        logger.info("Starting system...")
        self.running = True
        self.digital_twin.start_simulation()
        self.iot_sim.start_simulation()
        self.monitor.start_monitoring()
        threading.Thread(target=self.api.run_server, daemon=True).start()
        threading.Thread(target=self.mobile.start_server, daemon=True).start()
        threading.Thread(target=lambda: self.dashboard.run_server(host='127.0.0.1', debug=False), daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(self.ws.broadcast_sensor_data()), daemon=True).start()
        while self.running:
            time.sleep(30)

    def _shutdown(self, *args):
        logger.info("Shutting down...")
        self.running = False
        self.digital_twin.stop_simulation()
        self.iot_sim.stop_simulation()
        self.monitor.stop_monitoring()
        os._exit(0)

if __name__ == "__main__":
    print("🚀 KMRL IntelliFleet Enterprise System")
    system = KMRLIntelliFleetSystem()
    system.start()
