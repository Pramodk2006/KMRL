import asyncio
import json
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import threading
import logging
from collections import deque
import websockets
import aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """IoT sensor reading data structure"""
    sensor_id: str
    train_id: str
    sensor_type: str  # temperature, vibration, power, gps, brake, door
    value: float
    unit: str
    timestamp: datetime
    location: str
    quality: float = 1.0  # Data quality score 0-1
    alert_level: str = "normal"  # normal, warning, critical

@dataclass
class TrainTelemetry:
    """Complete train telemetry data"""
    train_id: str
    timestamp: datetime
    location: Dict[str, float]  # lat, lon, altitude
    speed_kmh: float
    temperature_c: float
    vibration_level: float
    power_consumption_kw: float
    brake_pressure_bar: float
    door_status: str  # closed, open, fault
    passenger_count: int
    maintenance_alerts: List[str]

class IoTSensorSimulator:
    """Simulates realistic IoT sensor data for trains"""
    
    def __init__(self, train_ids: List[str]):
        self.train_ids = train_ids
        self.sensor_readings = deque(maxlen=10000)  # Keep last 10k readings
        self.is_running = False
        self.simulation_thread = None
        
        # Sensor configurations
        self.sensor_configs = {
            'temperature': {'min': 15.0, 'max': 85.0, 'unit': '°C', 'normal_range': (20, 45)},
            'vibration': {'min': 0.0, 'max': 50.0, 'unit': 'mm/s', 'normal_range': (0, 15)},
            'power': {'min': 50.0, 'max': 500.0, 'unit': 'kW', 'normal_range': (80, 300)},
            'brake_pressure': {'min': 0.0, 'max': 10.0, 'unit': 'bar', 'normal_range': (6, 9)},
            'passenger_count': {'min': 0, 'max': 300, 'unit': 'persons', 'normal_range': (20, 250)}
        }
        
        # Train locations (simulated depot areas in Kochi)
        self.depot_locations = {
            'muttom_depot': {'lat': 9.9312, 'lon': 76.2673},
            'maharajas_college': {'lat': 9.9380, 'lon': 76.2828},
            'ernakulam_south': {'lat': 9.9816, 'lon': 76.2999},
            'kadavanthra': {'lat': 10.0090, 'lon': 76.3048}
        }
    
    def start_simulation(self):
        """Start IoT sensor data simulation"""
        if self.is_running:
            return
            
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        logger.info("🔗 IoT sensor simulation started")
    
    def stop_simulation(self):
        """Stop IoT sensor data simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
        logger.info("⏹️ IoT sensor simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop for generating sensor data"""
        while self.is_running:
            try:
                # Generate readings for each train
                for train_id in self.train_ids:
                    telemetry = self._generate_train_telemetry(train_id)
                    
                    # Convert to individual sensor readings
                    readings = self._telemetry_to_sensor_readings(telemetry)
                    
                    for reading in readings:
                        self.sensor_readings.append(reading)
                
                # Sleep for realistic data generation interval
                time.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"IoT simulation error: {e}")
                time.sleep(10)
    
    def _generate_train_telemetry(self, train_id: str) -> TrainTelemetry:
        """Generate realistic telemetry for a train"""
        current_time = datetime.now()
        
        # Select random depot location
        depot_name = random.choice(list(self.depot_locations.keys()))
        base_location = self.depot_locations[depot_name]
        
        # Add some random movement within depot area
        location = {
            'lat': base_location['lat'] + random.uniform(-0.01, 0.01),
            'lon': base_location['lon'] + random.uniform(-0.01, 0.01),
            'altitude': random.uniform(0, 50)
        }
        
        # Generate realistic sensor values with occasional anomalies
        def generate_sensor_value(sensor_type: str, has_anomaly: bool = False) -> float:
            config = self.sensor_configs[sensor_type]
            
            if has_anomaly:
                # Generate anomalous reading
                if random.random() < 0.5:
                    return random.uniform(config['min'], config['normal_range'][0])
                else:
                    return random.uniform(config['normal_range'][1], config['max'])
            else:
                # Generate normal reading
                return random.uniform(config['normal_range'][0], config['normal_range'][1])
        
        # 5% chance of anomaly for any sensor
        has_anomaly = random.random() < 0.05
        
        # Generate maintenance alerts based on sensor values
        maintenance_alerts = []
        temp = generate_sensor_value('temperature', has_anomaly)
        vibration = generate_sensor_value('vibration', has_anomaly)
        power = generate_sensor_value('power', has_anomaly)
        
        if temp > 65:
            maintenance_alerts.append(f"High temperature detected: {temp:.1f}°C")
        if vibration > 25:
            maintenance_alerts.append(f"Excessive vibration: {vibration:.1f}mm/s")
        if power > 400:
            maintenance_alerts.append(f"High power consumption: {power:.1f}kW")
        
        return TrainTelemetry(
            train_id=train_id,
            timestamp=current_time,
            location=location,
            speed_kmh=random.uniform(0, 5),  # Depot speeds are low
            temperature_c=temp,
            vibration_level=vibration,
            power_consumption_kw=power,
            brake_pressure_bar=generate_sensor_value('brake_pressure', has_anomaly),
            door_status=random.choice(['closed', 'closed', 'closed', 'open', 'fault']),
            passenger_count=int(generate_sensor_value('passenger_count', has_anomaly)),
            maintenance_alerts=maintenance_alerts
        )
    
    def _telemetry_to_sensor_readings(self, telemetry: TrainTelemetry) -> List[SensorReading]:
        """Convert telemetry to individual sensor readings"""
        readings = []
        
        # Temperature sensor
        readings.append(SensorReading(
            sensor_id=f"{telemetry.train_id}_temp_001",
            train_id=telemetry.train_id,
            sensor_type="temperature",
            value=telemetry.temperature_c,
            unit="°C",
            timestamp=telemetry.timestamp,
            location=f"{telemetry.location['lat']:.6f},{telemetry.location['lon']:.6f}",
            quality=random.uniform(0.9, 1.0),
            alert_level="critical" if telemetry.temperature_c > 70 else "warning" if telemetry.temperature_c > 55 else "normal"
        ))
        
        # Vibration sensor
        readings.append(SensorReading(
            sensor_id=f"{telemetry.train_id}_vib_001",
            train_id=telemetry.train_id,
            sensor_type="vibration",
            value=telemetry.vibration_level,
            unit="mm/s",
            timestamp=telemetry.timestamp,
            location=f"{telemetry.location['lat']:.6f},{telemetry.location['lon']:.6f}",
            quality=random.uniform(0.9, 1.0),
            alert_level="critical" if telemetry.vibration_level > 30 else "warning" if telemetry.vibration_level > 20 else "normal"
        ))
        
        # Power sensor
        readings.append(SensorReading(
            sensor_id=f"{telemetry.train_id}_pwr_001",
            train_id=telemetry.train_id,
            sensor_type="power",
            value=telemetry.power_consumption_kw,
            unit="kW",
            timestamp=telemetry.timestamp,
            location=f"{telemetry.location['lat']:.6f},{telemetry.location['lon']:.6f}",
            quality=random.uniform(0.95, 1.0),
            alert_level="warning" if telemetry.power_consumption_kw > 350 else "normal"
        ))
        
        # GPS sensor
        readings.append(SensorReading(
            sensor_id=f"{telemetry.train_id}_gps_001",
            train_id=telemetry.train_id,
            sensor_type="gps",
            value=telemetry.speed_kmh,
            unit="km/h",
            timestamp=telemetry.timestamp,
            location=f"{telemetry.location['lat']:.6f},{telemetry.location['lon']:.6f}",
            quality=random.uniform(0.85, 1.0),
            alert_level="normal"
        ))
        
        return readings
    
    def get_latest_readings(self, train_id: str = None, sensor_type: str = None, 
                          limit: int = 100) -> List[SensorReading]:
        """Get latest sensor readings with optional filtering"""
        readings = list(self.sensor_readings)
        
        # Filter by train_id if specified
        if train_id:
            readings = [r for r in readings if r.train_id == train_id]
        
        # Filter by sensor_type if specified
        if sensor_type:
            readings = [r for r in readings if r.sensor_type == sensor_type]
        
        # Sort by timestamp (newest first) and limit
        readings.sort(key=lambda x: x.timestamp, reverse=True)
        return readings[:limit]
    
    def get_alerts(self) -> List[SensorReading]:
        """Get all current alerts (warning and critical readings)"""
        readings = list(self.sensor_readings)
        alerts = [r for r in readings if r.alert_level in ['warning', 'critical']]
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:50]  # Last 50 alerts

class IoTDataProcessor:
    """Processes and analyzes IoT sensor data"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.anomaly_thresholds = {
            'temperature': {'warning': 55, 'critical': 70},
            'vibration': {'warning': 20, 'critical': 30},
            'power': {'warning': 350, 'critical': 450}
        }
    
    def process_reading(self, reading: SensorReading) -> Dict[str, Any]:
        """Process individual sensor reading and generate insights"""
        analysis = {
            'reading': asdict(reading),
            'anomaly_detected': False,
            'anomaly_score': 0.0,
            'recommendations': [],
            'trends': {}
        }
        
        # Anomaly detection
        if reading.sensor_type in self.anomaly_thresholds:
            thresholds = self.anomaly_thresholds[reading.sensor_type]
            
            if reading.value >= thresholds['critical']:
                analysis['anomaly_detected'] = True
                analysis['anomaly_score'] = 0.9
                analysis['recommendations'].append(
                    f"CRITICAL: {reading.sensor_type} at {reading.value}{reading.unit} - immediate maintenance required"
                )
            elif reading.value >= thresholds['warning']:
                analysis['anomaly_detected'] = True
                analysis['anomaly_score'] = 0.6
                analysis['recommendations'].append(
                    f"WARNING: {reading.sensor_type} at {reading.value}{reading.unit} - monitor closely"
                )
        
        # Cache in Redis if available
        if self.redis_client:
            cache_key = f"sensor_analysis:{reading.sensor_id}:{int(reading.timestamp.timestamp())}"
            self.redis_client.setex(cache_key, 3600, json.dumps(analysis, default=str))
        
        return analysis
    
    def calculate_train_health_score(self, train_id: str, readings: List[SensorReading]) -> float:
        """Calculate overall health score for a train based on sensor readings"""
        if not readings:
            return 1.0
        
        health_scores = []
        
        for reading in readings:
            if reading.sensor_type == 'temperature':
                # Temperature health score (0-1)
                if reading.value <= 45:
                    score = 1.0
                elif reading.value <= 70:
                    score = 1.0 - (reading.value - 45) / 25  # Linear decrease
                else:
                    score = 0.1  # Critical temperature
                health_scores.append(score)
                
            elif reading.sensor_type == 'vibration':
                # Vibration health score (0-1)
                if reading.value <= 15:
                    score = 1.0
                elif reading.value <= 30:
                    score = 1.0 - (reading.value - 15) / 15
                else:
                    score = 0.1
                health_scores.append(score)
                
            elif reading.sensor_type == 'power':
                # Power consumption health score (0-1)
                if 80 <= reading.value <= 300:
                    score = 1.0
                else:
                    deviation = abs(reading.value - 190) / 190  # 190 is optimal
                    score = max(0.1, 1.0 - deviation)
                health_scores.append(score)
        
        # Return weighted average
        return sum(health_scores) / len(health_scores) if health_scores else 1.0
    
    def generate_predictive_maintenance_alerts(self, train_id: str, 
                                             readings: List[SensorReading]) -> List[Dict]:
        """Generate predictive maintenance alerts based on sensor trends"""
        alerts = []
        
        # Group readings by sensor type
        sensor_data = {}
        for reading in readings:
            if reading.sensor_type not in sensor_data:
                sensor_data[reading.sensor_type] = []
            sensor_data[reading.sensor_type].append(reading)
        
        # Analyze trends for each sensor type
        for sensor_type, sensor_readings in sensor_data.items():
            if len(sensor_readings) < 3:
                continue
                
            # Sort by timestamp
            sensor_readings.sort(key=lambda x: x.timestamp)
            values = [r.value for r in sensor_readings[-10:]]  # Last 10 readings
            
            # Calculate trend (simple linear regression slope)
            if len(values) >= 3:
                x = list(range(len(values)))
                trend_slope = np.polyfit(x, values, 1)[0]
                
                # Generate alerts based on concerning trends
                if sensor_type == 'temperature' and trend_slope > 2:
                    alerts.append({
                        'type': 'predictive_maintenance',
                        'severity': 'warning',
                        'message': f'Temperature trending upward for {train_id} - cooling system check recommended',
                        'trend_slope': trend_slope,
                        'sensor_type': sensor_type
                    })
                elif sensor_type == 'vibration' and trend_slope > 1:
                    alerts.append({
                        'type': 'predictive_maintenance',
                        'severity': 'warning',
                        'message': f'Vibration increasing for {train_id} - mechanical inspection recommended',
                        'trend_slope': trend_slope,
                        'sensor_type': sensor_type
                    })
        
        return alerts

class IoTWebSocketServer:
    """WebSocket server for real-time IoT data streaming"""
    
    def __init__(self, sensor_simulator: IoTSensorSimulator, port: int = 8765):
        self.sensor_simulator = sensor_simulator
        self.port = port
        self.connected_clients = set()
        
    async def register_client(self, websocket, path):
        """Register new WebSocket client"""
        self.connected_clients.add(websocket)
        logger.info(f"📱 WebSocket client connected. Total clients: {len(self.connected_clients)}")
        
        try:
            await websocket.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"📱 WebSocket client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def broadcast_sensor_data(self):
        """Broadcast latest sensor data to all connected clients"""
        while True:
            try:
                if self.connected_clients:
                    # Get latest readings
                    latest_readings = self.sensor_simulator.get_latest_readings(limit=50)
                    
                    if latest_readings:
                        # Convert to JSON
                        data = {
                            'type': 'sensor_update',
                            'timestamp': datetime.now().isoformat(),
                            'readings': [asdict(r) for r in latest_readings]
                        }
                        
                        # Broadcast to all clients
                        message = json.dumps(data, default=str)
                        
                        # Remove disconnected clients
                        disconnected = set()
                        for client in self.connected_clients.copy():
                            try:
                                await client.send(message)
                            except Exception:
                                disconnected.add(client)
                        
                        # Clean up disconnected clients
                        self.connected_clients -= disconnected
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                await asyncio.sleep(10)
    
    def start_server(self):
        """Start WebSocket server"""
        logger.info(f"🌐 Starting IoT WebSocket server on port {self.port}")
        start_server = websockets.serve(self.register_client, "localhost", self.port)
        
        # Create event loop for broadcasting
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(
            start_server,
            self.broadcast_sensor_data()
        ))
