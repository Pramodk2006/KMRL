from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import logging
import qrcode
from PIL import Image
import sqlite3
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileAPIServer:
    """Mobile API server for field operations"""
    
    def __init__(self, digital_twin_engine, iot_system, cv_system, port: int = 5000):
        self.digital_twin = digital_twin_engine
        self.iot_system = iot_system
        self.cv_system = cv_system
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for mobile apps
        
        # Setup database for mobile operations
        self.setup_database()
        
        # Setup API routes
        self.setup_routes()
        
        # Active field operations
        self.active_operations = {}
        
    def setup_database(self):
        """Setup SQLite database for mobile operations"""
        self.db_connection = sqlite3.connect('mobile_operations.db', check_same_thread=False)
        cursor = self.db_connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS field_operations (
                operation_id TEXT PRIMARY KEY,
                train_id TEXT,
                operator_id TEXT,
                operation_type TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                location TEXT,
                notes TEXT,
                photos TEXT  -- JSON array of photo data
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mobile_users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                role TEXT,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                active INTEGER DEFAULT 1,
                last_login TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_checklist (
                checklist_id TEXT PRIMARY KEY,
                train_id TEXT,
                inspector_id TEXT,
                checklist_type TEXT,
                items TEXT,  -- JSON array of checklist items
                completed_at TEXT,
                overall_score INTEGER
            )
        ''')
        
        self.db_connection.commit()
        
        # Insert sample mobile users
        self.create_sample_users()
    
    def create_sample_users(self):
        """Create sample mobile users for demo"""
        sample_users = [
            {
                'user_id': 'mobile_001',
                'username': 'inspector1',
                'role': 'field_inspector',
                'full_name': 'Rajesh Kumar',
                'email': 'rajesh.kumar@kmrl.com',
                'phone': '+91-9876543210'
            },
            {
                'user_id': 'mobile_002',
                'username': 'technician1',
                'role': 'maintenance_technician',
                'full_name': 'Priya Nair',
                'email': 'priya.nair@kmrl.com',
                'phone': '+91-9876543211'
            },
            {
                'user_id': 'mobile_003',
                'username': 'supervisor1',
                'role': 'maintenance_supervisor',
                'full_name': 'Arun Menon',
                'email': 'arun.menon@kmrl.com',
                'phone': '+91-9876543212'
            }
        ]
        
        cursor = self.db_connection.cursor()
        
        for user in sample_users:
            cursor.execute('''
                INSERT OR REPLACE INTO mobile_users 
                (user_id, username, role, full_name, email, phone, active, last_login)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            ''', (
                user['user_id'], user['username'], user['role'],
                user['full_name'], user['email'], user['phone'],
                datetime.now().isoformat()
            ))
        
        self.db_connection.commit()
        logger.info("✅ Sample mobile users created")
    
    def setup_routes(self):
        """Setup Flask API routes for mobile app"""
        
        @self.app.route('/mobile/health', methods=['GET'])
        def mobile_health():
            """Mobile API health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '6.0.0',
                'services': {
                    'digital_twin': self.digital_twin.is_running,
                    'iot_system': self.iot_system.is_running if hasattr(self.iot_system, 'is_running') else True,
                    'computer_vision': True
                }
            })
        
        @self.app.route('/mobile/auth/login', methods=['POST'])
        def mobile_login():
            """Mobile user authentication"""
            data = request.json
            username = data.get('username')
            password = data.get('password')  # In production, verify against secure hash
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT user_id, role, full_name, email, phone 
                FROM mobile_users 
                WHERE username = ? AND active = 1
            ''', (username,))
            
            user = cursor.fetchone()
            
            if user and password == 'kmrl2025':  # Simple demo authentication
                # Update last login
                cursor.execute('''
                    UPDATE mobile_users SET last_login = ? WHERE user_id = ?
                ''', (datetime.now().isoformat(), user[0]))
                self.db_connection.commit()
                
                return jsonify({
                    'success': True,
                    'user': {
                        'user_id': user[0],
                        'role': user[1],
                        'full_name': user[2],
                        'email': user[3],
                        'phone': user[4]
                    },
                    'token': f"mobile_token_{user[0]}_{int(datetime.now().timestamp())}"
                })
            else:
                return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
        
        @self.app.route('/mobile/trains', methods=['GET'])
        def get_mobile_trains():
            """Get train list for mobile app"""
            trains_data = []
            
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            
            for train_id, train_info in trains.items():
                # Get latest IoT data
                latest_readings = self.iot_system.get_latest_readings(train_id=train_id, limit=5)
                
                # Calculate health score
                health_score = 85.0  # Default
                if latest_readings:
                    processor = IoTDataProcessor()
                    health_score = processor.calculate_train_health_score(train_id, latest_readings) * 100
                
                trains_data.append({
                    'train_id': train_id,
                    'status': train_info.get('status', 'unknown'),
                    'location': train_info.get('location', 'depot'),
                    'health_score': round(health_score, 1),
                    'last_maintenance': train_info.get('last_maintenance', '2025-08-15'),
                    'mileage_km': train_info.get('mileage_km', 0),
                    'alerts_count': len([r for r in latest_readings if r.alert_level != 'normal'])
                })
            
            return jsonify({
                'success': True,
                'trains': trains_data,
                'total_count': len(trains_data)
            })
        
        @self.app.route('/mobile/train/<train_id>/details', methods=['GET'])
        def get_train_details(train_id):
            """Get detailed train information for mobile app"""
            try:
                # Get train state
                current_state = self.digital_twin.get_current_state()
                train_info = current_state.get('trains', {}).get(train_id, {})
                
                if not train_info:
                    return jsonify({'success': False, 'message': 'Train not found'}), 404
                
                # Get IoT sensor data
                latest_readings = self.iot_system.get_latest_readings(train_id=train_id, limit=20)
                sensor_data = {}
                
                for reading in latest_readings:
                    if reading.sensor_type not in sensor_data:
                        sensor_data[reading.sensor_type] = []
                    sensor_data[reading.sensor_type].append({
                        'value': reading.value,
                        'unit': reading.unit,
                        'timestamp': reading.timestamp.isoformat(),
                        'alert_level': reading.alert_level
                    })
                
                # Get recent inspection results
                inspection_history = self.cv_system.get_inspection_history(train_id=train_id, days=7)
                
                return jsonify({
                    'success': True,
                    'train_details': {
                        'train_id': train_id,
                        'basic_info': train_info,
                        'sensor_data': sensor_data,
                        'recent_inspections': len(inspection_history),
                        'last_inspection': inspection_history[0].timestamp.isoformat() if inspection_history else None
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting train details: {e}")
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/mobile/inspection/start', methods=['POST'])
        def start_inspection():
            """Start mobile inspection operation"""
            data = request.json
            train_id = data.get('train_id')
            inspector_id = data.get('inspector_id')
            inspection_type = data.get('inspection_type', 'routine')
            
            if not train_id or not inspector_id:
                return jsonify({'success': False, 'message': 'Missing required fields'}), 400
            
            operation_id = str(uuid.uuid4())
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO field_operations 
                (operation_id, train_id, operator_id, operation_type, status, started_at, location)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                operation_id, train_id, inspector_id, f'inspection_{inspection_type}',
                'in_progress', datetime.now().isoformat(), 'depot'
            ))
            self.db_connection.commit()
            
            self.active_operations[operation_id] = {
                'train_id': train_id,
                'inspector_id': inspector_id,
                'type': inspection_type,
                'started_at': datetime.now()
            }
            
            return jsonify({
                'success': True,
                'operation_id': operation_id,
                'message': f'Inspection started for train {train_id}'
            })
        
        @self.app.route('/mobile/inspection/<operation_id>/checklist', methods=['GET'])
        def get_inspection_checklist(operation_id):
            """Get inspection checklist for mobile app"""
            if operation_id not in self.active_operations:
                return jsonify({'success': False, 'message': 'Operation not found'}), 404
            
            # Generate dynamic checklist based on train type and inspection type
            checklist_items = [
                {'id': 1, 'category': 'Exterior', 'item': 'Check body condition for dents/scratches', 'status': 'pending'},
                {'id': 2, 'category': 'Exterior', 'item': 'Inspect door mechanisms', 'status': 'pending'},
                {'id': 3, 'category': 'Exterior', 'item': 'Check window condition', 'status': 'pending'},
                {'id': 4, 'category': 'Wheels', 'item': 'Inspect wheel condition and wear', 'status': 'pending'},
                {'id': 5, 'category': 'Wheels', 'item': 'Check brake disc condition', 'status': 'pending'},
                {'id': 6, 'category': 'Electrical', 'item': 'Test pantograph operation', 'status': 'pending'},
                {'id': 7, 'category': 'Electrical', 'item': 'Check electrical connections', 'status': 'pending'},
                {'id': 8, 'category': 'Interior', 'item': 'Inspect seats and interior fittings', 'status': 'pending'},
                {'id': 9, 'category': 'Safety', 'item': 'Test emergency systems', 'status': 'pending'},
                {'id': 10, 'category': 'Safety', 'item': 'Check fire extinguisher', 'status': 'pending'}
            ]
            
            return jsonify({
                'success': True,
                'checklist': {
                    'operation_id': operation_id,
                    'train_id': self.active_operations[operation_id]['train_id'],
                    'total_items': len(checklist_items),
                    'completed_items': 0,
                    'items': checklist_items
                }
            })
        
        @self.app.route('/mobile/inspection/<operation_id>/complete', methods=['POST'])
        def complete_inspection(operation_id):
            """Complete mobile inspection with results"""
            if operation_id not in self.active_operations:
                return jsonify({'success': False, 'message': 'Operation not found'}), 404
            
            data = request.json
            checklist_results = data.get('checklist_results', [])
            notes = data.get('notes', '')
            photos = data.get('photos', [])
            
            # Calculate overall score
            total_items = len(checklist_results)
            passed_items = len([item for item in checklist_results if item.get('status') == 'passed'])
            overall_score = int((passed_items / total_items) * 100) if total_items > 0 else 0
            
            # Update database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                UPDATE field_operations 
                SET status = ?, completed_at = ?, notes = ?, photos = ?
                WHERE operation_id = ?
            ''', (
                'completed', datetime.now().isoformat(), notes,
                json.dumps(photos), operation_id
            ))
            
            # Store checklist results
            cursor.execute('''
                INSERT INTO inspection_checklist 
                (checklist_id, train_id, inspector_id, checklist_type, items, completed_at, overall_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"checklist_{operation_id}", 
                self.active_operations[operation_id]['train_id'],
                self.active_operations[operation_id]['inspector_id'],
                'mobile_inspection',
                json.dumps(checklist_results),
                datetime.now().isoformat(),
                overall_score
            ))
            
            self.db_connection.commit()
            
            # Remove from active operations
            del self.active_operations[operation_id]
            
            return jsonify({
                'success': True,
                'message': 'Inspection completed successfully',
                'overall_score': overall_score,
                'summary': {
                    'total_items': total_items,
                    'passed_items': passed_items,
                    'failed_items': total_items - passed_items
                }
            })
        
        @self.app.route('/mobile/qr/<train_id>', methods=['GET'])
        def generate_train_qr(train_id):
            """Generate QR code for train identification"""
            try:
                # Create QR code data
                qr_data = {
                    'train_id': train_id,
                    'type': 'kmrl_train',
                    'generated_at': datetime.now().isoformat(),
                    'api_endpoint': f'/mobile/train/{train_id}/details'
                }
                
                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(json.dumps(qr_data))
                qr.make(fit=True)
                
                # Create image
                qr_img = qr.make_image(fill_color="black", back_color="white")
                
                # Convert to bytes
                img_io = io.BytesIO()
                qr_img.save(img_io, 'PNG')
                img_io.seek(0)
                
                return send_file(img_io, mimetype='image/png')
                
            except Exception as e:
                logger.error(f"Error generating QR code: {e}")
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/mobile/alerts', methods=['GET'])
        def get_mobile_alerts():
            """Get alerts for mobile app"""
            try:
                # Get IoT alerts
                iot_alerts = self.iot_system.get_alerts()
                
                mobile_alerts = []
                for alert in iot_alerts:
                    mobile_alerts.append({
                        'alert_id': f"iot_{alert.sensor_id}_{int(alert.timestamp.timestamp())}",
                        'type': 'sensor_alert',
                        'train_id': alert.train_id,
                        'severity': alert.alert_level,
                        'message': f"{alert.sensor_type.title()} alert: {alert.value}{alert.unit}",
                        'timestamp': alert.timestamp.isoformat(),
                        'location': alert.location
                    })
                
                # Sort by timestamp (newest first)
                mobile_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'alerts': mobile_alerts[:20],  # Limit to 20 most recent
                    'total_count': len(mobile_alerts)
                })
                
            except Exception as e:
                logger.error(f"Error getting mobile alerts: {e}")
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/mobile/reports/daily', methods=['GET'])
        def get_daily_mobile_report():
            """Get daily summary report for mobile app"""
            try:
                today = datetime.now().date()
                
                # Get today's operations from database
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    SELECT operation_type, status, COUNT(*) as count
                    FROM field_operations
                    WHERE DATE(started_at) = ?
                    GROUP BY operation_type, status
                ''', (today.isoformat(),))
                
                operations_summary = cursor.fetchall()
                
                # Get today's inspections
                cursor.execute('''
                    SELECT AVG(overall_score) as avg_score, COUNT(*) as total_inspections
                    FROM inspection_checklist
                    WHERE DATE(completed_at) = ?
                ''', (today.isoformat(),))
                
                inspection_summary = cursor.fetchone()
                
                # Get train status from digital twin
                current_state = self.digital_twin.get_current_state()
                summary = current_state.get('summary', {})
                
                daily_report = {
                    'date': today.isoformat(),
                    'train_operations': {
                        'total_trains': summary.get('total_trains', 0),
                        'inducted_trains': summary.get('inducted_trains', 0),
                        'available_bays': summary.get('available_bays', 0),
                        'bay_utilization': summary.get('bay_utilization', 0)
                    },
                    'field_operations': {
                        'operations_summary': [
                            {'type': op[0], 'status': op[1], 'count': op[2]}
                            for op in operations_summary
                        ],
                        'total_operations': sum([op[2] for op in operations_summary])
                    },
                    'inspections': {
                        'total_inspections': inspection_summary[1] if inspection_summary[1] else 0,
                        'average_score': round(inspection_summary[0], 1) if inspection_summary[0] else 0
                    },
                    'alerts': {
                        'active_alerts': len(self.iot_system.get_alerts())
                    }
                }
                
                return jsonify({
                    'success': True,
                    'daily_report': daily_report
                })
                
            except Exception as e:
                logger.error(f"Error generating daily mobile report: {e}")
                return jsonify({'success': False, 'message': str(e)}), 500
    
    def start_server(self):
        """Start mobile API server"""
        logger.info(f"📱 Starting Mobile API Server on port {self.port}")
        
        # Run Flask server in a separate thread
        server_thread = threading.Thread(
            target=lambda: self.app.run(
                host='0.0.0.0', 
                port=self.port, 
                debug=False,
                threaded=True
            ),
            daemon=True
        )
        server_thread.start()
        
        logger.info(f"📱 Mobile API Server running at http://localhost:{self.port}")
        logger.info("🔗 API endpoints available:")
        logger.info("   - GET /mobile/health")
        logger.info("   - POST /mobile/auth/login")
        logger.info("   - GET /mobile/trains")
        logger.info("   - GET /mobile/train/<train_id>/details")
        logger.info("   - GET /mobile/qr/<train_id>")
        logger.info("   - GET /mobile/alerts")
        logger.info("   - GET /mobile/reports/daily")
        
        return server_thread

# Import required from other modules
try:
    from .iot_sensor_system import IoTDataProcessor
except ImportError:
    # Fallback if module not available
    class IoTDataProcessor:
        def calculate_train_health_score(self, train_id, readings):
            return 0.85
