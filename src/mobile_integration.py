"""
src/mobile_integration.py

Mobile API server for KMRL IntelliFleet mobile operations.
Provides field worker interface for inspections, updates, and alerts.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import time

logger = logging.getLogger(__name__)

class MobileAPIServer:
    """Mobile API server for field operations and inspections"""
    
    def __init__(self, digital_twin, iot_simulator, cv_system, port=5000):
        self.digital_twin = digital_twin
        self.iot_simulator = iot_simulator
        self.cv_system = cv_system
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize mobile database
        self._init_mobile_db()
        
        # Setup routes
        self._setup_routes()
        
        # Mobile session tracking
        self.active_sessions = {}
        self.pending_tasks = {}
        
        logger.info("Mobile API Server initialized")

    def _init_mobile_db(self):
        """Initialize mobile operations database"""
        try:
            conn = sqlite3.connect('mobile_operations.db')
            cursor = conn.cursor()
            
            # Create tables for mobile operations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mobile_inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    train_id TEXT NOT NULL,
                    inspector_id TEXT NOT NULL,
                    inspection_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    findings TEXT,
                    images TEXT,  -- JSON array of image paths
                    location TEXT,
                    timestamp TEXT NOT NULL,
                    completed_at TEXT,
                    priority INTEGER DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mobile_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    train_id TEXT,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mobile_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    login_time TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    device_info TEXT,
                    location TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Mobile database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize mobile database: {e}")

    def _setup_routes(self):
        """Setup mobile API routes"""
        
        @self.app.route('/')
        def mobile_dashboard():
            """Mobile dashboard for field workers"""
            return render_template_string(self._get_mobile_dashboard_template())
        
        @self.app.route('/api/mobile/login', methods=['POST'])
        def mobile_login():
            """Mobile user authentication"""
            data = request.get_json()
            user_id = data.get('user_id')
            password = data.get('password')
            device_info = data.get('device_info', {})
            location = data.get('location', 'Unknown')
            
            # Simple authentication (in production, use proper auth)
            if user_id and password:
                session_id = f"mobile_{user_id}_{int(time.time())}"
                
                # Store session
                self.active_sessions[session_id] = {
                    'user_id': user_id,
                    'role': 'field_inspector',  # Could be determined from user_id
                    'login_time': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat(),
                    'device_info': device_info,
                    'location': location
                }
                
                # Store in database
                try:
                    conn = sqlite3.connect('mobile_operations.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO mobile_sessions 
                        (session_id, user_id, role, login_time, last_activity, device_info, location)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, user_id, 'field_inspector', 
                          datetime.now().isoformat(), datetime.now().isoformat(),
                          json.dumps(device_info), location))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to store session: {e}")
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'user_info': {
                        'user_id': user_id,
                        'role': 'field_inspector',
                        'permissions': ['inspect', 'alert', 'update_status']
                    }
                })
            
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        @self.app.route('/api/mobile/trains', methods=['GET'])
        def get_mobile_train_list():
            """Get train list for mobile inspection"""
            session_id = request.headers.get('Session-ID')
            if not self._validate_session(session_id):
                return jsonify({'error': 'Invalid session'}), 401
            
            try:
                # Get current train state from digital twin
                state = self.digital_twin.get_current_state()
                trains = state.get('trains', {})
                
                mobile_trains = []
                for train_id, train_data in trains.items():
                    # Get IoT data if available
                    iot_data = {}
                    if hasattr(self.iot_simulator, 'get_train_readings'):
                        iot_data = self.iot_simulator.get_train_readings(train_id)
                    
                    mobile_trains.append({
                        'train_id': train_id,
                        'status': train_data.get('status', 'unknown'),
                        'location': train_data.get('location', 'unknown'),
                        'mileage_km': train_data.get('mileage_km', 0),
                        'fitness_valid_until': train_data.get('fitness_valid_until', ''),
                        'priority_score': train_data.get('priority_score', 0),
                        'bay_assignment': train_data.get('bay_assignment', 'N/A'),
                        'last_inspection': self._get_last_inspection(train_id),
                        'iot_status': iot_data.get('health_score', 100),
                        'alerts_count': self._get_active_alerts_count(train_id)
                    })
                
                return jsonify({
                    'success': True,
                    'trains': mobile_trains,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting mobile train list: {e}")
                return jsonify({'error': 'Failed to get train list'}), 500
        
        @self.app.route('/api/mobile/train/<train_id>/inspect', methods=['POST'])
        def start_mobile_inspection(train_id):
            """Start mobile inspection for a train"""
            session_id = request.headers.get('Session-ID')
            if not self._validate_session(session_id):
                return jsonify({'error': 'Invalid session'}), 401
            
            data = request.get_json()
            inspection_type = data.get('inspection_type', 'general')
            location = data.get('location', 'depot')
            
            session = self.active_sessions.get(session_id)
            inspector_id = session['user_id']
            
            try:
                conn = sqlite3.connect('mobile_operations.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO mobile_inspections 
                    (train_id, inspector_id, inspection_type, status, location, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (train_id, inspector_id, inspection_type, 'in_progress', 
                      location, datetime.now().isoformat()))
                
                inspection_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                # Trigger CV system if visual inspection
                cv_results = {}
                if inspection_type in ['visual', 'exterior', 'interior']:
                    try:
                        cv_results = self.cv_system.inspect_train(train_id)
                    except Exception as e:
                        logger.warning(f"CV inspection failed: {e}")
                        cv_results = {'status': 'cv_unavailable'}
                
                return jsonify({
                    'success': True,
                    'inspection_id': inspection_id,
                    'train_id': train_id,
                    'inspector_id': inspector_id,
                    'cv_results': cv_results,
                    'started_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error starting inspection: {e}")
                return jsonify({'error': 'Failed to start inspection'}), 500
        
        @self.app.route('/api/mobile/inspection/<int:inspection_id>/complete', methods=['POST'])
        def complete_mobile_inspection(inspection_id):
            """Complete mobile inspection"""
            session_id = request.headers.get('Session-ID')
            if not self._validate_session(session_id):
                return jsonify({'error': 'Invalid session'}), 401
            
            data = request.get_json()
            findings = data.get('findings', '')
            images = data.get('images', [])  # Base64 encoded images
            status = data.get('status', 'completed')
            
            try:
                # Store images (in production, save to file system or cloud)
                image_paths = []
                for idx, image_data in enumerate(images):
                    # Here you would save the image and store the path
                    image_path = f"inspection_{inspection_id}_{idx}.jpg"
                    image_paths.append(image_path)
                
                conn = sqlite3.connect('mobile_operations.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE mobile_inspections 
                    SET status = ?, findings = ?, images = ?, completed_at = ?
                    WHERE id = ?
                ''', (status, findings, json.dumps(image_paths), 
                      datetime.now().isoformat(), inspection_id))
                conn.commit()
                conn.close()
                
                # Create alert if issues found
                if 'issue' in findings.lower() or 'problem' in findings.lower():
                    self._create_mobile_alert(
                        alert_type='inspection_issue',
                        message=f"Inspection {inspection_id} found issues: {findings[:100]}",
                        severity='medium',
                        train_id=self._get_inspection_train_id(inspection_id)
                    )
                
                return jsonify({
                    'success': True,
                    'inspection_id': inspection_id,
                    'status': status,
                    'completed_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error completing inspection: {e}")
                return jsonify({'error': 'Failed to complete inspection'}), 500
        
        @self.app.route('/api/mobile/alerts', methods=['GET'])
        def get_mobile_alerts():
            """Get mobile alerts for field workers"""
            session_id = request.headers.get('Session-ID')
            if not self._validate_session(session_id):
                return jsonify({'error': 'Invalid session'}), 401
            
            try:
                conn = sqlite3.connect('mobile_operations.db')
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, alert_type, train_id, message, severity, 
                           acknowledged, created_at, acknowledged_at, acknowledged_by
                    FROM mobile_alerts 
                    WHERE acknowledged = FALSE 
                    ORDER BY created_at DESC 
                    LIMIT 50
                ''')
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'id': row[0],
                        'alert_type': row[1],
                        'train_id': row[2],
                        'message': row[3],
                        'severity': row[4],
                        'acknowledged': bool(row[5]),
                        'created_at': row[6],
                        'acknowledged_at': row[7],
                        'acknowledged_by': row[8]
                    })
                
                conn.close()
                
                return jsonify({
                    'success': True,
                    'alerts': alerts,
                    'count': len(alerts)
                })
                
            except Exception as e:
                logger.error(f"Error getting mobile alerts: {e}")
                return jsonify({'error': 'Failed to get alerts'}), 500
        
        @self.app.route('/api/mobile/alert/<int:alert_id>/acknowledge', methods=['POST'])
        def acknowledge_mobile_alert(alert_id):
            """Acknowledge mobile alert"""
            session_id = request.headers.get('Session-ID')
            if not self._validate_session(session_id):
                return jsonify({'error': 'Invalid session'}), 401
            
            session = self.active_sessions.get(session_id)
            user_id = session['user_id']
            
            try:
                conn = sqlite3.connect('mobile_operations.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE mobile_alerts 
                    SET acknowledged = TRUE, acknowledged_at = ?, acknowledged_by = ?
                    WHERE id = ?
                ''', (datetime.now().isoformat(), user_id, alert_id))
                conn.commit()
                conn.close()
                
                return jsonify({
                    'success': True,
                    'alert_id': alert_id,
                    'acknowledged_by': user_id,
                    'acknowledged_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
                return jsonify({'error': 'Failed to acknowledge alert'}), 500

    def _validate_session(self, session_id):
        """Validate mobile session"""
        if not session_id or session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check if session is still active (24 hour timeout)
        login_time = datetime.fromisoformat(session['login_time'])
        if datetime.now() - login_time > timedelta(hours=24):
            del self.active_sessions[session_id]
            return False
        
        # Update last activity
        session['last_activity'] = datetime.now().isoformat()
        return True

    def _get_last_inspection(self, train_id):
        """Get last inspection date for train"""
        try:
            conn = sqlite3.connect('mobile_operations.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT completed_at FROM mobile_inspections 
                WHERE train_id = ? AND status = 'completed' 
                ORDER BY completed_at DESC LIMIT 1
            ''', (train_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 'Never'
        except:
            return 'Unknown'

    def _get_active_alerts_count(self, train_id):
        """Get count of active alerts for train"""
        try:
            conn = sqlite3.connect('mobile_operations.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM mobile_alerts 
                WHERE train_id = ? AND acknowledged = FALSE
            ''', (train_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 0
        except:
            return 0

    def _create_mobile_alert(self, alert_type, message, severity, train_id=None):
        """Create mobile alert"""
        try:
            conn = sqlite3.connect('mobile_operations.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO mobile_alerts 
                (alert_type, train_id, message, severity, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_type, train_id, message, severity, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to create mobile alert: {e}")

    def _get_inspection_train_id(self, inspection_id):
        """Get train ID for inspection"""
        try:
            conn = sqlite3.connect('mobile_operations.db')
            cursor = conn.cursor()
            cursor.execute('SELECT train_id FROM mobile_inspections WHERE id = ?', (inspection_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except:
            return None

    def _get_mobile_dashboard_template(self):
        """Simple mobile dashboard HTML template"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>KMRL Mobile Operations</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 400px; margin: 0 auto; }
        .header { background: #2196F3; color: white; padding: 15px; text-align: center; border-radius: 8px; }
        .card { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; }
        .status.ready { background: #4CAF50; color: white; }
        .status.maintenance { background: #FF9800; color: white; }
        .status.ineligible { background: #F44336; color: white; }
        .btn { background: #2196F3; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
        .alert { background: #ffebcd; border-left: 4px solid #ff9800; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>KMRL Mobile Operations</h2>
        </div>
        <div class="card">
            <h3>Field Inspector Dashboard</h3>
            <p>Mobile API server running on port 5000</p>
            <div class="alert">
                <strong>API Endpoints:</strong><br>
                • POST /api/mobile/login<br>
                • GET /api/mobile/trains<br>
                • POST /api/mobile/train/&lt;id&gt;/inspect<br>
                • GET /api/mobile/alerts
            </div>
        </div>
    </div>
</body>
</html>
        '''

    def start_server(self):
        """Start the mobile API server"""
        try:
            logger.info(f"Starting Mobile API server on port {self.port}")
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"Failed to start mobile server: {e}")

    def create_sample_alerts(self):
        """Create sample alerts for testing"""
        sample_alerts = [
            {
                'alert_type': 'maintenance_due',
                'train_id': 'KMRL_001',
                'message': 'Preventive maintenance due in 2 days',
                'severity': 'medium'
            },
            {
                'alert_type': 'inspection_overdue', 
                'train_id': 'KMRL_002',
                'message': 'Visual inspection overdue by 3 days',
                'severity': 'high'
            },
            {
                'alert_type': 'iot_anomaly',
                'train_id': 'KMRL_003', 
                'message': 'Temperature sensor reading anomaly detected',
                'severity': 'low'
            }
        ]
        
        for alert in sample_alerts:
            self._create_mobile_alert(**alert)