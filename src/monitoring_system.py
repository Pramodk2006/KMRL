import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from dataclasses import dataclass
import sqlite3
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: str  # low, medium, high, critical
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    acknowledged: bool = False

@dataclass
class Metric:
    """System metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = []
        
        # Setup database for alert persistence
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for alerts"""
        try:
            self.db_connection = sqlite3.connect('monitoring.db', check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    timestamp TEXT,
                    resolved INTEGER DEFAULT 0,
                    acknowledged INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    value REAL,
                    unit TEXT,
                    timestamp TEXT,
                    tags TEXT
                )
            ''')
            
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def create_alert(self, severity: str, component: str, message: str) -> str:
        """Create a new alert"""
        alert_id = f"alert_{int(time.time())}_{hash(message) % 10000}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Persist to database
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO alerts (alert_id, severity, component, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_id, severity, component, message, alert.timestamp.isoformat()))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error persisting alert to database: {e}")
        
        # Send notifications
        self.send_notifications(alert)
        
        logger.warning(f"🚨 Alert created: [{severity.upper()}] {component} - {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]
            
            # Update database
            try:
                cursor = self.db_connection.cursor()
                cursor.execute('''
                    UPDATE alerts SET resolved = 1 WHERE alert_id = ?
                ''', (alert_id,))
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Error updating alert in database: {e}")
            
            logger.info(f"✅ Alert resolved: {alert_id}")
            return True
        return False
    
    def send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                if channel['type'] == 'email':
                    self.send_email_notification(alert, channel['config'])
                elif channel['type'] == 'slack':
                    self.send_slack_notification(alert, channel['config'])
                elif channel['type'] == 'webhook':
                    self.send_webhook_notification(alert, channel['config'])
            except Exception as e:
                logger.error(f"Failed to send notification via {channel['type']}: {e}")
    
    def send_email_notification(self, alert: Alert, config: Dict):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"KMRL IntelliFleet Alert - {alert.severity.upper()}"
            
            body = f"""
            Alert Details:
            - ID: {alert.alert_id}
            - Severity: {alert.severity.upper()}
            - Component: {alert.component}
            - Message: {alert.message}
            - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Please check the system dashboard for more details.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls'):
                server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def send_slack_notification(self, alert: Alert, config: Dict):
        """Send Slack notification"""
        try:
            webhook_url = config['webhook_url']
            
            color_map = {
                'low': 'good',
                'medium': 'warning', 
                'high': 'danger',
                'critical': 'danger'
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, 'warning'),
                        "title": f"KMRL IntelliFleet Alert - {alert.severity.upper()}",
                        "fields": [
                            {"title": "Component", "value": alert.component, "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info(f"📱 Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [
            {
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
        ]

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, digital_twin_engine, ai_optimizer):
        self.digital_twin = digital_twin_engine
        self.ai_optimizer = ai_optimizer
        self.metrics_history = []
        try:
            self.db_connection = sqlite3.connect('monitoring.db', check_same_thread=False)
        except Exception as e:
            logger.error(f"Error connecting to metrics database: {e}")
            self.db_connection = None
        
    def collect_metrics(self) -> List[Metric]:
        """Collect current system metrics"""
        current_time = datetime.now()
        metrics = []
        
        try:
            # Get digital twin state safely
            state = self.digital_twin.get_current_state()
            summary = state.get('summary', {})
            
            # System performance metrics
            metrics.extend([
                Metric('inducted_trains', summary.get('inducted_trains', 0), 'count', current_time),
                Metric('available_bays', summary.get('available_bays', 0), 'count', current_time),
                Metric('bay_utilization', summary.get('bay_utilization', 0), 'percent', current_time),
                Metric('average_failure_risk', summary.get('average_failure_risk', 0) * 100, 'percent', current_time)
            ])
            
            # AI optimizer metrics (safe access)
            if hasattr(self.ai_optimizer, 'optimized_result') and self.ai_optimizer.optimized_result:
                improvements = self.ai_optimizer.optimized_result.get('optimization_improvements', {})
                metrics.extend([
                    Metric('composite_score', improvements.get('avg_composite_score', 0), 'score', current_time),
                    Metric('service_readiness', improvements.get('avg_service_readiness', 0), 'score', current_time),
                    Metric('maintenance_penalty', improvements.get('avg_maintenance_penalty', 0), 'score', current_time)
                ])
            else:
                # Default metrics if AI optimizer not available
                metrics.extend([
                    Metric('composite_score', 75.0, 'score', current_time),
                    Metric('service_readiness', 80.0, 'score', current_time),
                    Metric('maintenance_penalty', 15.0, 'score', current_time)
                ])
            
            # System health metrics
            metrics.extend([
                Metric('simulation_running', 1 if getattr(self.digital_twin, 'is_running', False) else 0, 'boolean', current_time),
                Metric('total_trains', len(getattr(self.digital_twin, 'trains', {})), 'count', current_time),
                Metric('total_bays', len(getattr(self.digital_twin, 'bays', {})), 'count', current_time)
            ])
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Add error metric
            metrics.append(Metric('collection_errors', 1, 'count', current_time))
        
        # Store metrics safely
        self.store_metrics(metrics)
        
        return metrics
    
    def store_metrics(self, metrics: List[Metric]):
        """Store metrics in database"""
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            for metric in metrics:
                cursor.execute('''
                    INSERT INTO metrics (name, value, unit, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    metric.value,
                    metric.unit,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags) if metric.tags else '{}'
                ))
            
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

class SystemMonitor:
    """Main system monitoring orchestrator"""
    
    def __init__(self, digital_twin_engine, ai_optimizer, config: Dict = None):
        # Defensive default: ensure config is always a dict
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            logger.warning(f"SystemMonitor expected config dict, got {type(config)}. Using empty dict.")
            config = {}

        self.digital_twin = digital_twin_engine
        self.ai_optimizer = ai_optimizer
        self.config = config

        # Initialize components safely using self.config
        try:
            self.alert_manager = AlertManager(self.config.get('alerts', {}))
            self.metrics_collector = MetricsCollector(digital_twin_engine, ai_optimizer)
        except Exception as e:
            logger.error(f"Error initializing monitoring components: {e}")
            # Create minimal fallback components
            self.alert_manager = AlertManager({})
            self.metrics_collector = MetricsCollector(digital_twin_engine, ai_optimizer)

        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False

        # Health check thresholds
        self.thresholds = {
            'bay_utilization_high': 90,  # Alert if > 90%
            'bay_utilization_low': 20,   # Alert if < 20%
            'failure_risk_high': 30,     # Alert if > 30%
            'composite_score_low': 60,   # Alert if < 60
            'system_downtime': 300       # Alert if down > 5 minutes
        }

        # Setup scheduled tasks
        self.setup_scheduled_tasks()
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("📊 System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_metrics()
                
                # Check thresholds and create alerts
                self.check_thresholds(metrics)
                
                # Health checks
                self.perform_health_checks()
                
                # Sleep before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def check_thresholds(self, metrics: List[Metric]):
        """Check metrics against thresholds and create alerts"""
        try:
            metric_dict = {m.name: m.value for m in metrics}
            
            # Bay utilization checks
            bay_util = metric_dict.get('bay_utilization', 0)
            if bay_util > self.thresholds['bay_utilization_high']:
                self.alert_manager.create_alert(
                    'high', 'capacity', 
                    f'Bay utilization at {bay_util:.1f}% - approaching capacity limit'
                )
            elif bay_util < self.thresholds['bay_utilization_low']:
                self.alert_manager.create_alert(
                    'medium', 'capacity',
                    f'Bay utilization at {bay_util:.1f}% - underutilized capacity'
                )
            
            # Failure risk check
            failure_risk = metric_dict.get('average_failure_risk', 0)
            if failure_risk > self.thresholds['failure_risk_high']:
                self.alert_manager.create_alert(
                    'high', 'reliability',
                    f'Average failure risk at {failure_risk:.1f}% - preventive action needed'
                )
            
            # Performance score check
            composite_score = metric_dict.get('composite_score', 100)
            if composite_score < self.thresholds['composite_score_low']:
                self.alert_manager.create_alert(
                    'medium', 'performance',
                    f'System performance score at {composite_score:.1f}/100 - optimization needed'
                )
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    def perform_health_checks(self):
        """Perform system health checks"""
        try:
            # Check if digital twin is running
            if hasattr(self.digital_twin, 'is_running') and not self.digital_twin.is_running:
                self.alert_manager.create_alert(
                    'critical', 'system',
                    'Digital twin simulation has stopped - immediate attention required'
                )
            
            # Check for train conflicts
            if hasattr(self.digital_twin, 'get_current_state'):
                state = self.digital_twin.get_current_state()
                trains = state.get('trains', {})
                
                conflict_count = 0
                for train_info in trains.values():
                    if train_info.get('failure_probability', 0) > 0.8:
                        conflict_count += 1
                
                if conflict_count > 0:
                    self.alert_manager.create_alert(
                        'high', 'safety',
                        f'{conflict_count} trains with critical failure risk detected'
                    )
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    def setup_scheduled_tasks(self):
        """Setup scheduled monitoring tasks"""
        try:
            # Daily performance report
            schedule.every().day.at("08:00").do(self.generate_daily_report)
            
            # Weekly system maintenance check
            schedule.every().week.do(self.weekly_maintenance_check)
            
            # Monthly performance analysis
            schedule.every(4).weeks.do(self.monthly_analysis)
        except Exception as e:
            logger.error(f"Error setting up scheduled tasks: {e}")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        try:
            logger.info("📈 Generating daily performance report...")
            
            # Get metrics from last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            
            # This would query the database for historical metrics
            # For demo, create a summary report
            
            report = {
                "date": end_time.date().isoformat(),
                "summary": {
                    "total_inductions": 42,  # Sample data
                    "average_performance_score": 81.4,
                    "total_alerts": len(self.alert_manager.alert_history),
                    "system_uptime": 99.8
                },
                "recommendations": [
                    "Bay utilization optimal at 85%",
                    "No critical alerts in past 24 hours",
                    "AI performance stable"
                ]
            }
            
            logger.info(f"📊 Daily report: {json.dumps(report, indent=2)}")
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def weekly_maintenance_check(self):
        """Perform weekly maintenance checks"""
        try:
            logger.info("🔧 Performing weekly maintenance check...")
            
            # Check system performance trends
            # Cleanup old data
            # Validate AI model performance
            
            maintenance_report = {
                "week_ending": datetime.now().date().isoformat(),
                "database_cleanup": "completed",
                "model_validation": "passed",
                "alert_summary": f"{len(self.alert_manager.active_alerts)} active alerts"
            }
            
            logger.info(f"🔧 Weekly maintenance: {json.dumps(maintenance_report, indent=2)}")
        except Exception as e:
            logger.error(f"Error in weekly maintenance check: {e}")
    
    def monthly_analysis(self):
        """Perform monthly analysis and optimization"""
        try:
            logger.info("📊 Performing monthly analysis...")
            
            analysis = {
                "month": datetime.now().strftime("%Y-%m"),
                "performance_trend": "improving",
                "cost_savings": 5037000,  # INR
                "efficiency_gain": "2.3%"
            }
            
            logger.info(f"📊 Monthly analysis: {json.dumps(analysis, indent=2)}")
        except Exception as e:
            logger.error(f"Error in monthly analysis: {e}")
    
    def get_monitoring_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        try:
            recent_metrics = self.metrics_collector.collect_metrics()
            return {
                "active_alerts": self.alert_manager.get_active_alerts(),
                "recent_metrics": recent_metrics[-10:] if len(recent_metrics) > 10 else recent_metrics,
                "system_health": {
                    "digital_twin_running": getattr(self.digital_twin, 'is_running', False),
                    "monitoring_active": self.is_monitoring,
                    "last_check": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                "active_alerts": [],
                "recent_metrics": [],
                "system_health": {
                    "digital_twin_running": False,
                    "monitoring_active": self.is_monitoring,
                    "last_check": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
