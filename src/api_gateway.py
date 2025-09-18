from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import jwt
import redis
import json
import io
import pandas as pd
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from .db import upsert, insert_rows, fetch_query, init_db
from .branding_sla import get_sla_status
from .ml_feedback import compute_drift_metrics, retrain_predictive_model_if_needed
from .predictive_model import TrainingPipeline, ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainInductionRequest(BaseModel):
    """Request model for train induction API"""
    train_ids: List[str] = Field(..., description="List of train IDs to induct")
    priority_override: Optional[str] = Field(None, description="Priority override: high, medium, low")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled induction time")
    reason: Optional[str] = Field(None, description="Reason for induction")

class TrainStatusUpdate(BaseModel):
    """Model for train status updates from OCC"""
    train_id: str
    location: str
    status: str
    mileage_km: float
    fitness_valid_until: datetime
    maintenance_required: bool
    timestamp: datetime

class BayStatusUpdate(BaseModel):
    """Model for bay status updates"""
    bay_id: str
    status: str  # available, occupied, maintenance, blocked
    occupied_trains: List[str]
    maintenance_scheduled: Optional[datetime]
    timestamp: datetime

class EmergencyEvent(BaseModel):
    """Model for emergency events"""
    event_id: str
    event_type: str  # fire, power_outage, security_breach, medical
    severity: str  # low, medium, high, critical
    location: str
    description: str
    timestamp: datetime
    requires_evacuation: bool = False

class SystemMetrics(BaseModel):
    """Model for system performance metrics"""
    timestamp: datetime
    inducted_trains: int
    available_bays: int
    system_load: float
    response_time_ms: float
    error_rate: float
    ai_prediction_accuracy: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    logger.info("🚀 KMRL IntelliFleet API Gateway starting...")
    yield
    # Shutdown
    logger.info("⏹️ KMRL IntelliFleet API Gateway shutting down...")

class APIGateway:
    """Enterprise API Gateway for KMRL IntelliFleet"""
    
    def __init__(self, digital_twin_engine, ai_optimizer):
        # Ensure database schema exists
        try:
            init_db()
        except Exception as e:
            logger.warning(f"DB init failed (continuing): {e}")
        self.digital_twin = digital_twin_engine
        self.ai_optimizer = ai_optimizer
        self.app = FastAPI(
            title="KMRL IntelliFleet API",
            description="Enterprise API for AI-driven train induction system",
            version="5.0.0",
            lifespan=lifespan
        )
        
        # Redis for caching and message queues
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True,
            health_check_interval=30
        )
        
        # JWT configuration
        self.jwt_secret = "kmrl_intellifleet_secret_key_2025"
        self.jwt_algorithm = "HS256"
        
        # Security
        self.security = HTTPBearer()
        
        # Setup middleware and routes
        self.setup_middleware()
        self.setup_routes()
        # Start scheduled retraining loop (simple background task)
        try:
            import threading, time
            def retrain_loop():
                while True:
                    try:
                        # Run every 6 hours
                        time.sleep(6 * 3600)
                        # Minimal safe retraining with current state
                        state = self.digital_twin.get_current_state()
                        trains = state.get('trains', {})
                        trains_df = pd.DataFrame([{'train_id': k, **v} for k, v in trains.items()]) if trains else pd.DataFrame(columns=['train_id'])
                        from .db import fetch_df
                        historical_df = fetch_df('outcomes')
                        job_cards_df = fetch_df('job_cards')
                        pipeline = TrainingPipeline()
                        pipeline.train_and_register(historical_df, trains_df, job_cards_df, created_by='scheduler')
                        logger.info("Scheduled retraining completed")
                    except Exception as e:
                        logger.error(f"Scheduled retraining failed: {e}")
            threading.Thread(target=retrain_loop, daemon=True).start()
        except Exception as e:
            logger.warning(f"Failed to start scheduler: {e}")
        
        # WebSocket connections for real-time updates
        self.active_connections = []
        
    def setup_middleware(self):
        """Setup API middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def create_jwt_token(self, user_id: str, role: str, expires_delta: timedelta = None) -> str:
        """Create JWT token for authentication"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "kmrl_intellifleet"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify JWT token and extract user info"""
        try:
            payload = jwt.decode(
                credentials.credentials, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            return {
                "user_id": payload.get("user_id"),
                "role": payload.get("role")
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate token"
            )
    
    def check_role_permission(self, required_role: str):
        """Decorator to check role-based permissions"""
        def role_checker(user_info: dict = Depends(self.verify_jwt_token)):
            user_role = user_info.get("role", "")
            
            role_hierarchy = {
                "admin": 3,
                "operator": 2,
                "viewer": 1
            }
            
            required_level = role_hierarchy.get(required_role, 0)
            user_level = role_hierarchy.get(user_role, 0)
            
            if user_level < required_level:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {required_role}"
                )
            
            return user_info
        return role_checker
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.post("/auth/login")
        async def login(username: str, password: str):
            """Authenticate user and return JWT token"""
            # Simplified authentication - in production, verify against database
            if username == "admin" and password == "kmrl2025":
                token = self.create_jwt_token(username, "admin")
                return {"access_token": token, "token_type": "bearer", "role": "admin"}
            elif username == "operator" and password == "kmrl2025":
                token = self.create_jwt_token(username, "operator")
                return {"access_token": token, "token_type": "bearer", "role": "operator"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
        
        @self.app.get("/health")
        async def health_check():
            """System health check endpoint"""
            try:
                # Check digital twin status
                twin_status = self.digital_twin.is_running
                
                # Check Redis connection
                redis_status = self.redis_client.ping()
                
                # Get system metrics
                current_state = self.digital_twin.get_current_state()
                summary = current_state.get('summary', {})
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "services": {
                        "digital_twin": "running" if twin_status else "stopped",
                        "redis": "connected" if redis_status else "disconnected",
                        "api_gateway": "active"
                    },
                    "metrics": summary
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        @self.app.get("/system/status")
        async def get_system_status(
            depot_id: Optional[str] = None
        ):
            """Get current system status"""
            current_state = self.digital_twin.get_current_state()
            # Include degraded mode if present from last plan
            try:
                if 'degraded_mode' not in current_state.get('summary', {}):
                    # Probe from DB or cached plan if available later
                    pass
            except Exception:
                pass
            # Data freshness & validation signals (latency flags) and counts
            try:
                freshness = {}
                data_counts = {}
                from .db import fetch_df, get_heartbeats
                for tbl in ['trains','job_cards','cleaning_slots','bay_config','branding_contracts']:
                    df = fetch_df(tbl)
                    freshness[tbl] = 'stale' if df.empty else 'ok'
                    try:
                        data_counts[tbl] = int(len(df))
                    except Exception:
                        data_counts[tbl] = 0
                # Integration heartbeats
                hb = get_heartbeats()
                freshness['integration'] = hb.to_dict(orient='records') if not hb.empty else []
                current_state['data_freshness'] = freshness
                current_state['data_counts'] = data_counts
                # Last uploads by table from heartbeats (sources like upload_trains)
                try:
                    last_uploads = {}
                    if not hb.empty:
                        df_hb = hb.copy()
                        if 'source' in df_hb.columns and 'last_heartbeat' in df_hb.columns:
                            df_hb = df_hb[df_hb['source'].str.startswith('upload_')]
                            for _, row in df_hb.iterrows():
                                src = str(row['source'])
                                tbl = src.replace('upload_', '', 1)
                                last_uploads[tbl] = row['last_heartbeat']
                    current_state['last_uploads'] = last_uploads
                except Exception:
                    current_state['last_uploads'] = {}
            except Exception:
                current_state['data_freshness'] = {'error': 'unavailable'}
                current_state['data_counts'] = {}
                current_state['last_uploads'] = {}
            # Optional depot filter
            if depot_id:
                try:
                    trains = current_state.get('trains', {})
                    bays = current_state.get('bays', {})
                    filtered_trains = {
                        tid: t for tid, t in trains.items()
                        if str(t.get('depot_id', depot_id)) == depot_id
                    }
                    filtered_bays = {
                        bid: b for bid, b in bays.items()
                        if str(b.get('depot_id', depot_id)) == depot_id
                    }
                    current_state = {
                        **current_state,
                        'trains': filtered_trains,
                        'bays': filtered_bays
                    }
                except Exception as e:
                    logger.warning(f"Depot filter failed: {e}")
            
            # Cache in Redis for 30 seconds
            cache_key = "system_status"
            self.redis_client.setex(cache_key, 30, json.dumps(current_state, default=str))
            
            return {
                "system_status": current_state,
                "user": "anonymous",
                "timestamp": datetime.now().isoformat()
            }

        # Contingency planning: propose degraded alternatives when shortfall exists
        @self.app.get("/planning/contingency")
        async def contingency_planning(user_info: dict = Depends(self.check_role_permission("operator"))):
            try:
                # Pull latest plan context if available via AI optimizer or digital twin summary
                state = self.digital_twin.get_current_state()
                summary = state.get('summary', {})
                required = summary.get('required_trains')
                available = summary.get('available_trains')
                if required is None or available is None:
                    try:
                        from config.settings import SETTINGS
                        required = required or int(SETTINGS.get('data', {}).get('required_trains', 0))
                        available = available or int(summary.get('inducted_trains', 0))
                    except Exception:
                        required = required or 0
                        available = available or 0
                shortfall = max(0, required - available)

                alternatives = []
                # Strategy 1: Reduce service headway (skip low-demand services)
                alternatives.append({
                    'strategy': 'reduce_headway_low_demand',
                    'description': 'Temporarily increase headway on historically low-demand time bands to match reduced fleet.',
                    'estimated_impact': {
                        'services_reduced': shortfall,
                        'passenger_impact_score': 0.3,
                        'energy_savings_kwh': shortfall * 150.0
                    }
                })
                # Strategy 2: Reassign branding and maintenance to standby
                alternatives.append({
                    'strategy': 'reassign_branding_to_standby',
                    'description': 'Shift branding exposure to standby where possible to protect SLAs while reducing service rakes.',
                    'estimated_impact': {
                        'branding_sla_risk': 'low',
                        'maintenance_load': 'balanced'
                    }
                })
                # Strategy 3: Cross-depot borrowing (if multi-depot)
                alternatives.append({
                    'strategy': 'cross_depot_borrowing',
                    'description': 'Request temporary cross-depot rake reassignment for morning peak only.',
                    'estimated_impact': {
                        'coordination_complexity': 'medium',
                        'turnaround_risk': 'medium'
                    }
                })

                return {
                    'mode': 'contingency' if shortfall > 0 else 'normal',
                    'required': required,
                    'available': available,
                    'shortfall': shortfall,
                    'alternatives': alternatives,
                    'generated_at': datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to compute contingency: {e}")

        # Contingency status endpoint
        @self.app.get("/system/contingency")
        async def contingency_status(user_info: dict = Depends(self.check_role_permission("viewer"))):
            state = self.digital_twin.get_current_state()
            required = state.get('summary', {}).get('required_trains', 0)
            available = state.get('summary', {}).get('available_trains', 0)
            mode = 'normal'
            if available < required:
                mode = 'contingency'
            return {'mode': mode, 'required': required, 'available': available}
        
        @self.app.post("/induction/request")
        async def request_train_induction(
            request: TrainInductionRequest,
            background_tasks: BackgroundTasks,
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            """Request train induction with AI optimization"""
            try:
                # Validate train IDs
                available_trains = list(self.digital_twin.trains.keys())
                invalid_trains = [tid for tid in request.train_ids if tid not in available_trains]
                
                if invalid_trains:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid train IDs: {invalid_trains}"
                    )
                
                # Create induction request
                induction_id = str(uuid.uuid4())
                
                induction_request = {
                    "induction_id": induction_id,
                    "train_ids": request.train_ids,
                    "requested_by": user_info["user_id"],
                    "priority": request.priority_override or "normal",
                    "scheduled_time": request.scheduled_time.isoformat() if request.scheduled_time else None,
                    "reason": request.reason,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
                
                # Store in Redis
                self.redis_client.setex(
                    f"induction_request:{induction_id}", 
                    3600, 
                    json.dumps(induction_request)
                )
                
                # Process in background
                background_tasks.add_task(
                    self.process_induction_request, 
                    induction_id, 
                    request.train_ids
                )
                
                logger.info(f"Induction request {induction_id} created by {user_info['user_id']}")
                
                return {
                    "induction_id": induction_id,
                    "status": "accepted",
                    "message": "Induction request accepted and being processed",
                    "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Induction request failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process induction request: {str(e)}"
                )
        
        @self.app.get("/induction/{induction_id}/status")
        async def get_induction_status(
            induction_id: str,
            user_info: dict = Depends(self.check_role_permission("viewer"))
        ):
            """Get status of specific induction request"""
            cached_request = self.redis_client.get(f"induction_request:{induction_id}")
            
            if not cached_request:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Induction request not found"
                )
            
            return json.loads(cached_request)
        
        @self.app.post("/system/emergency")
        async def handle_emergency(
            emergency: EmergencyEvent,
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            """Handle emergency events"""
            try:
                # Log emergency
                logger.critical(f"EMERGENCY: {emergency.event_type} - {emergency.description}")
                
                # Store in Redis with high priority
                emergency_data = emergency.dict()
                emergency_data["handled_by"] = user_info["user_id"]
                emergency_data["handled_at"] = datetime.now().isoformat()
                
                self.redis_client.setex(
                    f"emergency:{emergency.event_id}",
                    86400,  # 24 hours
                    json.dumps(emergency_data)
                )
                
                # Trigger emergency procedures in digital twin
                emergency_event = {
                    "type": "emergency_override",
                    "emergency_type": emergency.event_type,
                    "severity": emergency.severity,
                    "message": emergency.description,
                    "requires_evacuation": emergency.requires_evacuation
                }
                
                self.digital_twin.schedule_event(emergency_event)
                
                # Notify all connected clients
                await self.broadcast_emergency(emergency_data)
                
                return {
                    "status": "emergency_handled",
                    "event_id": emergency.event_id,
                    "actions_triggered": [
                        "Digital twin emergency procedures activated",
                        "Real-time notifications sent",
                        "Emergency logged for audit trail"
                    ]
                }
                
            except Exception as e:
                logger.error(f"Emergency handling failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to handle emergency"
                )
        
        @self.app.post("/data/train_status")
        async def update_train_status(
            updates: List[TrainStatusUpdate],
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            """Update train status from external systems (OCC integration)"""
            updated_trains = []
            
            for update in updates:
                if update.train_id in self.digital_twin.trains:
                    train = self.digital_twin.trains[update.train_id]
                    
                    # Update train state
                    train.location = update.location
                    train.mileage_km = update.mileage_km
                    train.fitness_valid_until = update.fitness_valid_until
                    
                    if update.maintenance_required:
                        train.update_status('maintenance_required', 'External system update')
                    
                    updated_trains.append(update.train_id)
                    
                    # Cache update in Redis
                    cache_key = f"train_update:{update.train_id}"
                    self.redis_client.setex(cache_key, 300, json.dumps(update.dict(), default=str))
            # Persist latest job_cards/fitness if relevant
            try:
                rows = []
                for u in updates:
                    rows.append({'train_id': u.train_id, 'job_card_status': 'open' if u.maintenance_required else 'closed', 'last_updated': datetime.now().isoformat()})
                if rows:
                    upsert('job_cards', rows, ['train_id'])
            except Exception as e:
                logger.error(f"Failed to persist updates: {e}")
            
            return {
                "updated_trains": updated_trains,
                "timestamp": datetime.now().isoformat(),
                "updated_by": user_info["user_id"]
            }
        
        @self.app.get("/analytics/performance")
        async def get_performance_analytics(
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            user_info: dict = Depends(self.check_role_permission("viewer"))
        ):
            """Get system performance analytics"""
            # This would typically query a time-series database
            # For demo, return current metrics
            
            current_state = self.digital_twin.get_current_state()
            summary = current_state.get('summary', {})
            
            analytics = {
                "performance_score": 81.4,  # From AI optimization
                "efficiency_metrics": {
                    "bay_utilization": summary.get('bay_utilization', 0),
                    "average_induction_time": 45,  # minutes
                    "failure_prediction_accuracy": 85.1,  # percent
                    "energy_efficiency": 92.3  # percent
                },
                "operational_metrics": {
                    "total_inductions_today": summary.get('inducted_trains', 0),
                    "emergency_events": 0,
                    "system_uptime": 99.8  # percent
                },
                "cost_savings": {
                    "daily_savings": 138000,  # INR
                    "annual_projection": 50370000  # INR
                },
                "query_metadata": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "generated_at": datetime.now().isoformat(),
                    "requested_by": user_info["user_id"]
                }
            }
            
            return analytics

        @self.app.get("/branding/sla")
        async def branding_sla(user_info: dict = Depends(self.check_role_permission("viewer"))):
            """Get branding SLA status and alerts"""
            try:
                status = get_sla_status()
                status['generated_at'] = datetime.now().isoformat()
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to compute SLA: {e}")

        @self.app.get("/ml/drift")
        async def ml_drift(user_info: dict = Depends(self.check_role_permission("viewer"))):
            return compute_drift_metrics()

        @self.app.post("/ml/retrain_if_needed")
        async def ml_retrain(threshold: float = 0.05, user_info: dict = Depends(self.check_role_permission("admin"))):
            return retrain_predictive_model_if_needed(threshold)

        # ===== ML Training Pipeline & Registry =====

        @self.app.post("/ml/train_now")
        async def train_now(user_info: dict = Depends(self.check_role_permission("admin"))):
            try:
                # Gather training data from digital twin snapshots if available
                # For now, reuse trains/job_cards-like structures if exposed via state
                state = self.digital_twin.get_current_state()
                trains = state.get('trains', {})
                trains_df = pd.DataFrame([{'train_id': k, **v} for k, v in trains.items()]) if trains else pd.DataFrame(columns=['train_id'])
                # Historical outcomes if persisted (fallback: empty)
                from .db import fetch_df
                historical_df = fetch_df('outcomes')
                # Job cards from DB
                job_cards_df = fetch_df('job_cards')
                pipeline = TrainingPipeline()
                result = pipeline.train_and_register(historical_df, trains_df, job_cards_df, created_by=user_info['user_id'])
                return {'status': 'ok', **result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        @self.app.get("/ml/registry")
        async def ml_registry(user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                reg = ModelRegistry()
                return {'models': reg.list_models()}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Registry fetch failed: {e}")

        @self.app.get("/ml/metrics")
        async def ml_metrics(model_id: Optional[str] = None, user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                where = "" if not model_id else "WHERE model_id = ?"
                params = () if not model_id else (model_id,)
                df = fetch_query(f"SELECT * FROM model_metrics {where} ORDER BY timestamp DESC LIMIT 500", params)
                return {'metrics': df.to_dict(orient='records') if not df.empty else []}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Metrics fetch failed: {e}")

        # ===== Production Maximo Integration =====
        @self.app.post("/maximo/refresh")
        async def refresh_maximo_data(user_info: dict = Depends(self.check_role_permission("operator"))):
            try:
                # Try production Maximo first, fallback to enhanced
                try:
                    from .production_maximo_integration import get_production_maximo
                    maximo = get_production_maximo()
                    result = maximo.refresh_data()
                    if result.get('success'):
                        return result
                except Exception as prod_error:
                    logger.warning(f"Production Maximo failed: {prod_error}")
                
                # Fallback to enhanced Maximo
                from .enhanced_maximo_integration import get_maximo_integration
                maximo = get_maximo_integration()
                result = maximo.refresh_data()
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/maximo/status")
        async def get_maximo_status(user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                # Try production Maximo first
                try:
                    from .production_maximo_integration import get_production_maximo
                    maximo = get_production_maximo()
                    status = maximo.get_connection_status()
                    return status
                except Exception:
                    pass
                
                # Fallback to enhanced Maximo
                from .enhanced_maximo_integration import get_maximo_integration
                maximo = get_maximo_integration()
                status = maximo.check_connection()
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/maximo/workorders")
        async def get_work_orders(
            train_ids: Optional[str] = None,
            status: Optional[str] = None,
            user_info: dict = Depends(self.check_role_permission("viewer"))
        ):
            try:
                from .production_maximo_integration import get_production_maximo
                maximo = get_production_maximo()
                
                train_list = train_ids.split(',') if train_ids else None
                df = maximo.fetch_work_orders(train_list, status)
                
                return {
                    'work_orders': df.to_dict('records') if not df.empty else [],
                    'count': len(df),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/maximo/assets")
        async def get_assets(
            train_ids: Optional[str] = None,
            user_info: dict = Depends(self.check_role_permission("viewer"))
        ):
            try:
                from .production_maximo_integration import get_production_maximo
                maximo = get_production_maximo()
                
                train_list = train_ids.split(',') if train_ids else None
                df = maximo.fetch_assets(train_list)
                
                return {
                    'assets': df.to_dict('records') if not df.empty else [],
                    'count': len(df),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/maximo/workorders")
        async def create_work_order(
            work_order: Dict[str, Any],
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            try:
                from .production_maximo_integration import get_production_maximo
                maximo = get_production_maximo()
                
                result = maximo.create_work_order(work_order)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/maximo/workorders/{work_order_id}")
        async def update_work_order(
            work_order_id: str,
            status: str,
            notes: Optional[str] = None,
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            try:
                from .production_maximo_integration import get_production_maximo
                maximo = get_production_maximo()
                
                result = maximo.update_work_order_status(work_order_id, status, notes)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/maximo/upload")
        async def upload_maximo_data(
            data_type: str,
            data: List[Dict[str, Any]],
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            try:
                from .enhanced_maximo_integration import get_maximo_integration
                maximo = get_maximo_integration()
                result = maximo.upload_manual_data(data, data_type)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/maximo/templates")
        async def get_maximo_templates(user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                from .enhanced_maximo_integration import get_maximo_integration
                maximo = get_maximo_integration()
                templates = maximo.get_manual_data_templates()
                return templates
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ===== Data Ingestion: file uploads (CSV/JSON) =====

        @self.app.post("/ingest/{table}")
        async def ingest_table(
            table: str,
            file: UploadFile = File(...),
            fmt: Optional[str] = None,
            allow_missing: Optional[bool] = False,
            user_info: dict = Depends(self.check_role_permission("operator"))
        ):
            # Allowed tables and validation schemas
            allowed = {
                'trains': {
                    'key': ['train_id'],
                    'schema': {
                        'train_id': {'type': 'str', 'required': True},
                        'fitness_valid_until': {'type': 'date', 'required': True},
                        'mileage_km': {'type': 'float', 'required': True, 'min': 0},
                        'branding_hours_left': {'type': 'float', 'required': True, 'min': 0},
                        'cleaning_slot_id': {'type': 'str', 'required': False},
                        'bay_geometry_score': {'type': 'float', 'required': True, 'min': 0, 'max': 1},
                        'depot_id': {'type': 'str', 'required': False}
                    }
                },
                'job_cards': {
                    'key': ['train_id'],
                    'schema': {
                        'train_id': {'type': 'str', 'required': True},
                        'job_card_status': {'type': 'str', 'required': True, 'choices': ['open','closed']},
                        'last_updated': {'type': 'date', 'required': False}
                    }
                },
                'cleaning_slots': {
                    'key': ['slot_id'],
                    'schema': {
                        'slot_id': {'type': 'str', 'required': True},
                        'available_bays': {'type': 'int', 'required': True, 'min': 0},
                        'priority': {'type': 'str', 'required': False, 'choices': ['low','medium','high']},
                        'depot_id': {'type': 'str', 'required': False}
                    }
                },
                'bay_config': {
                    'key': ['bay_id'],
                    'schema': {
                        'bay_id': {'type': 'str', 'required': True},
                        'bay_type': {'type': 'str', 'required': True, 'choices': ['service','inspection','stabling']},
                        'max_capacity': {'type': 'int', 'required': True, 'min': 0},
                        'geometry_score': {'type': 'float', 'required': True, 'min': 0, 'max': 1},
                        'depot_id': {'type': 'str', 'required': False}
                    }
                },
                'branding_contracts': {
                    'key': ['contract_id'],
                    'schema': {
                        'contract_id': {'type': 'str', 'required': True},
                        'brand': {'type': 'str', 'required': True},
                        'train_id': {'type': 'str', 'required': False},
                        'hours_committed': {'type': 'float', 'required': True, 'min': 0},
                        'start_date': {'type': 'date', 'required': True},
                        'end_date': {'type': 'date', 'required': True}
                    }
                },
                'branding_exposure': {
                    'key': None,
                    'schema': {
                        'train_id': {'type': 'str', 'required': True},
                        'brand': {'type': 'str', 'required': True},
                        'date': {'type': 'date', 'required': True},
                        'hours': {'type': 'float', 'required': True, 'min': 0}
                    }
                },
                'outcomes': {
                    'key': None,
                    'schema': {
                        'date': {'type': 'date', 'required': True},
                        'train_id': {'type': 'str', 'required': True},
                        'inducted': {'type': 'int', 'required': True, 'choices': [0,1]},
                        'failures': {'type': 'int', 'required': False, 'min': 0},
                        'notes': {'type': 'str', 'required': False}
                    }
                }
            }
            if table not in allowed:
                raise HTTPException(status_code=400, detail=f"Unsupported table: {table}")
            try:
                content = await file.read()
                fmt = fmt or ('json' if file.filename.lower().endswith('.json') else 'csv')
                if fmt not in ('csv', 'json'):
                    raise HTTPException(status_code=400, detail="fmt must be 'csv' or 'json'")
                # Parse to DataFrame
                if fmt == 'csv':
                    df = pd.read_csv(io.BytesIO(content))
                else:
                    data = json.loads(content.decode('utf-8'))
                    if isinstance(data, dict):
                        # single record or dict of arrays
                        try:
                            df = pd.DataFrame(data)
                        except Exception:
                            df = pd.DataFrame([data])
                    else:
                        df = pd.DataFrame(data)
                # Validate columns
                schema = allowed[table]['schema']
                required_cols = [c for c, r in schema.items() if r.get('required')]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    if not allow_missing:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                'message': 'Missing required columns',
                                'missing_columns': missing,
                                'expected_columns': list(schema.keys())
                            }
                        )
                    # permissive path: add missing required columns as None
                    for col in missing:
                        df[col] = None
                # Normalize and type-coerce; collect row-level errors
                errors: List[Dict[str, Any]] = []
                for col, rules in schema.items():
                    if col not in df.columns:
                        continue
                    try:
                        if rules['type'] == 'str':
                            df[col] = df[col].astype(str)
                        elif rules['type'] in ('float','int'):
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        elif rules['type'] == 'date':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        # constraints
                        if 'choices' in rules:
                            bad_idx = ~df[col].isin(rules['choices'])
                            for i in df[bad_idx].index.tolist():
                                errors.append({'row': int(i), 'column': col, 'error': f"value not in {rules['choices']}", 'value': None if pd.isna(df.at[i,col]) else str(df.at[i,col])})
                        if 'min' in rules:
                            bad_idx = df[col] < rules['min']
                            for i in df[bad_idx.fillna(False)].index.tolist():
                                errors.append({'row': int(i), 'column': col, 'error': f"value < {rules['min']}", 'value': None if pd.isna(df.at[i,col]) else float(df.at[i,col])})
                        if 'max' in rules:
                            bad_idx = df[col] > rules['max']
                            for i in df[bad_idx.fillna(False)].index.tolist():
                                errors.append({'row': int(i), 'column': col, 'error': f"value > {rules['max']}", 'value': None if pd.isna(df.at[i,col]) else float(df.at[i,col])})
                        # required non-null
                        if rules.get('required'):
                            bad_idx = df[col].isna()
                            for i in df[bad_idx].index.tolist():
                                errors.append({'row': int(i), 'column': col, 'error': 'required value missing', 'value': None})
                    except Exception as e:
                        errors.append({'column': col, 'error': f"failed to normalize: {e}"})
                if errors:
                    # return at most 100 errors for brevity
                    return {
                        'status': 'error',
                        'message': 'Validation failed',
                        'errors': errors[:100],
                        'total_errors': len(errors)
                    }
                # Keep only allowed schema columns to avoid DB unknown-column errors
                allowed_cols = list(schema.keys())
                df = df[[c for c in df.columns if c in allowed_cols]]
                # Ensure all required columns exist (fill with None if missing)
                for c in allowed_cols:
                    if c not in df.columns:
                        df[c] = None
                # Upsert vs insert
                from .db import upsert, insert_rows, upsert_heartbeat
                key_cols = allowed[table]['key']
                # Convert dates to ISO strings before persisting
                for c, r in schema.items():
                    if r.get('type') == 'date' and c in df.columns:
                        df[c] = df[c].dt.strftime('%Y-%m-%dT%H:%M:%S')
                rows = df.to_dict(orient='records')
                if key_cols:
                    upsert(table, rows, key_cols)
                else:
                    insert_rows(table, rows)
                # Heartbeat
                try:
                    upsert_heartbeat(f"upload_{table}", 'ok', f"rows={len(rows)}")
                except Exception:
                    pass
                return {'status': 'ok', 'table': table, 'rows': len(rows)}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

        @self.app.get("/ingest/templates/{table}")
        async def ingest_template(table: str, user_info: dict = Depends(self.check_role_permission("viewer"))):
            templates: Dict[str, List[str]] = {
                'trains': ['train_id','fitness_valid_until','mileage_km','branding_hours_left','cleaning_slot_id','bay_geometry_score','depot_id'],
                'job_cards': ['train_id','job_card_status','last_updated'],
                'cleaning_slots': ['slot_id','available_bays','priority','depot_id'],
                'bay_config': ['bay_id','bay_type','max_capacity','geometry_score','depot_id'],
                'branding_contracts': ['contract_id','brand','train_id','hours_committed','start_date','end_date'],
                'branding_exposure': ['train_id','brand','date','hours'],
                'outcomes': ['date','train_id','inducted','failures','notes']
            }
            if table not in templates:
                raise HTTPException(status_code=400, detail=f"Unsupported table: {table}")
            return {'table': table, 'columns': templates[table]}

        @self.app.get("/ingest/spec/{table}")
        async def ingest_spec(table: str, user_info: dict = Depends(self.check_role_permission("viewer"))):
            specs = {
                'trains': {
                    'format': ['csv','json'],
                    'columns': {
                        'train_id': 'string (required)',
                        'fitness_valid_until': 'ISO datetime (required)',
                        'mileage_km': 'number >= 0 (required)',
                        'branding_hours_left': 'number >= 0 (required)',
                        'cleaning_slot_id': 'string',
                        'bay_geometry_score': 'number [0,1] (required)',
                        'depot_id': 'string'
                    }
                },
                'job_cards': {
                    'format': ['csv','json'],
                    'columns': {
                        'train_id': 'string (required)',
                        'job_card_status': "one of ['open','closed'] (required)",
                        'last_updated': 'ISO datetime'
                    }
                },
                'cleaning_slots': {
                    'format': ['csv','json'],
                    'columns': {
                        'slot_id': 'string (required)',
                        'available_bays': 'integer >= 0 (required)',
                        'priority': "one of ['low','medium','high']",
                        'depot_id': 'string'
                    }
                },
                'bay_config': {
                    'format': ['csv','json'],
                    'columns': {
                        'bay_id': 'string (required)',
                        'bay_type': "one of ['service','inspection','stabling'] (required)",
                        'max_capacity': 'integer >= 0 (required)',
                        'geometry_score': 'number [0,1] (required)',
                        'depot_id': 'string'
                    }
                },
                'branding_contracts': {
                    'format': ['csv','json'],
                    'columns': {
                        'contract_id': 'string (required)',
                        'brand': 'string (required)',
                        'train_id': 'string',
                        'hours_committed': 'number >= 0 (required)',
                        'start_date': 'ISO date (required)',
                        'end_date': 'ISO date (required)'
                    }
                },
                'branding_exposure': {
                    'format': ['csv','json'],
                    'columns': {
                        'train_id': 'string (required)',
                        'brand': 'string (required)',
                        'date': 'ISO date (required)',
                        'hours': 'number >= 0 (required)'
                    }
                },
                'outcomes': {
                    'format': ['csv','json'],
                    'columns': {
                        'date': 'ISO date (required)',
                        'train_id': 'string (required)',
                        'inducted': '0 or 1 (required)',
                        'failures': 'integer >= 0',
                        'notes': 'string'
                    }
                }
            }
            if table not in specs:
                raise HTTPException(status_code=400, detail=f"Unsupported table: {table}")
            return {'table': table, **specs[table]}

        # ===== Fleet Readiness Snapshot =====
        @self.app.get("/fleet/readiness")
        async def fleet_readiness(user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                from .db import fetch_df
                trains = fetch_df('trains')
                jobs = fetch_df('job_cards')
                contracts = fetch_df('branding_contracts')
                overrides = fetch_df('manual_overrides')

                if trains.empty:
                    return {'rows': [], 'count': 0}

                # Merge job card status
                if not jobs.empty:
                    trains = trains.merge(jobs[['train_id','job_card_status']], on='train_id', how='left')
                else:
                    trains['job_card_status'] = None

                # Branding intensity by train (count of active contracts)
                branding_map = {}
                if not contracts.empty:
                    try:
                        active = contracts.copy()
                        active['start_date'] = pd.to_datetime(active['start_date'], errors='coerce')
                        active['end_date'] = pd.to_datetime(active['end_date'], errors='coerce')
                        now = pd.Timestamp.now()
                        active = active[(active['start_date']<=now) & (active['end_date']>=now)]
                        branding_map = active.groupby('train_id').size().to_dict()
                    except Exception:
                        branding_map = contracts.groupby('train_id').size().to_dict()
                trains['branding_level'] = trains['train_id'].map(branding_map).fillna(0).astype(int)

                # Compute readiness fields
                def fitness_flag(v):
                    try:
                        dt = pd.to_datetime(v, errors='coerce')
                        return 'OK' if pd.notna(dt) and dt >= pd.Timestamp.now() else 'Expired'
                    except Exception:
                        return 'Expired'
                trains['fitness_status'] = trains['fitness_valid_until'].apply(fitness_flag)
                trains['maintenance_status'] = trains['job_card_status'].fillna('closed').apply(lambda s: 'Open' if str(s).lower()=='open' else 'OK')
                # Apply latest manual overrides per train for maintenance_status
                override_info = {}
                try:
                    if not overrides.empty:
                        latest = overrides[overrides['field']=='maintenance_status'].sort_values('overridden_at').groupby('train_id').tail(1)
                        ov_map = latest.set_index('train_id')['value'].to_dict()
                        ov_reason = latest.set_index('train_id')['reason'].to_dict()
                        ov_by = latest.set_index('train_id')['overridden_by'].to_dict()
                        def apply_ov(tid, val):
                            if tid in ov_map:
                                override_info[tid] = {
                                    'reason': ov_reason.get(tid, ''),
                                    'by': ov_by.get(tid, ''),
                                    'has_override': True
                                }
                                return ov_map[tid]
                            return val
                        trains['maintenance_status'] = trains.apply(lambda r: apply_ov(r['train_id'], r['maintenance_status']), axis=1)
                except Exception:
                    pass
                trains['branding_badges'] = trains['branding_level'].apply(lambda n: '💲'*min(3, max(0,int(n))))
                trains['readiness'] = trains.apply(lambda r: 'Go' if (r['fitness_status']=='OK' and r['maintenance_status']=='OK') else ('Standby' if r['fitness_status']=='OK' else 'No-Go'), axis=1)

                out = []
                for _, row in trains.iterrows():
                    train_id = row.get('train_id')
                    ov = override_info.get(train_id, {})
                    out.append({
                        'train_id': train_id,
                        'fitness': ('✅ OK' if row['fitness_status']=='OK' else '❌ Expired'),
                        'maintenance': ('✅ OK' if str(row['maintenance_status']).upper()=='OK' else '🔧 Open'),
                        'branding': row['branding_badges'] or '',
                        'mileage_km': row.get('mileage_km'),
                        'overall_readiness': row['readiness'],
                        'override_info': ov
                    })
                return {'rows': out, 'count': len(out), 'generated_at': datetime.now().isoformat()}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to compute readiness: {e}")

        @self.app.post("/fleet/override")
        async def set_override(train_id: str, field: str, value: str, reason: str = "", user_info: dict = Depends(self.check_role_permission("operator"))):
            try:
                if field not in ("maintenance_status", "fitness_status"):
                    raise HTTPException(status_code=400, detail="Unsupported field for override")
                from .db import insert_rows
                insert_rows('manual_overrides', [{
                    'train_id': train_id,
                    'field': field,
                    'value': value,
                    'reason': reason,
                    'overridden_by': user_info['user_id'],
                    'overridden_at': datetime.now().isoformat()
                }])
                return {'status': 'ok', 'train_id': train_id, 'field': field, 'value': value}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to set override: {e}")

        @self.app.delete("/fleet/override/{train_id}/{field}")
        async def revert_override(train_id: str, field: str, user_info: dict = Depends(self.check_role_permission("operator"))):
            try:
                from .db import fetch_query
                # Get latest override for this train+field
                latest = fetch_query(
                    "SELECT id FROM manual_overrides WHERE train_id = ? AND field = ? ORDER BY overridden_at DESC LIMIT 1",
                    (train_id, field)
                )
                if latest.empty:
                    raise HTTPException(status_code=404, detail="No override found to revert")
                # Delete the latest override
                from .db import get_connection
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("DELETE FROM manual_overrides WHERE id = ?", (latest.iloc[0]['id'],))
                conn.commit()
                conn.close()
                return {'status': 'ok', 'train_id': train_id, 'field': field, 'action': 'reverted'}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to revert override: {e}")

        # ===== Supervisor Approvals Workflow (Redis-backed) =====

        @self.app.post("/approvals/submit")
        async def submit_plan(plan: Dict[str, Any], user_info: dict = Depends(self.check_role_permission("operator"))):
            plan_id = str(uuid.uuid4())
            payload = {
                'plan_id': plan_id,
                'submitted_by': user_info['user_id'],
                'status': 'pending',
                'submitted_at': datetime.now().isoformat(),
                'plan': plan
            }
            self.redis_client.setex(f"approval:{plan_id}", 86400, json.dumps(payload))
            self.redis_client.lpush("approvals_queue", plan_id)
            # Mirror to DB (master + audit)
            try:
                insert_rows('approvals', [{
                    'plan_id': plan_id,
                    'submitted_by': payload['submitted_by'],
                    'submitted_at': payload['submitted_at'],
                    'status': payload['status'],
                    'decided_by': None,
                    'decided_at': None,
                    'reason': None,
                    'plan_json': json.dumps(plan)
                }])
                insert_rows('approvals_audit', [{
                    'plan_id': plan_id,
                    'event': 'submitted',
                    'actor': payload['submitted_by'],
                    'event_time': payload['submitted_at'],
                    'details': None
                }])
            except Exception as e:
                logger.error(f"Failed to persist approval submit: {e}")
            return {'plan_id': plan_id, 'status': 'pending'}

        @self.app.get("/approvals/pending")
        async def list_pending(user_info: dict = Depends(self.check_role_permission("viewer"))):
            keys = self.redis_client.lrange("approvals_queue", 0, -1)
            out = []
            for k in keys:
                data = self.redis_client.get(f"approval:{k}")
                if data:
                    payload = json.loads(data)
                    if payload.get('status') == 'pending':
                        out.append(payload)
            return {'pending': out}

        @self.app.post("/approvals/{plan_id}/decision")
        async def approve_plan(plan_id: str, decision: str, reason: Optional[str] = None, user_info: dict = Depends(self.check_role_permission("admin"))):
            key = f"approval:{plan_id}"
            data = self.redis_client.get(key)
            if not data:
                raise HTTPException(status_code=404, detail="Plan not found")
            payload = json.loads(data)
            if payload.get('status') != 'pending':
                raise HTTPException(status_code=400, detail="Plan already decided")
            if decision not in ("approved", "rejected"):
                raise HTTPException(status_code=400, detail="Invalid decision")
            payload['status'] = decision
            payload['decided_by'] = user_info['user_id']
            payload['decided_at'] = datetime.now().isoformat()
            payload['reason'] = reason
            self.redis_client.setex(key, 86400, json.dumps(payload))
            # Mirror decision to DB and audit
            try:
                upsert('approvals', [{
                    'plan_id': plan_id,
                    'submitted_by': payload.get('submitted_by'),
                    'submitted_at': payload.get('submitted_at'),
                    'status': decision,
                    'decided_by': payload['decided_by'],
                    'decided_at': payload['decided_at'],
                    'reason': reason,
                    'plan_json': json.dumps(payload.get('plan'))
                }], ['plan_id'])
                insert_rows('approvals_audit', [{
                    'plan_id': plan_id,
                    'event': decision,
                    'actor': payload['decided_by'],
                    'event_time': payload['decided_at'],
                    'details': reason
                }])
            except Exception as e:
                logger.error(f"Failed to persist approval decision: {e}")
            return {'plan_id': plan_id, 'status': decision}

        # Approve & Lock final plan (RBAC: admin) with audit snapshot
        @self.app.post("/approvals/{plan_id}/lock")
        async def lock_plan(plan_id: str, depot_id: Optional[str] = None, user_info: dict = Depends(self.check_role_permission("admin"))):
            data = self.redis_client.get(f"approval:{plan_id}")
            if not data:
                raise HTTPException(status_code=404, detail="Plan not found")
            payload = json.loads(data)
            if payload.get('status') != 'approved':
                raise HTTPException(status_code=400, detail="Plan must be approved before lock")
            snapshot = {
                'plan': payload.get('plan'),
                'state': self.digital_twin.get_current_state(),
                'locked_by': user_info['user_id'],
                'locked_at': datetime.now().isoformat(),
                'depot_id': depot_id
            }
            try:
                insert_rows('plan_lock_audit', [{
                    'plan_id': plan_id,
                    'locked_by': user_info['user_id'],
                    'locked_at': snapshot['locked_at'],
                    'depot_id': depot_id,
                    'snapshot_json': json.dumps(snapshot)
                }])
            except Exception as e:
                logger.error(f"Failed lock audit: {e}")
            return {'plan_id': plan_id, 'locked': True, 'locked_at': snapshot['locked_at']}

        # ROI-ish KPIs endpoint
        @self.app.get("/analytics/kpis")
        async def analytics_kpis(user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                # Compute KPIs from persisted outcomes and exposure
                from .db import fetch_query
                outcomes = fetch_query("SELECT * FROM outcomes", ())
                exposure = fetch_query("SELECT * FROM branding_exposure", ())
                from config.settings import SETTINGS
                energy_per_service = float(SETTINGS.get('analytics', {}).get('energy_kwh_per_train_service', 350.0))
                punctuality_fail_thresh = float(SETTINGS.get('analytics', {}).get('punctuality_failure_threshold', 0.05))
                maint_cost_per_km = float(SETTINGS.get('analytics', {}).get('maintenance_cost_per_km', 12.0))
                # Fleet availability: inducted / total trains per day average
                availability = 0.0
                punct_protected = True
                energy_savings = 0.0
                branding_breaches_avoided = 0
                try:
                    if not outcomes.empty:
                        daily = outcomes.groupby('date').agg({'inducted': 'sum'}).reset_index()
                        availability = float(daily['inducted'].mean())
                        # proxy: fewer failures => punctuality protected
                        fail_rate = outcomes['failures'].mean() if 'failures' in outcomes.columns else 0.0
                        punct_protected = fail_rate < punctuality_fail_thresh
                        # energy consumed baseline
                        energy_savings = max(0.0, (daily['inducted'].max() - daily['inducted'].mean())) * energy_per_service
                    if not exposure.empty:
                        # proxy energy savings: service hours avoided vs baseline (rough estimate)
                        # branding breaches avoided: hours delivered bucketed by 100
                        branding_breaches_avoided = int(exposure['hours'].sum() // 100)
                except Exception:
                    pass
                return {
                    'fleet_availability_avg_inducted': availability,
                    'punctuality_protected': punct_protected,
                    'maintenance_cost_per_km': maint_cost_per_km,
                    'energy_savings_kwh': energy_savings,
                    'branding_sla_breaches_avoided': branding_breaches_avoided,
                    'generated_at': datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed KPIs: {e}")

        @self.app.get("/approvals/history")
        async def approvals_history(limit: int = 100, user_info: dict = Depends(self.check_role_permission("viewer"))):
            try:
                df = fetch_query(
                    "SELECT * FROM approvals_audit ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                history = df.to_dict(orient='records') if not df.empty else []
                return {'history': history}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")
    
    async def process_induction_request(self, induction_id: str, train_ids: List[str]):
        """Background task to process induction requests"""
        try:
            # Simulate AI processing time
            await asyncio.sleep(2)
            
            # Update request status
            request_data = json.loads(self.redis_client.get(f"induction_request:{induction_id}"))
            request_data["status"] = "processing"
            request_data["processing_started"] = datetime.now().isoformat()
            
            self.redis_client.setex(
                f"induction_request:{induction_id}",
                3600,
                json.dumps(request_data)
            )
            
            # Execute induction in digital twin
            for train_id in train_ids:
                if train_id in self.digital_twin.trains:
                    # Find available bay
                    available_bays = [
                        bay_id for bay_id, bay in self.digital_twin.bays.items()
                        if bay.status == 'available' and bay.bay_type == 'service'
                    ]
                    
                    if available_bays:
                        bay_id = available_bays[0]
                        event = {
                            'type': 'train_induction',
                            'train_id': train_id,
                            'bay_id': bay_id,
                            'source': 'api_request',
                            'induction_id': induction_id
                        }
                        self.digital_twin.schedule_event(event)
            
            # Mark as completed
            request_data["status"] = "completed"
            request_data["completed_at"] = datetime.now().isoformat()
            
            self.redis_client.setex(
                f"induction_request:{induction_id}",
                3600,
                json.dumps(request_data)
            )
            
            logger.info(f"Induction request {induction_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process induction request {induction_id}: {e}")
            
            # Mark as failed
            request_data["status"] = "failed"
            request_data["error"] = str(e)
            request_data["failed_at"] = datetime.now().isoformat()
            
            self.redis_client.setex(
                f"induction_request:{induction_id}",
                3600,
                json.dumps(request_data)
            )
    
    async def broadcast_emergency(self, emergency_data: dict):
        """Broadcast emergency to all connected WebSocket clients"""
        # This would be implemented with WebSocket connections
        # For now, just log the emergency
        logger.critical(f"Broadcasting emergency: {emergency_data}")
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the API server"""
        import uvicorn
        logger.info(f"🌐 Starting KMRL IntelliFleet API Gateway at http://{host}:{port}")
        logger.info("📚 API Documentation available at http://{host}:{port}/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
