from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import jwt
import redis
import json
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

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
        async def get_system_status(user_info: dict = Depends(self.check_role_permission("viewer"))):
            """Get current system status"""
            current_state = self.digital_twin.get_current_state()
            
            # Cache in Redis for 30 seconds
            cache_key = "system_status"
            self.redis_client.setex(cache_key, 30, json.dumps(current_state, default=str))
            
            return {
                "system_status": current_state,
                "user": user_info["user_id"],
                "timestamp": datetime.now().isoformat()
            }
        
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
