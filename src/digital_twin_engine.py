"""
KMRL IntelliFleet Digital Twin Engine
Complete implementation with dict-based initialization
"""

import queue
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import json


class TrainState:
    """Represents the current state of a train in the digital twin"""
    
    def __init__(self, train_id: str, initial_data: Dict[str, Any]):
        self.train_id = train_id
        self.location = initial_data.get('location', 'depot')
        self.assigned_bay = initial_data.get('assigned_bay', None)
        self.status = initial_data.get('status', 'idle')  # idle, moving, maintenance, service, cleaning
        self.mileage_km = initial_data.get('mileage_km', 0)
        self.branding_hours_left = initial_data.get('branding_hours_left', 100)
        
        # Handle fitness_valid_until as string or datetime
        fitness_date = initial_data.get('fitness_valid_until')
        if isinstance(fitness_date, str):
            self.fitness_valid_until = datetime.fromisoformat(fitness_date)
        elif isinstance(fitness_date, datetime):
            self.fitness_valid_until = fitness_date
        else:
            self.fitness_valid_until = datetime.now() + timedelta(days=365)
            
        self.cleaning_slot_id = initial_data.get('cleaning_slot_id', None)
        self.bay_geometry_score = initial_data.get('bay_geometry_score', 5)
        self.failure_probability = initial_data.get('failure_probability', 0.1)
        self.last_updated = datetime.now()
        self.event_history: List[Dict[str, Any]] = []
        
    def update_status(self, new_status: str, reason: str = ""):
        """Update train status with timestamp"""
        old_status = self.status
        self.status = new_status
        self.last_updated = datetime.now()
        
        event = {
            'timestamp': self.last_updated.isoformat(),
            'event_type': 'status_change',
            'old_status': old_status,
            'new_status': new_status,
            'reason': reason
        }
        self.event_history.append(event)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert train state to dictionary for JSON serialization"""
        return {
            'train_id': self.train_id,
            'location': self.location,
            'assigned_bay': self.assigned_bay,
            'status': self.status,
            'mileage_km': self.mileage_km,
            'branding_hours_left': self.branding_hours_left,
            'fitness_valid_until': self.fitness_valid_until.isoformat() if self.fitness_valid_until else None,
            'cleaning_slot_id': self.cleaning_slot_id,
            'bay_geometry_score': self.bay_geometry_score,
            'failure_probability': self.failure_probability,
            'last_updated': self.last_updated.isoformat(),
            'recent_events': self.event_history[-5:]  # Last 5 events
        }


class BayState:
    """Represents the current state of a bay in the digital twin"""
    
    def __init__(self, bay_id: str, bay_config: Dict[str, Any]):
        self.bay_id = bay_id
        self.bay_type = bay_config.get('bay_type', 'service')
        self.max_capacity = bay_config.get('max_capacity', 1)
        self.geometry_score = bay_config.get('geometry_score', 5)
        self.power_available = bay_config.get('power_available', True)
        self.occupied_trains: List[str] = []
        self.scheduled_trains: List[str] = []
        self.status = bay_config.get('status', 'available')  # available, occupied, maintenance, blocked
        self.last_updated = datetime.now()
        
    def assign_train(self, train_id: str) -> bool:
        """Assign a train to this bay"""
        if len(self.occupied_trains) < self.max_capacity:
            self.occupied_trains.append(train_id)
            self.status = 'occupied' if len(self.occupied_trains) == self.max_capacity else 'partial'
            self.last_updated = datetime.now()
            return True
        return False
    
    def remove_train(self, train_id: str) -> bool:
        """Remove a train from this bay"""
        if train_id in self.occupied_trains:
            self.occupied_trains.remove(train_id)
            self.status = 'occupied' if len(self.occupied_trains) == self.max_capacity else \
                         'partial' if len(self.occupied_trains) > 0 else 'available'
            self.last_updated = datetime.now()
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bay state to dictionary"""
        capacity = self.max_capacity if self.max_capacity > 0 else 1
        return {
            'bay_id': self.bay_id,
            'bay_type': self.bay_type,
            'max_capacity': self.max_capacity,
            'geometry_score': self.geometry_score,
            'power_available': self.power_available,
            'occupied_trains': self.occupied_trains,
            'scheduled_trains': self.scheduled_trains,
            'status': self.status,
            'utilization_rate': len(self.occupied_trains) / capacity,
            'last_updated': self.last_updated.isoformat()
        }


class DigitalTwinEngine:
    """Main digital twin simulation engine"""
    
    def __init__(self, initial_data: Dict[str, Any]):
        self.trains: Dict[str, TrainState] = {}
        self.bays: Dict[str, BayState] = {}
        self.simulation_time = datetime.now()
        self.time_multiplier = 1.0  # 1x real time
        self.is_running = False
        self.event_queue = queue.Queue()
        self.observers: List = []  # For real-time updates
        self.simulation_thread: Optional[threading.Thread] = None
        
        # Initialize from data dictionaries
        self._initialize_trains(initial_data.get('trains', {}))
        self._initialize_bays(initial_data.get('bay_config', {}))
        
        # Initialize scenario manager after engine setup
        self.scenario_manager = ScenarioManager(self)
    def _inject_ai_data_into_digital_twin(self):
        ai_state = {
            'ai_summary': self.ai_data_processor.get_train_status_summary(),
            'ai_train_details': self.ai_data_processor.get_detailed_train_list(),
            'ai_performance': self.ai_data_processor.get_performance_metrics(),
            'ai_violations': self.ai_data_processor.get_constraint_violations(),
            'last_updated': datetime.now().isoformat()
        }
        
        current_state = self.digital_twin.get_current_state()
        current_state['ai_data'] = ai_state
   
    def _initialize_trains(self, trains_data: Dict[str, Dict[str, Any]]):
        """Initialize train states from dictionary"""
        for train_id, train_data in trains_data.items():
            train_state = TrainState(train_id, train_data)
            self.trains[train_id] = train_state
        
        print(f"🚂 Initialized {len(self.trains)} trains in digital twin")
    
    def _initialize_bays(self, bay_data: Dict[str, Dict[str, Any]]):
        """Initialize bay states from dictionary"""
        for bay_id, bay_config in bay_data.items():
            bay_state = BayState(bay_id, bay_config)
            self.bays[bay_id] = bay_state
        
        print(f"🏗️ Initialized {len(self.bays)} bays in digital twin")
    
    def start_simulation(self, time_multiplier: float = 1.0):
        """Start the real-time simulation"""
        if self.is_running:
            return
            
        self.time_multiplier = time_multiplier
        self.is_running = True
        
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        print(f"🎬 Digital twin simulation started ({time_multiplier}x speed)")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        print("⏹️ Digital twin simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread"""
        while self.is_running:
            try:
                # Advance simulation time
                self.simulation_time += timedelta(seconds=30 * self.time_multiplier)
                
                # Process scheduled events
                self._process_events()
                
                # Update train states
                self._update_train_states()
                
                # Notify observers
                self._notify_observers()
                
                # Sleep based on time multiplier
                time.sleep(1.0 / self.time_multiplier)
            except Exception as e:
                print(f"Simulation loop error: {e}")
                time.sleep(5)
    
    def _process_events(self):
        """Process events from the event queue"""
        processed = 0
        while not self.event_queue.empty() and processed < 10:  # Limit processing per cycle
            try:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
                processed += 1
            except queue.Empty:
                break
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle individual simulation events"""
        event_type = event.get('type')
        
        try:
            if event_type == 'train_induction':
                self._handle_train_induction(event)
            elif event_type == 'train_completion':
                self._handle_train_completion(event)
            elif event_type == 'emergency_override':
                self._handle_emergency_override(event)
            elif event_type == 'bay_maintenance':
                self._handle_bay_maintenance(event)
        except Exception as e:
            print(f"Error handling event {event_type}: {e}")
    
    def _handle_train_induction(self, event: Dict[str, Any]):
        """Handle train induction event"""
        train_id = event.get('train_id')
        bay_id = event.get('bay_id')
        
        if train_id in self.trains and bay_id in self.bays:
            train = self.trains[train_id]
            bay = self.bays[bay_id]
            
            if bay.assign_train(train_id):
                train.assigned_bay = bay_id
                train.update_status('service', f'Inducted to {bay_id}')
                print(f"🚂 Train {train_id} inducted to {bay_id}")
    
    def _handle_train_completion(self, event: Dict[str, Any]):
        """Handle train service completion"""
        train_id = event.get('train_id')
        
        if train_id in self.trains:
            train = self.trains[train_id]
            if train.assigned_bay and train.assigned_bay in self.bays:
                bay = self.bays[train.assigned_bay]
                bay.remove_train(train_id)
                train.assigned_bay = None
                train.update_status('idle', 'Service completed')
                print(f"✅ Train {train_id} service completed")
    
    def _handle_emergency_override(self, event: Dict[str, Any]):
        """Handle emergency override events"""
        message = event.get('message', 'Unknown emergency')
        print(f"🚨 Emergency override: {message}")
        # Additional emergency handling logic can be added here
    
    def _handle_bay_maintenance(self, event: Dict[str, Any]):
        """Handle bay maintenance events"""
        bay_id = event.get('bay_id')
        if bay_id in self.bays:
            bay = self.bays[bay_id]
            bay.status = 'maintenance'
            print(f"🔧 Bay {bay_id} under maintenance")
    
    def _update_train_states(self):
        """Update train states based on simulation time"""
        for train in self.trains.values():
            # Simulate gradual changes
            if train.status == 'service':
                # Reduce branding hours during service
                train.branding_hours_left = max(0, train.branding_hours_left - 0.1)
            
            # Update failure probability based on conditions
            self._update_failure_probability(train)
    
    def _update_failure_probability(self, train: TrainState):
        """Update train failure probability based on current conditions"""
        base_risk = 0.05
        
        # Mileage factor
        if train.mileage_km > 30000:
            base_risk += 0.2
        elif train.mileage_km > 25000:
            base_risk += 0.1
        
        # Fitness certificate factor
        if train.fitness_valid_until:
            days_to_expiry = (train.fitness_valid_until.date() - self.simulation_time.date()).days
            if days_to_expiry < 7:
                base_risk += 0.15
        
        # Status factor
        if train.status == 'service':
            base_risk -= 0.05  # Service reduces immediate risk
        
        train.failure_probability = min(0.95, max(0.01, base_risk))
    
    def _notify_observers(self):
        """Notify all registered observers of state changes"""
        try:
            state_snapshot = self.get_current_state()
            for observer in self.observers[:]:  # Use slice to avoid modification during iteration
                try:
                    observer(state_snapshot)
                except Exception as e:
                    print(f"Error notifying observer: {e}")
        except Exception as e:
            print(f"Error in notify observers: {e}")
    
    def add_observer(self, callback):
        """Add an observer for real-time updates"""
        if callback not in self.observers:
            self.observers.append(callback)
    
    def remove_observer(self, callback):
        """Remove an observer"""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of entire digital twin"""
        try:
            return {
                'simulation_time': self.simulation_time.isoformat(),
                'trains': {train_id: train.to_dict() for train_id, train in self.trains.items()},
                'bays': {bay_id: bay.to_dict() for bay_id, bay in self.bays.items()},
                'is_running': self.is_running,
                'time_multiplier': self.time_multiplier,
                'summary': self._get_summary_stats()
            }
        except Exception as e:
            print(f"Error getting current state: {e}")
            return {
                'simulation_time': datetime.now().isoformat(),
                'trains': {},
                'bays': {},
                'is_running': self.is_running,
                'time_multiplier': self.time_multiplier,
                'summary': {}
            }
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        try:
            total_trains = len(self.trains)
            inducted_trains = len([t for t in self.trains.values() if t.status == 'service'])
            available_bays = len([b for b in self.bays.values() if b.status == 'available'])
            
            if self.trains:
                avg_failure_risk = float(np.mean([t.failure_probability for t in self.trains.values()]))
            else:
                avg_failure_risk = 0.0
            
            bay_utilization = 0.0
            if self.bays:
                bay_utilization = (len(self.bays) - available_bays) / len(self.bays) * 100
            
            return {
                'total_trains': total_trains,
                'inducted_trains': inducted_trains,
                'available_bays': available_bays,
                'bay_utilization': bay_utilization,
                'average_failure_risk': avg_failure_risk
            }
        except Exception as e:
            print(f"Error calculating summary stats: {e}")
            return {
                'total_trains': 0,
                'inducted_trains': 0,
                'available_bays': 0,
                'bay_utilization': 0.0,
                'average_failure_risk': 0.0
            }
    
    def schedule_event(self, event: Dict[str, Any], delay_seconds: int = 0):
        """Schedule an event to occur after delay"""
        try:
            event['scheduled_time'] = (self.simulation_time + timedelta(seconds=delay_seconds)).isoformat()
            self.event_queue.put(event)
        except Exception as e:
            print(f"Error scheduling event: {e}")
    
    def execute_induction_plan(self, induction_plan: Dict[str, Any]):
        """Execute an AI-generated induction plan in the simulation"""
        try:
            inducted_trains = induction_plan.get('inducted_trains', [])
            
            for train_info in inducted_trains:
                train_id = train_info.get('train_id')
                bay_id = train_info.get('assigned_bay')
                
                if train_id and bay_id:
                    event = {
                        'type': 'train_induction',
                        'train_id': train_id,
                        'bay_id': bay_id,
                        'source': 'ai_optimizer'
                    }
                    delay = int(np.random.randint(60, 300))
                    self.schedule_event(event, delay_seconds=delay)
            
            print(f"📋 Scheduled {len(inducted_trains)} train inductions")
        except Exception as e:
            print(f"Error executing induction plan: {e}")


class ScenarioManager:
    """Manages what-if scenarios and testing"""
    
    def __init__(self, digital_twin: DigitalTwinEngine):
        self.digital_twin = digital_twin
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        
    def create_scenario(self, scenario_name: str, scenario_config: Dict[str, Any]) -> str:
        """Create a new scenario for testing"""
        scenario_id = str(uuid.uuid4())
        
        scenario = {
            'id': scenario_id,
            'name': scenario_name,
            'config': scenario_config,
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        }
        
        self.scenarios[scenario_id] = scenario
        return scenario_id
    
    def run_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Run a specific scenario"""
        if scenario_id not in self.scenarios:
            return {'error': 'Scenario not found'}
        
        scenario = self.scenarios[scenario_id]
        config = scenario['config']
        
        # Save current state
        original_state = self.digital_twin.get_current_state()
        
        try:
            # Apply scenario modifications
            self._apply_scenario_config(config)
            
            # Run simulation for specified duration
            duration = config.get('duration_minutes', 60)
            time_multiplier = config.get('time_multiplier', 10.0)
            
            self.digital_twin.start_simulation(time_multiplier=time_multiplier)
            
            # Let it run
            time.sleep(duration / time_multiplier)
            
            # Collect results
            final_state = self.digital_twin.get_current_state()
            
            scenario['status'] = 'completed'
            scenario['results'] = self._analyze_scenario_results(original_state, final_state)
            
            return scenario['results']
            
        except Exception as e:
            scenario['status'] = 'failed'
            scenario['error'] = str(e)
            return {'error': str(e)}
        
        finally:
            self.digital_twin.stop_simulation()
    
    def _apply_scenario_config(self, config: Dict[str, Any]):
        """Apply scenario configuration to digital twin"""
        try:
            # Emergency scenarios
            if config.get('emergency_type'):
                self._simulate_emergency(config['emergency_type'])
            
            # Train failures
            if config.get('simulate_failures'):
                self._simulate_train_failures(config['simulate_failures'])
            
            # Bay outages
            if config.get('bay_outages'):
                self._simulate_bay_outages(config['bay_outages'])
        except Exception as e:
            print(f"Error applying scenario config: {e}")
    
    def _simulate_emergency(self, emergency_type: str):
        """Simulate emergency conditions"""
        event = {
            'type': 'emergency_override',
            'emergency_type': emergency_type,
            'message': f'Simulated emergency: {emergency_type}'
        }
        self.digital_twin.schedule_event(event)
    
    def _simulate_train_failures(self, failure_config: Dict[str, Any]):
        """Simulate train failures"""
        try:
            train_ids = list(self.digital_twin.trains.keys())
            if not train_ids:
                return
                
            num_failures = failure_config.get('count', 1)
            
            for _ in range(min(num_failures, len(train_ids))):
                train_id = np.random.choice(train_ids)
                train = self.digital_twin.trains[train_id]
                train.failure_probability = min(0.95, train.failure_probability + 0.4)
        except Exception as e:
            print(f"Error simulating train failures: {e}")
    
    def _simulate_bay_outages(self, outage_config: Dict[str, Any]):
        """Simulate bay outages"""
        try:
            bay_ids = list(self.digital_twin.bays.keys())
            if not bay_ids:
                return
                
            num_outages = outage_config.get('count', 1)
            
            for _ in range(min(num_outages, len(bay_ids))):
                bay_id = np.random.choice(bay_ids)
                event = {
                    'type': 'bay_maintenance',
                    'bay_id': bay_id,
                    'duration_hours': outage_config.get('duration_hours', 4)
                }
                self.digital_twin.schedule_event(event)
        except Exception as e:
            print(f"Error simulating bay outages: {e}")
    
    def _analyze_scenario_results(self, original_state: Dict[str, Any], final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scenario results"""
        try:
            original_summary = original_state.get('summary', {})
            final_summary = final_state.get('summary', {})
            
            return {
                'duration_analyzed': 'scenario_completion',
                'performance_changes': {
                    'inducted_trains_change': final_summary.get('inducted_trains', 0) - original_summary.get('inducted_trains', 0),
                    'bay_utilization_change': final_summary.get('bay_utilization', 0) - original_summary.get('bay_utilization', 0),
                    'risk_change': final_summary.get('average_failure_risk', 0) - original_summary.get('average_failure_risk', 0)
                },
                'recommendations': self._generate_recommendations(original_summary, final_summary)
            }
        except Exception as e:
            print(f"Error analyzing scenario results: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, original: Dict[str, Any], final: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scenario results"""
        recommendations = []
        
        try:
            if final.get('average_failure_risk', 0) > original.get('average_failure_risk', 0):
                recommendations.append("Consider increasing maintenance schedule frequency")
            
            if final.get('bay_utilization', 0) < 70:
                recommendations.append("Bay capacity underutilized - optimize train scheduling")
            
            if final.get('inducted_trains', 0) < original.get('inducted_trains', 0):
                recommendations.append("Review induction criteria - fewer trains inducted than expected")
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        return recommendations
