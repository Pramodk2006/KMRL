import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np

class CustomConstraintEngine:
    """Layer 1: Custom Constraint Programming Engine"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], config: dict = None):
        self.data = data
        self.config = config or {}
        self.solution_found = False
        self.conflicts = []
        self.induction_plan = {}
        self.eligible_trains = []
        self.ineligible_trains = []
        
    def check_hard_constraints(self):
        trains = self.data['trains']
        job_cards = self.data['job_cards']
        
        today = datetime.now().date()
        eligible = []
        ineligible = []
        conflicts = []
        
        print("⚙️ Checking hard constraints...")
        
        train_job_data = trains.merge(job_cards, on='train_id', how='left')
        
        for _, row in train_job_data.iterrows():
            train_id = row['train_id']
            is_eligible = True
            train_conflicts = []
            
            # Constraint 1: Fitness Certificate Validity
            fitness_date = pd.to_datetime(row['fitness_valid_until']).date()
            if fitness_date < today:
                is_eligible = False
                train_conflicts.append(f"Expired fitness certificate ({fitness_date})")
            
            # Constraint 2: Job Card Status
            job_status = row['job_card_status']
            if job_status == 'open':
                is_eligible = False
                train_conflicts.append("Open job card - maintenance required")
            
            # Constraint 3: Cleaning Slot Availability
            cleaning_slot = row['cleaning_slot_id']
            if cleaning_slot == 'None':
                is_eligible = False
                train_conflicts.append("No cleaning slot assigned")
            
            # High Mileage Warning
            mileage = row['mileage_km']
            if mileage > 30000:
                train_conflicts.append(f"High mileage warning ({mileage:,} km)")
            
            train_info = {
                'train_id': train_id,
                'fitness_valid_until': fitness_date,
                'job_card_status': job_status,
                'cleaning_slot_id': cleaning_slot,
                'mileage_km': mileage,
                'branding_hours_left': row['branding_hours_left'],
                'bay_geometry_score': row['bay_geometry_score'],
                'current_location': row.get('current_location', 'Unknown'),
                'conflicts': train_conflicts
            }
            
            if is_eligible:
                eligible.append(train_info)
                if train_conflicts:
                    print(f"⚠️ {train_id}: Eligible with warnings - {', '.join(train_conflicts)}")
            else:
                ineligible.append(train_info)
                conflicts.extend([f"{train_id}: {conflict}" for conflict in train_conflicts])
                print(f"❌ {train_id}: Ineligible - {', '.join(train_conflicts)}")
        
        self.eligible_trains = eligible
        self.ineligible_trains = ineligible
        self.conflicts = conflicts
        
        print(f"\n📊 Constraint Check Results:")
        print(f"   - Eligible trains: {len(eligible)}")
        print(f"   - Ineligible trains: {len(ineligible)}")
        print(f"   - Total conflicts: {len(conflicts)}")
        
        return eligible, ineligible
    
    def optimize_train_selection(self, eligible_trains: List[Dict]) -> Dict:
        bay_config = self.data['bay_config']
        cleaning_slots = self.data['cleaning_slots']
        
        service_bays = bay_config[bay_config['bay_type'] == 'service']
        total_bay_capacity = service_bays['max_capacity'].sum()
        total_cleaning_capacity = cleaning_slots[cleaning_slots['available_bays'] > 0]['available_bays'].sum()
        effective_capacity = min(total_bay_capacity, total_cleaning_capacity, len(eligible_trains))
        
        print(f"📊 Capacity Analysis:")
        print(f"   - Bay capacity: {total_bay_capacity}")
        print(f"   - Cleaning capacity: {total_cleaning_capacity}")
        print(f"   - Eligible trains: {len(eligible_trains)}")
        print(f"   - Effective capacity: {effective_capacity}")
        
        if len(eligible_trains) == 0:
            return {'inducted_trains': [], 'standby_trains': [], 'status': 'No eligible trains'}
        
        # Priority scoring for selection
        scored_trains = []
        for train in eligible_trains:
            score = self._calculate_priority_score(train)
            train['priority_score'] = score
            scored_trains.append(train)
        
        scored_trains.sort(key=lambda x: x['priority_score'], reverse=True)
        
        inducted_trains = scored_trains[:effective_capacity]
        standby_trains = scored_trains[effective_capacity:]
        
        inducted_with_bays = self._assign_bays_to_trains(inducted_trains, service_bays)
        
        return {
            'inducted_trains': inducted_with_bays,
            'standby_trains': standby_trains,
            'status': 'Optimal' if len(inducted_trains) == effective_capacity else 'Feasible'
        }
    
    def _calculate_priority_score(self, train: Dict) -> float:
        score = 0.0
        
        branding_hours = train['branding_hours_left']
        score += branding_hours * 2.0
        
        mileage = train['mileage_km']
        mileage_factor = max(0, (35000 - mileage) / 35000)
        score += mileage_factor * 10.0
        
        geometry_score = train['bay_geometry_score']
        score += geometry_score * 1.5
        
        fitness_date = train['fitness_valid_until']
        today = datetime.now().date()
        days_buffer = (fitness_date - today).days
        score += min(days_buffer, 30) * 0.5
        
        return score
    
    def _assign_bays_to_trains(self, trains: List[Dict], service_bays: pd.DataFrame) -> List[Dict]:
        assigned_trains = []
        bay_assignments = {}
        
        for _, bay in service_bays.iterrows():
            bay_assignments[bay['bay_id']] = {
                'capacity': bay['max_capacity'],
                'assigned': 0,
                'geometry_score': bay['geometry_score'],
                'trains': []
            }
        
        for train in trains:
            train_geometry = train['bay_geometry_score']
            best_bay = None
            best_score_diff = float('inf')
            
            for bay_id, bay_info in bay_assignments.items():
                if bay_info['assigned'] < bay_info['capacity']:
                    score_diff = abs(bay_info['geometry_score'] - train_geometry)
                    if score_diff < best_score_diff:
                        best_score_diff = score_diff
                        best_bay = bay_id
            
            if best_bay:
                bay_assignments[best_bay]['assigned'] += 1
                bay_assignments[best_bay]['trains'].append(train['train_id'])
                
                train_copy = train.copy()
                train_copy['assigned_bay'] = best_bay
                train_copy['bay_geometry_match'] = service_bays[service_bays['bay_id'] == best_bay]['geometry_score'].iloc[0]
                assigned_trains.append(train_copy)
            else:
                for bay_id, bay_info in bay_assignments.items():
                    if bay_info['assigned'] < bay_info['capacity']:
                        bay_assignments[bay_id]['assigned'] += 1
                        train_copy = train.copy()
                        train_copy['assigned_bay'] = bay_id
                        train_copy['bay_geometry_match'] = bay_info['geometry_score']
                        assigned_trains.append(train_copy)
                        break
        
        print(f"🏗️ Bay Assignment Summary:")
        for bay_id, bay_info in bay_assignments.items():
            if bay_info['assigned'] > 0:
                print(f"   - {bay_id}: {bay_info['assigned']}/{bay_info['capacity']} trains assigned")
        
        return assigned_trains
    
    def run_constraint_optimization(self) -> Dict:
        print("🚀 Starting Custom Constraint Engine (Layer 1)")
        
        eligible_trains, ineligible_trains = self.check_hard_constraints()
        
        if len(eligible_trains) > 0:
            optimization_result = self.optimize_train_selection(eligible_trains)
            self.solution_found = True
        else:
            optimization_result = {
                'inducted_trains': [],
                'standby_trains': [],
                'status': 'No eligible trains - all have constraint violations'
            }
            self.solution_found = False
        
        result = {
            'status': optimization_result['status'],
            'solution_found': self.solution_found,
            'conflicts': self.conflicts,
            'eligible_trains': eligible_trains,
            'ineligible_trains': ineligible_trains,
            'inducted_trains': optimization_result['inducted_trains'],
            'standby_trains': optimization_result['standby_trains'],
            'total_inducted': len(optimization_result['inducted_trains']),
            'total_standby': len(optimization_result['standby_trains']),
            'total_ineligible': len(ineligible_trains)
        }
        
        self.induction_plan = result
        return result
