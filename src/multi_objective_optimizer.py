import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class MultiObjectiveOptimizer:
    """Layer 2: Multi-Objective Scoring and Optimization"""
    
    def __init__(self, constraint_result: Dict = None, data: Dict[str, pd.DataFrame] = None, weights: Dict = None):
        # Allow no-arg init for tests; data can be provided later or auto-loaded
        self.constraint_result = constraint_result or {}
        if data is None:
            try:
                from .data_loader import DataLoader
            except ImportError:
                from src.data_loader import DataLoader
            data = DataLoader().get_integrated_data()
        self.data = data
        self.weights = weights or self._get_default_weights()
        self.optimized_result = {}
        
    def _get_default_weights(self) -> Dict[str, float]:
        return {
            "service_readiness": 0.25,
            "maintenance_penalty": 0.25,
            "branding_priority": 0.20,
            "mileage_balance": 0.15,
            "shunting_cost": 0.15
        }
    
    def calculate_service_readiness_score(self, train: Dict) -> float:
        score = 100.0
        
        # Always valid fitness date gives full points
        fitness_date = train['fitness_valid_until']
        today = datetime.now().date()
        days_buffer = (fitness_date - today).days
        
        if days_buffer > 30:  # More than a month valid
            score += 20
        elif days_buffer > 0:  # Still valid
            score += 10
        
        # Closed job cards are good
        if train.get('job_card_status', 'closed') == 'closed':
            score += 20
        
        # Having a cleaning slot assigned is good
        if train['cleaning_slot_id'] != 'None':
            score += 20
        
        # Lower mileage is better
        if train['mileage_km'] < 25000:
            score += 20
        elif train['mileage_km'] < 30000:
            score += 10
        
        # Good bay geometry score is important
        bay_score = float(train['bay_geometry_score'])
        if bay_score >= 0.8:
            score += 20
        elif bay_score >= 0.6:
            score += 10
            
        return max(60, min(100, score))  # Minimum score of 60 to be inducted
    
    def calculate_maintenance_penalty(self, train: Dict) -> float:
        penalty = 0.0
        
        mileage = train['mileage_km']
        if mileage > 30000:
            penalty += ((mileage - 30000) / 5000) * 30
        elif mileage > 25000:
            penalty += ((mileage - 25000) / 5000) * 15
        
        if train.get('maintenance_priority', 'low') == 'high':
            penalty += 25
        elif train.get('maintenance_priority', 'low') == 'medium':
            penalty += 10
        
        return min(100, penalty)
    
    def calculate_branding_priority_score(self, train: Dict) -> float:
        branding_hours = train['branding_hours_left']
        
        if branding_hours == 0:
            return 0
        elif branding_hours < 2:
            return 25
        elif branding_hours < 10:
            return 60
        else:
            return 100
    
    def calculate_mileage_balance_score(self, train: Dict, fleet_avg_mileage: float) -> float:
        train_mileage = train['mileage_km']
        deviation = abs(train_mileage - fleet_avg_mileage)
        max_acceptable_deviation = 5000
        
        if deviation <= max_acceptable_deviation:
            balance_score = 100 - (deviation / max_acceptable_deviation) * 30
        else:
            balance_score = 70 - min(30, ((deviation - max_acceptable_deviation) / 2000) * 10)
        
        return max(0, balance_score)
    
    def calculate_shunting_cost(self, train: Dict) -> float:
        geometry_score = train['bay_geometry_score']
        assigned_bay_geometry = train.get('bay_geometry_match', geometry_score)
        
        geometry_diff = abs(geometry_score - assigned_bay_geometry)
        shunting_cost = geometry_diff * 10
        
        if train.get('current_location') == 'Depot_B':
            shunting_cost += 15
        
        return min(100, shunting_cost)
    
    def score_individual_train(self, train: Dict, fleet_avg_mileage: float) -> Dict:
        scores = {
            'service_readiness': self.calculate_service_readiness_score(train),
            'maintenance_penalty': self.calculate_maintenance_penalty(train),
            'branding_priority': self.calculate_branding_priority_score(train),
            'mileage_balance': self.calculate_mileage_balance_score(train, fleet_avg_mileage),
            'shunting_cost': self.calculate_shunting_cost(train)
        }
        
        print(f"\nScoring Train {train['train_id']}:")
        print(f"  Service Readiness: {scores['service_readiness']:.2f}")
        print(f"  Maintenance Penalty: {scores['maintenance_penalty']:.2f}")
        print(f"  Branding Priority: {scores['branding_priority']:.2f}")
        print(f"  Mileage Balance: {scores['mileage_balance']:.2f}")
        print(f"  Shunting Cost: {scores['shunting_cost']:.2f}")
        
        composite_score = 0.0
        for objective, score in scores.items():
            weight = self.weights[objective]
            
            if objective in ['maintenance_penalty', 'shunting_cost']:
                normalized_score = (100 - score) / 100
            else:
                normalized_score = score / 100
            
            composite_score += weight * normalized_score
        
        scores['composite_score'] = composite_score * 100
        return scores
    
    def optimize_induction_ranking(self) -> Dict:
        """Optimize and rank trains for induction based on multiple objectives."""
        # Get eligible trains from constraint engine result
        eligible_trains = self.constraint_result.get('eligible_trains', [])
        
        # If no eligible trains, return empty result
        if not eligible_trains:
            return {
                'status': 'no_eligible_trains',
                'inducted_trains': [],
                'standby_trains': [],
                'recommendations': [],
                'total_inducted': 0,
                'total_standby': 0
            }
        
        # Calculate fleet average mileage
        all_trains = self.data['trains']
        fleet_avg_mileage = all_trains['mileage_km'].mean()
        
        print(f"📊 Multi-Objective Optimization Analysis:")
        print(f"   - Fleet average mileage: {fleet_avg_mileage:,.0f} km")
        print(f"   - Scoring weights: {self.weights}")
        
        # Score all eligible trains
        scored_trains = []
        for train in eligible_trains:
            scores = self.score_individual_train(train, fleet_avg_mileage)
            train_with_scores = train.copy()
            train_with_scores.update(scores)
            scored_trains.append(train_with_scores)
        
        # Sort trains by composite score
        scored_trains.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        print("\nRanked trains by composite score:")
        for train in scored_trains:
            print(f"Train {train['train_id']}: Composite Score = {train['composite_score']:.2f}")
        
        # Select top trains for induction based on bay capacity - FIXED: Calculate actual capacity
        bay_config = self.data.get('bay_config', pd.DataFrame())
        if not bay_config.empty:
            # Calculate total service bay capacity
            service_bays = bay_config[bay_config['bay_type'] == 'service']
            bay_capacity = service_bays['max_capacity'].sum() if not service_bays.empty else 4
        else:
            bay_capacity = 4  # Fallback if no bay config
        
        print(f"📊 Bay Capacity Analysis:")
        print(f"   - Available service bay capacity: {bay_capacity}")
        print(f"   - Eligible trains: {len(scored_trains)}")
        
        inducted_trains = scored_trains[:bay_capacity] if scored_trains else []
        standby_trains = scored_trains[bay_capacity:] if scored_trains else []
        
        print(f"\nSelected for induction ({len(inducted_trains)} trains):")
        for train in inducted_trains:
            print(f"Train {train['train_id']}: Score = {train['composite_score']:.2f}")
        
        # Assign bay IDs to inducted trains - FIXED: Use actual bay IDs from config
        if not bay_config.empty and not service_bays.empty:
            # Get actual service bay IDs from configuration
            bay_ids = service_bays['bay_id'].tolist()
        else:
            # Fallback bay IDs
            bay_ids = ['SB001', 'SB004']  # Only service bays from the CSV
        
        # Assign trains to bays, considering bay capacity
        bay_assignments = {}
        train_index = 0
        
        for bay_id in bay_ids:
            if train_index >= len(inducted_trains):
                break
                
            # Get bay capacity
            bay_capacity = service_bays[service_bays['bay_id'] == bay_id]['max_capacity'].iloc[0] if not service_bays.empty else 2
            
            # Assign trains to this bay up to its capacity
            for _ in range(int(bay_capacity)):
                if train_index < len(inducted_trains):
                    inducted_trains[train_index]['assigned_bay'] = bay_id
                    train_index += 1
                else:
                    break
        
        # Assign remaining trains to N/A if no more bays available
        for i in range(train_index, len(inducted_trains)):
            inducted_trains[i]['assigned_bay'] = 'N/A'
        
        # Store optimized result
        self.optimized_result = {
            'status': 'success',
            'inducted_trains': inducted_trains,
            'standby_trains': standby_trains,
            'recommendations': [],
            'total_inducted': len(inducted_trains),
            'total_standby': len(standby_trains)
        }
        
        return self.optimized_result
    
    def _analyze_induction_recommendations(self, inducted: List[Dict], standby: List[Dict]) -> List[Dict]:
        recommendations = []
        
        if not standby:
            return recommendations
        
        if inducted:
            lowest_inducted = min(inducted, key=lambda x: x['composite_score'])
            highest_standby = max(standby, key=lambda x: x['composite_score'])
            
            if highest_standby['composite_score'] > lowest_inducted['composite_score']:
                recommendations.append({
                    'type': 'swap_recommendation',
                    'remove_train': lowest_inducted['train_id'],
                    'add_train': highest_standby['train_id'],
                    'score_improvement': highest_standby['composite_score'] - lowest_inducted['composite_score'],
                    'reason': f"Standby train {highest_standby['train_id']} scores {highest_standby['composite_score']:.1f} vs inducted train {lowest_inducted['train_id']} at {lowest_inducted['composite_score']:.1f}"
                })
        
        return recommendations
    
    def _calculate_improvements(self, trains: List[Dict]) -> Dict:
        if not trains:
            return {}
        
        avg_composite = np.mean([t['composite_score'] for t in trains])
        avg_service_readiness = np.mean([t['service_readiness'] for t in trains])
        avg_maintenance_penalty = np.mean([t['maintenance_penalty'] for t in trains])
        avg_branding_priority = np.mean([t['branding_priority'] for t in trains])
        
        return {
            'avg_composite_score': avg_composite,
            'avg_service_readiness': avg_service_readiness,
            'avg_maintenance_penalty': avg_maintenance_penalty,
            'avg_branding_priority': avg_branding_priority,
            'score_distribution': {
                'excellent': len([t for t in trains if t['composite_score'] >= 80]),
                'good': len([t for t in trains if 60 <= t['composite_score'] < 80]),
                'acceptable': len([t for t in trains if 40 <= t['composite_score'] < 60]),
                'poor': len([t for t in trains if t['composite_score'] < 40])
            }
        }
