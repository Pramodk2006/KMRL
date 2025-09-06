import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class MultiObjectiveOptimizer:
    """Layer 2: Multi-Objective Scoring and Optimization"""
    
    def __init__(self, constraint_result: Dict, data: Dict[str, pd.DataFrame], weights: Dict = None):
        self.constraint_result = constraint_result
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
        
        fitness_date = train['fitness_valid_until']
        today = datetime.now().date()
        days_buffer = (fitness_date - today).days
        
        if days_buffer < 7:
            score -= (7 - days_buffer) * 10
        
        if train['job_card_status'] == 'closed':
            score += 10
        
        if train['cleaning_slot_id'] != 'None':
            score += 15
        else:
            score -= 20
        
        return max(0, min(100, score))
    
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
        inducted_trains = self.constraint_result['inducted_trains'].copy()
        standby_trains = self.constraint_result['standby_trains'].copy()
        
        if not inducted_trains:
            return self.constraint_result
        
        all_trains = self.data['trains']
        fleet_avg_mileage = all_trains['mileage_km'].mean()
        
        print(f"📊 Multi-Objective Optimization Analysis:")
        print(f"   - Fleet average mileage: {fleet_avg_mileage:,.0f} km")
        print(f"   - Scoring weights: {self.weights}")
        
        scored_trains = []
        for train in inducted_trains:
            scores = self.score_individual_train(train, fleet_avg_mileage)
            train_with_scores = train.copy()
            train_with_scores.update(scores)
            scored_trains.append(train_with_scores)
        
        scored_standby = []
        for train in standby_trains:
            scores = self.score_individual_train(train, fleet_avg_mileage)
            train_with_scores = train.copy()
            train_with_scores.update(scores)
            scored_standby.append(train_with_scores)
        
        scored_trains.sort(key=lambda x: x['composite_score'], reverse=True)
        scored_standby.sort(key=lambda x: x['composite_score'], reverse=True)
        
        recommendations = self._analyze_induction_recommendations(scored_trains, scored_standby)
        
        return {
            'status': 'Multi-objective optimization complete',
            'inducted_trains': scored_trains,
            'standby_trains': scored_standby,
            'recommendations': recommendations,
            'fleet_avg_mileage': fleet_avg_mileage,
            'weights_used': self.weights,
            'total_inducted': len(scored_trains),
            'optimization_improvements': self._calculate_improvements(scored_trains)
        }
    
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
