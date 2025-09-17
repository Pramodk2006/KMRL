import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from .multi_objective_optimizer import MultiObjectiveOptimizer
from .predictive_model import PredictiveFailureModel, ReinforcementLearningAgent, HistoricalPatternAnalyzer

class EnhancedMultiObjectiveOptimizer(MultiObjectiveOptimizer):
    """Enhanced optimizer with predictive intelligence"""
    
    def __init__(self, constraint_result: Dict = None, data: Dict[str, pd.DataFrame] = None, weights: Dict = None):
        super().__init__(constraint_result, data, weights)
        
        # Initialize AI components
        self.failure_model = PredictiveFailureModel()
        self.rl_agent = ReinforcementLearningAgent()
        self.pattern_analyzer = HistoricalPatternAnalyzer()
        
        # Enhanced optimization results
        self.ai_insights = {}
        self.predictive_scores = {}
        
    def initialize_ai_models(self, historical_df: pd.DataFrame):
        """Initialize and train AI models"""
        print("🤖 Initializing AI models...")
        
        # Train predictive failure model
        trains_df = self.data['trains']
        job_cards_df = self.data['job_cards']
        
        if not historical_df.empty:
            self.failure_model.train_model(historical_df, trains_df, job_cards_df)
        
        # Analyze historical patterns
        pattern_results = self.pattern_analyzer.analyze_patterns(historical_df)
        
        # Get seasonal recommendations
        current_month = datetime.now().month
        seasonal_rec = self.pattern_analyzer.get_seasonal_recommendations(current_month)
        
        self.ai_insights = {
            'pattern_analysis': pattern_results,
            'seasonal_recommendations': seasonal_rec,
            'models_ready': True
        }
        # Compatibility alias expected by tests
        self.ai_insights['seasonal_patterns'] = pattern_results.get('seasonal_trends', {})
        
        print("✅ AI models initialized successfully")
    
    def calculate_enhanced_scores(self, train: Dict, fleet_avg_mileage: float, failure_predictions: Dict) -> Dict:
        """Calculate scores with AI enhancements"""
        # Get base scores
        base_scores = self.score_individual_train(train, fleet_avg_mileage)
        
        # Add predictive failure penalty
        train_id = train['train_id']
        failure_prob = failure_predictions.get(train_id, 0.1)
        
        # Adjust maintenance penalty based on failure prediction
        failure_penalty = failure_prob * 50  # Scale 0-1 to 0-50
        enhanced_maintenance_penalty = min(100, base_scores['maintenance_penalty'] + failure_penalty)
        
        # Calculate predictive risk score
        risk_score = 100 - (failure_prob * 100)  # Higher is better
        
        # Enhanced composite score with predictive component
        predictive_weight = 0.1  # 10% weight for predictive component
        base_composite = base_scores['composite_score']
        
        enhanced_composite = (
            base_composite * (1 - predictive_weight) + 
            risk_score * predictive_weight
        )
        
        # Update scores
        enhanced_scores = base_scores.copy()
        enhanced_scores.update({
            'maintenance_penalty': enhanced_maintenance_penalty,
            'predictive_risk_score': risk_score,
            'failure_probability': failure_prob,
            'composite_score': enhanced_composite
        })
        
        return enhanced_scores
    
    def optimize_with_ai(self, historical_df: pd.DataFrame = None) -> Dict:
        """Run optimization with AI enhancements"""
        print("🧠 Running AI-Enhanced Optimization...")
        
        # Initialize AI models if historical data provided
        if historical_df is not None and not historical_df.empty:
            self.initialize_ai_models(historical_df)
        
        # Get failure predictions
        trains_df = self.data['trains']
        job_cards_df = self.data['job_cards']
        failure_predictions = self.failure_model.predict_failure_probability(trains_df, job_cards_df)
        
        # Get risk insights
        risk_insights = self.failure_model.get_risk_insights(failure_predictions)
        
        # Adaptive weight adjustment using RL
        total_trains = len(trains_df) if trains_df is not None else 0
        avg_mileage = trains_df['mileage_km'].mean() if total_trains > 0 else 0.0
        conflicts_ct = len(self.constraint_result.get('conflicts', [])) if isinstance(self.constraint_result, dict) else 0
        conflict_rate = (conflicts_ct / total_trains) if total_trains > 0 else 0.0
        current_context = {
            'season': self._get_season(),
            'avg_mileage': avg_mileage,
            'conflict_rate': conflict_rate
        }
        
        # Apply seasonal weight adjustments if available
        adjusted_weights = self._apply_seasonal_adjustments()
        self.weights = adjusted_weights
        
        # Enhanced scoring for inducted trains
        inducted_trains = self.constraint_result['inducted_trains'].copy()
        standby_trains = self.constraint_result['standby_trains'].copy()
        
        fleet_avg_mileage = trains_df['mileage_km'].mean()
        
        enhanced_inducted = []
        for train in inducted_trains:
            enhanced_scores = self.calculate_enhanced_scores(train, fleet_avg_mileage, failure_predictions)
            train_with_scores = train.copy()
            train_with_scores.update(enhanced_scores)
            enhanced_inducted.append(train_with_scores)
        
        enhanced_standby = []
        for train in standby_trains:
            enhanced_scores = self.calculate_enhanced_scores(train, fleet_avg_mileage, failure_predictions)
            train_with_scores = train.copy()
            train_with_scores.update(enhanced_scores)
            enhanced_standby.append(train_with_scores)
        
        # Re-rank by enhanced composite score
        enhanced_inducted.sort(key=lambda x: x['composite_score'], reverse=True)
        enhanced_standby.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Generate AI-powered recommendations
        ai_recommendations = self._generate_ai_recommendations(enhanced_inducted, enhanced_standby, risk_insights)
        
        return {
            'status': 'AI-enhanced optimization complete',
            'inducted_trains': enhanced_inducted,
            'standby_trains': enhanced_standby,
            'recommendations': ai_recommendations,
            'risk_insights': risk_insights,
            'ai_insights': self.ai_insights,
            'weights_used': self.weights,
            'total_inducted': len(enhanced_inducted),
            'optimization_improvements': self._calculate_enhanced_improvements(enhanced_inducted)
        }
    
    def _get_season(self) -> str:
        """Determine current season"""
        month = datetime.now().month
        if month in [6, 7, 8, 9]:
            return 'monsoon'
        elif month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'summer'
        else:
            return 'normal'
    
    def _apply_seasonal_adjustments(self) -> Dict:
        """Apply seasonal weight adjustments"""
        adjusted_weights = self.weights.copy()
        
        if 'seasonal_recommendations' in self.ai_insights:
            seasonal_rec = self.ai_insights['seasonal_recommendations']
            weight_adjustments = seasonal_rec.get('weight_adjustments', {})
            
            for weight_name, adjustment in weight_adjustments.items():
                if weight_name in adjusted_weights:
                    adjusted_weights[weight_name] = max(0.05, min(0.5, adjusted_weights[weight_name] + adjustment))
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _generate_ai_recommendations(self, inducted: List[Dict], standby: List[Dict], risk_insights: Dict) -> List[Dict]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # High-risk train recommendations
        high_risk_trains = risk_insights.get('high_risk_trains', [])
        for train_id, risk_prob in high_risk_trains[:3]:  # Top 3 high-risk
            # Check if high-risk train is inducted
            inducted_train = next((t for t in inducted if t['train_id'] == train_id), None)
            if inducted_train:
                recommendations.append({
                    'type': 'high_risk_alert',
                    'train_id': train_id,
                    'risk_probability': risk_prob,
                    'message': f"Train {train_id} has {risk_prob:.1%} failure risk - consider additional inspection",
                    'priority': 'high'
                })
        
        # Capacity optimization recommendations
        avg_risk = risk_insights.get('average_risk', 0.1)
        if avg_risk > 0.3:
            recommendations.append({
                'type': 'fleet_risk_alert',
                'message': f"Fleet average risk at {avg_risk:.1%} - consider reducing induction capacity",
                'priority': 'medium'
            })
        
        # Seasonal recommendations
        if 'seasonal_recommendations' in self.ai_insights:
            seasonal_rec = self.ai_insights['seasonal_recommendations']
            for rec in seasonal_rec.get('recommendations', []):
                recommendations.append({
                    'type': 'seasonal_adjustment',
                    'message': rec,
                    'priority': 'low'
                })
        
        return recommendations
    
    def _calculate_enhanced_improvements(self, trains: List[Dict]) -> Dict:
        """Calculate improvements with AI metrics"""
        if not trains:
            return {}
        
        base_improvements = self._calculate_improvements(trains)
        
        # Add AI-specific metrics
        avg_failure_risk = np.mean([t.get('failure_probability', 0.1) for t in trains])
        avg_predictive_score = np.mean([t.get('predictive_risk_score', 50) for t in trains])
        
        ai_improvements = {
            'avg_failure_risk': avg_failure_risk,
            'avg_predictive_score': avg_predictive_score,
            'ai_enhancement_active': True,
            'risk_distribution': {
                'low_risk': len([t for t in trains if t.get('failure_probability', 0.1) < 0.3]),
                'medium_risk': len([t for t in trains if 0.3 <= t.get('failure_probability', 0.1) < 0.6]),
                'high_risk': len([t for t in trains if t.get('failure_probability', 0.1) >= 0.6])
            }
        }
        
        # Combine with base improvements
        base_improvements.update(ai_improvements)
        return base_improvements
