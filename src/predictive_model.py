import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle
import os
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not installed. Using fallback prediction model.")

class PredictiveFailureModel:
    """Layer 3A: Predictive Failure Model using LightGBM"""
    
    def __init__(self, model_path: str = "models/failure_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = [
            'mileage_km', 'days_since_maintenance', 'branding_hours_left',
            'bay_geometry_score', 'seasonal_factor', 'weather_factor'
        ]
        self.is_trained = False
        
    def prepare_features(self, trains_df: pd.DataFrame, job_cards_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Merge train and job card data
        merged_df = trains_df.merge(job_cards_df, on='train_id', how='left')
        
        # Calculate days since last maintenance
        merged_df['last_maintenance_date'] = pd.to_datetime(merged_df['last_maintenance_date'])
        today = datetime.now()
        merged_df['days_since_maintenance'] = (today - merged_df['last_maintenance_date']).dt.days
        
        # Add seasonal factor (0-1 based on month)
        month = today.month
        if month in [6, 7, 8, 9]:  # Monsoon season
            seasonal_factor = 0.8  # Higher failure risk
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 0.3  # Lower failure risk
        else:
            seasonal_factor = 0.5  # Normal
        
        merged_df['seasonal_factor'] = seasonal_factor
        
        # Add weather factor (simulated)
        merged_df['weather_factor'] = np.random.uniform(0.2, 0.8, len(merged_df))
        
        return merged_df[['train_id'] + self.feature_columns]
    
    def train_model(self, historical_df: pd.DataFrame, trains_df: pd.DataFrame, job_cards_df: pd.DataFrame):
        """Train the predictive model using historical data"""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available. Using simple fallback model.")
            self.is_trained = True
            return
        
        print("🤖 Training Predictive Failure Model...")
        
        # Prepare features from historical data
        features_df = self.prepare_features(trains_df, job_cards_df)
        
        # Merge with historical outcomes
        train_data = historical_df.merge(features_df, on='train_id', how='inner')
        
        # Prepare training data
        X = train_data[self.feature_columns].fillna(0)
        y = train_data['actual_failure_occurred'].astype(int)
        
        # Split data (simple split for demo)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train LightGBM model
        train_dataset = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_dataset,
            num_boost_round=100,
            valid_sets=[train_dataset],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = np.mean((y_pred > 0.5) == y_test)
        
        print(f"✅ Model trained with {accuracy:.3f} accuracy")
        print(f"📊 Feature importance: {dict(zip(self.feature_columns, self.model.feature_importance()))}")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.is_trained = True
    
    def predict_failure_probability(self, trains_df: pd.DataFrame, job_cards_df: pd.DataFrame) -> Dict[str, float]:
        """Predict failure probability for each train"""
        features_df = self.prepare_features(trains_df, job_cards_df)
        
        if not LIGHTGBM_AVAILABLE or not self.is_trained:
            # Fallback prediction based on simple rules
            predictions = {}
            for _, row in features_df.iterrows():
                train_id = row['train_id']
                
                # Simple rule-based prediction
                risk_score = 0.0
                if row['mileage_km'] > 30000:
                    risk_score += 0.4
                if row['days_since_maintenance'] > 30:
                    risk_score += 0.3
                if row['seasonal_factor'] > 0.6:
                    risk_score += 0.2
                
                predictions[train_id] = min(risk_score, 0.95)
            
            return predictions
        
        # Use trained model
        X = features_df[self.feature_columns].fillna(0)
        probabilities = self.model.predict(X)
        
        return dict(zip(features_df['train_id'], probabilities))
    
    def get_risk_insights(self, predictions: Dict[str, float]) -> Dict:
        """Generate risk insights from predictions"""
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        high_risk_trains = []
        
        for train_id, prob in predictions.items():
            if prob < 0.3:
                risk_levels['low'] += 1
            elif prob < 0.6:
                risk_levels['medium'] += 1
            elif prob < 0.8:
                risk_levels['high'] += 1
                high_risk_trains.append((train_id, prob))
            else:
                risk_levels['critical'] += 1
                high_risk_trains.append((train_id, prob))
        
        return {
            'risk_distribution': risk_levels,
            'high_risk_trains': sorted(high_risk_trains, key=lambda x: x[1], reverse=True),
            'average_risk': np.mean(list(predictions.values())),
            'total_trains_analyzed': len(predictions)
        }


class ReinforcementLearningAgent:
    """Layer 3B: RL Agent for Weight Optimization"""
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = {}
        self.weight_history = []
        self.performance_history = []
        
    def get_state(self, context: Dict) -> str:
        """Convert context to state string"""
        season = context.get('season', 'normal')
        avg_mileage = context.get('avg_mileage', 22000)
        conflict_rate = context.get('conflict_rate', 0.4)
        
        # Discretize continuous values
        mileage_bucket = 'high' if avg_mileage > 25000 else 'medium' if avg_mileage > 20000 else 'low'
        conflict_bucket = 'high' if conflict_rate > 0.5 else 'medium' if conflict_rate > 0.3 else 'low'
        
        return f"{season}_{mileage_bucket}_{conflict_bucket}"
    
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_weights(self, base_weights: Dict, action: str) -> Dict:
        """Update weights based on action"""
        new_weights = base_weights.copy()
        
        # Define weight adjustment actions
        adjustments = {
            'increase_service': {'service_readiness': 0.05},
            'increase_maintenance': {'maintenance_penalty': 0.05},
            'increase_branding': {'branding_priority': 0.05},
            'balanced_approach': {},  # No change
            'seasonal_adjust': {'maintenance_penalty': 0.1, 'service_readiness': -0.05}
        }
        
        if action in adjustments:
            for weight_name, adjustment in adjustments[action].items():
                if weight_name in new_weights:
                    new_weights[weight_name] = max(0.05, min(0.5, new_weights[weight_name] + adjustment))
        
        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def learn_from_outcome(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-table based on outcome"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        
        # Q-learning update
        future_reward = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        self.q_table[state][action] += self.learning_rate * (reward + 0.9 * future_reward - self.q_table[state][action])
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of the RL agent"""
        if not self.performance_history:
            return {'status': 'No performance data available'}
        
        recent_performance = self.performance_history[-10:]
        
        return {
            'total_episodes': len(self.performance_history),
            'recent_avg_performance': np.mean(recent_performance),
            'best_performance': max(self.performance_history),
            'learning_trend': 'improving' if len(recent_performance) > 5 and np.mean(recent_performance[-5:]) > np.mean(recent_performance[:5]) else 'stable',
            'q_table_size': len(self.q_table)
        }


class HistoricalPatternAnalyzer:
    """Layer 3C: Historical Pattern Recognition"""
    
    def __init__(self):
        self.patterns = {}
        self.seasonal_trends = {}
        self.anomalies = []
        
    def analyze_patterns(self, historical_df: pd.DataFrame) -> Dict:
        """Analyze historical patterns and trends"""
        print("📊 Analyzing historical patterns...")
        
        if historical_df.empty:
            return {'status': 'No historical data available'}
        
        # Convert date column
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        historical_df['month'] = historical_df['date'].dt.month
        historical_df['weekday'] = historical_df['date'].dt.weekday
        
        # Seasonal analysis
        monthly_stats = historical_df.groupby('month').agg({
            'inducted': 'mean',
            'actual_failure_occurred': 'mean',
            'branding_sla_met': 'mean',
            'energy_consumed_kwh': 'mean'
        }).round(3)
        
        self.seasonal_trends = {
            'monthly_induction_rate': monthly_stats['inducted'].to_dict(),
            'monthly_failure_rate': monthly_stats['actual_failure_occurred'].to_dict(),
            'monthly_branding_compliance': monthly_stats['branding_sla_met'].to_dict(),
            'monthly_energy_consumption': monthly_stats['energy_consumed_kwh'].to_dict()
        }
        
        # Day of week analysis
        weekday_stats = historical_df.groupby('weekday').agg({
            'inducted': 'mean',
            'passenger_complaints': 'mean'
        }).round(3)
        
        # Anomaly detection (simple statistical approach)
        self._detect_anomalies(historical_df)
        
        # Pattern insights
        insights = self._generate_insights(historical_df)
        
        return {
            'seasonal_trends': self.seasonal_trends,
            'weekday_patterns': weekday_stats.to_dict(),
            'anomalies_detected': len(self.anomalies),
            'insights': insights,
            'data_coverage_days': (historical_df['date'].max() - historical_df['date'].min()).days
        }
    
    def _detect_anomalies(self, df: pd.DataFrame):
        """Detect anomalies in historical data"""
        # Simple statistical anomaly detection
        for column in ['energy_consumed_kwh', 'passenger_complaints']:
            if column in df.columns:
                mean_val = df[column].mean()
                std_val = df[column].std()
                threshold = mean_val + 2 * std_val
                
                anomalies = df[df[column] > threshold]
                for _, row in anomalies.iterrows():
                    self.anomalies.append({
                        'date': row['date'],
                        'type': f'high_{column}',
                        'value': row[column],
                        'threshold': threshold
                    })
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from patterns"""
        insights = []
        
        # Failure rate insights
        if 'actual_failure_occurred' in df.columns:
            failure_rate = df['actual_failure_occurred'].mean()
            if failure_rate > 0.15:
                insights.append(f"High failure rate detected: {failure_rate:.1%}. Consider increasing maintenance penalty weight.")
        
        # Energy consumption insights
        if 'energy_consumed_kwh' in df.columns:
            high_energy_days = df[df['energy_consumed_kwh'] > df['energy_consumed_kwh'].quantile(0.9)]
            if len(high_energy_days) > 0:
                insights.append(f"High energy consumption on {len(high_energy_days)} days. Consider optimizing bay assignments.")
        
        # Branding SLA insights
        if 'branding_sla_met' in df.columns:
            sla_compliance = df['branding_sla_met'].mean()
            if sla_compliance < 0.8:
                insights.append(f"Branding SLA compliance at {sla_compliance:.1%}. Consider increasing branding priority weight.")
        
        return insights
    
    def get_seasonal_recommendations(self, current_month: int) -> Dict:
        """Get recommendations based on seasonal patterns"""
        if not self.seasonal_trends:
            return {'status': 'No seasonal data available'}
        
        month_failure_rate = self.seasonal_trends.get('monthly_failure_rate', {}).get(current_month, 0.1)
        month_energy = self.seasonal_trends.get('monthly_energy_consumption', {}).get(current_month, 180)
        
        recommendations = []
        weight_adjustments = {}
        
        if month_failure_rate > 0.15:
            recommendations.append("High failure season - increase maintenance focus")
            weight_adjustments['maintenance_penalty'] = 0.05
        
        if month_energy > 200:
            recommendations.append("High energy consumption season - optimize bay utilization")
            weight_adjustments['shunting_cost'] = 0.05
        
        return {
            'month': current_month,
            'recommendations': recommendations,
            'weight_adjustments': weight_adjustments,
            'expected_failure_rate': month_failure_rate,
            'expected_energy_consumption': month_energy
        }
