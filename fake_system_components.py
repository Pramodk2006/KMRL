# fake_system_components.py
"""
Fake system components that create convincing demonstrations of all missing features.
These provide realistic outputs for mileage balancing, cleaning management, etc.
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from dash import html, dcc

class FakeMileageBalancer:
    """Fake mileage balancing system with realistic optimization"""
    
    def __init__(self, mileage_data):
        self.mileage_data = mileage_data
        self.optimization_results = {}
        
    def run_optimization(self):
        """Run fake mileage optimization"""
        # Calculate fake optimization results
        train_mileages = self.mileage_data.groupby('train_id')['cumulative_km'].last()
        target_mileage = train_mileages.mean()
        
        self.optimization_results = {}
        for train_id, mileage in train_mileages.items():
            difference = mileage - target_mileage
            if difference > 1500:
                self.optimization_results[train_id] = {
                    'status': 'over_target',
                    'difference': difference,
                    'recommendation': 'Reduce service hours',
                    'savings': difference * 0.15
                }
            elif difference < -1500:
                self.optimization_results[train_id] = {
                    'status': 'under_target', 
                    'difference': difference,
                    'recommendation': 'Increase service allocation',
                    'savings': abs(difference) * 0.12
                }
        
    def create_distribution_chart(self):
        """Create mileage distribution visualization"""
        train_mileages = self.mileage_data.groupby('train_id')['cumulative_km'].last()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_mileages.values,
            nbinsx=10,
            name='Mileage Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add target line
        target = train_mileages.mean()
        fig.add_vline(x=target, line_dash="dash", line_color="red", 
                     annotation_text=f"Target: {target:.0f} km")
        
        fig.update_layout(
            title='Fleet Mileage Distribution Analysis',
            xaxis_title='Cumulative Kilometers',
            yaxis_title='Number of Trains',
            height=300
        )
        
        return fig
    
    def create_wear_analysis_chart(self):
        """Create component wear analysis"""
        latest_data = self.mileage_data.groupby('train_id').last()
        
        fig = go.Figure()
        
        # Bogie wear
        fig.add_trace(go.Scatter(
            x=latest_data.index,
            y=latest_data['bogie_wear_index'],
            mode='markers',
            name='Bogie Wear %',
            marker=dict(size=10, color='blue')
        ))
        
        # Brake pad wear
        fig.add_trace(go.Scatter(
            x=latest_data.index,
            y=latest_data['brake_pad_wear'],
            mode='markers',
            name='Brake Pad Wear %',
            marker=dict(size=10, color='red')
        ))
        
        # Warning line at 80%
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="Warning Level (80%)")
        
        fig.update_layout(
            title='Component Wear Analysis by Train',
            xaxis_title='Train ID',
            yaxis_title='Wear Percentage',
            height=300,
            hovermode='x unified'
        )
        
        return fig

class FakeCleaningManager:
    """Fake cleaning bay and staff management system"""
    
    def __init__(self, cleaning_data):
        self.cleaning_data = cleaning_data
        self.current_assignments = {}
        
    def optimize_scheduling(self):
        """Run fake cleaning optimization"""
        # Generate fake current assignments
        bays = ['CB1', 'CB2', 'CB3', 'CB4']
        trains = [f'T{i:03d}' for i in range(1, 26)]
        
        self.current_assignments = {}
        for bay in bays:
            if random.random() > 0.3:  # 70% chance bay is occupied
                self.current_assignments[bay] = {
                    'train': random.choice(trains),
                    'type': random.choice(['Deep Clean', 'Standard', 'Express']),
                    'staff': random.sample(['Ravi Kumar', 'Priya S', 'Anand M', 'Lakshmi R'], 2),
                    'time_remaining': random.randint(1, 4)
                }
    
    def create_bay_status_chart(self):
        """Create cleaning bay status visualization"""
        bays = ['CB1', 'CB2', 'CB3', 'CB4']
        statuses = []
        colors = []
        
        for bay in bays:
            if bay in self.current_assignments:
                statuses.append('Occupied')
                colors.append('red')
            else:
                statuses.append('Available')
                colors.append('green')
        
        fig = go.Figure(go.Bar(
            x=bays,
            y=[1]*len(bays),
            text=statuses,
            textposition='middle center',
            marker_color=colors,
            hovertemplate='%{x}<br>Status: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Real-Time Cleaning Bay Status',
            yaxis_visible=False,
            height=250
        )
        
        return fig

class FakeExplainableAI:
    """Fake explainable AI reasoning system"""
    
    def __init__(self):
        self.decision_factors = [
            'fitness_certificate', 'job_cards', 'mileage_status', 
            'branding_sla', 'bay_geometry', 'component_health'
        ]
        
    def explain_decision(self, train_id, rank):
        """Generate fake explanation for train ranking"""
        factors = {}
        total_score = 0
        
        for factor in self.decision_factors:
            if factor == 'fitness_certificate':
                score = random.randint(20, 25)
                factors[factor] = {
                    'score': score,
                    'explanation': 'Valid until 2026-12-31' if score > 22 else 'Expires soon'
                }
            elif factor == 'job_cards':
                score = random.randint(10, 20)
                factors[factor] = {
                    'score': score,
                    'explanation': 'All tasks completed' if score > 17 else 'Minor tasks pending'
                }
            # Add more factor explanations...
            
            total_score += score
        
        return {
            'train_id': train_id,
            'rank': rank,
            'total_score': total_score,
            'factors': factors,
            'confidence': min(99, total_score + random.randint(5, 15))
        }

class FakeScenarioSimulator:
    """Fake what-if scenario simulation system"""
    
    def run_simulation(self, scenario_type, affected_trains):
        """Run fake scenario simulation"""
        if scenario_type == 'fitness_failure':
            return self._simulate_fitness_failure(affected_trains)
        elif scenario_type == 'emergency_maintenance':
            return self._simulate_emergency_maintenance(affected_trains)
        else:
            return self._simulate_generic_scenario(scenario_type, affected_trains)
    
    def _simulate_fitness_failure(self, affected_trains):
        """Simulate fitness certificate failure"""
        train_list = affected_trains if isinstance(affected_trains, list) else [affected_trains]
        
        results = []
        for train in train_list:
            backup_train = f'T{random.randint(19, 25):03d}'
            additional_cost = random.randint(15000, 35000)
            
            results.append(html.Div([
                html.H5(f"🔄 Scenario: {train} Fitness Failure", style={'color': '#f44336'}),
                html.Hr(),
                html.Div([
                    "📉 Impact Analysis:",
                    html.Ul([
                        html.Li(f"Service capacity reduced by 1 train (-5.6%)"),
                        html.Li(f"{backup_train} promoted from standby to service"),
                        html.Li(f"Additional cost: ₹{additional_cost:,} (suboptimal backup)"),
                        html.Li("Bay reshuffling required: +12 minutes setup time"),
                        html.Li("Branding contract at risk - reassignment needed")
                    ])
                ], style={'marginBottom': '15px'}),
                html.Div([
                    "✅ Auto-Generated Solution:",
                    html.Ul([
                        html.Li(f"Promote {backup_train} to primary service roster"),
                        html.Li("Reassign branding contracts to available trains"),
                        html.Li(f"Schedule {train} for expedited fitness renewal"),
                        html.Li("Alert supervisor for backup train approval")
                    ])
                ], style={'background': '#e8f5e8', 'padding': '10px', 'borderRadius': '5px'})
            ]))
        
        return results
    
    def _simulate_generic_scenario(self, scenario_type, affected_trains):
        """Simulate other scenario types"""
        impact_messages = {
            'emergency_maintenance': 'Emergency maintenance reduces fleet availability',
            'staff_shortage': 'Cleaning operations delayed by staff shortage', 
            'bay_failure': 'Service bay equipment failure affects capacity',
            'demand_surge': 'Peak demand requires additional service trains'
        }
        
        return [html.Div([
            html.H5(f"🔄 Scenario: {scenario_type.replace('_', ' ').title()}", 
                   style={'color': '#f44336'}),
            html.P(impact_messages.get(scenario_type, 'Scenario impact analysis')),
            html.Div("Detailed simulation results would appear here...", 
                    style={'background': '#f5f5f5', 'padding': '15px', 'borderRadius': '5px'})
        ])]

class FakeMLEngine:
    """Fake machine learning engine with learning simulation"""
    
    def __init__(self):
        self.accuracy = 94.2
        self.learning_rate = 0.001
        self.decisions_processed = 247
        
    def process_feedback(self):
        """Simulate ML learning from feedback"""
        # Simulate slight accuracy improvement over time
        if random.random() < 0.1:  # 10% chance of improvement
            self.accuracy += random.uniform(0.1, 0.3)
            self.accuracy = min(99.5, self.accuracy)  # Cap at 99.5%
        
        self.decisions_processed += random.randint(1, 5)

class FakeMaximoConnector:
    """Fake IBM Maximo integration connector"""
    
    def __init__(self, maximo_data):
        self.maximo_data = maximo_data
        self.connection_status = "Active"
        self.last_sync = datetime.now()
        
    def get_live_data_feed(self):
        """Generate fake live data feed"""
        current_time = datetime.now()
        
        feed_items = []
        for i in range(10):
            timestamp = current_time - timedelta(seconds=i*30)
            train_id = f'T{random.randint(1, 25):03d}'
            sensor = random.choice(['Door_Sensor', 'HVAC_Temp', 'Brake_Pressure', 'Battery_Voltage'])
            value = round(random.uniform(20, 100), 1)
            
            feed_items.append(html.Div(
                f"{timestamp.strftime('%H:%M:%S')} | {train_id} | {sensor} | {value} | Status: Normal",
                style={
                    'fontFamily': 'monospace',
                    'padding': '3px 5px',
                    'background': '#f8f9fa' if i % 2 == 0 else '#ffffff',
                    'fontSize': '0.85rem'
                }
            ))
        
        return feed_items

class FakeKPITracker:
    """Fake KPI tracking and performance monitoring"""
    
    def __init__(self, performance_data):
        self.performance_data = performance_data
        self.current_kpis = {}
        
    def update_metrics(self):
        """Update fake KPI metrics"""
        self.current_kpis = {
            'punctuality': 99.7,
            'availability': 96.0,
            'customer_satisfaction': 4.6,
            'energy_efficiency': 87.3,
            'maintenance_cost': 12.4
        }
    
    def create_trends_chart(self):
        """Create KPI trends visualization"""
        # Generate fake trend data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        punctuality = np.random.normal(99.5, 0.5, 30)
        punctuality = np.clip(punctuality, 98.5, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=punctuality,
            mode='lines+markers',
            name='Punctuality %',
            line=dict(color='#4caf50', width=3)
        ))
        
        # Add target line
        fig.add_hline(y=99.5, line_dash="dash", line_color="red",
                     annotation_text="Target: 99.5%")
        
        fig.update_layout(
            title='Punctuality KPI Trend (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Punctuality %',
            height=300,
            yaxis=dict(range=[98, 100.5])
        )
        
        return fig

# Export all fake components
__all__ = [
    'FakeMileageBalancer',
    'FakeCleaningManager', 
    'FakeExplainableAI',
    'FakeScenarioSimulator',
    'FakeMLEngine',
    'FakeMaximoConnector',
    'FakeKPITracker'
]