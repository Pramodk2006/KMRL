import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from src import monitoring_system, iot_sensor_system
from src import enhanced_optimizer  # or predictive_model / multi_objective_optimizer depending on your intended AI optimizer

class InteractiveWebDashboard:
    """Interactive web dashboard for digital twin visualization"""
    
    def __init__(self, digital_twin_engine):
        self.digital_twin = digital_twin_engine
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        self.update_thread = None
        self.is_updating = False
        self.ai_optimizer = enhanced_optimizer  # or whichever optimizer module you want
        self.monitor = monitoring_system
        self.iot_processor = iot_sensor_system

        
        
        # Add self as observer to digital twin
        self.digital_twin.add_observer(self._on_state_update)
        self.current_state = self.digital_twin.get_current_state()
    
    def _on_state_update(self, state):
        """Callback for digital twin state updates"""
        self.current_state = state
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚄 KMRL IntelliFleet - Digital Twin Dashboard", 
                       className="text-center mb-4"),
                html.P("Real-time train induction simulation and monitoring", 
                      className="text-center text-muted")
            ], className="header-section"),
            
            # Control Panel
            html.Div([
                html.H3("🎮 Simulation Control"),
                html.Div([
                    html.Button("▶️ Start Simulation", id="start-btn", 
                               n_clicks=0, className="btn btn-success me-2"),
                    html.Button("⏸️ Pause Simulation", id="pause-btn", 
                               n_clicks=0, className="btn btn-warning me-2"),
                    html.Button("⏹️ Stop Simulation", id="stop-btn", 
                               n_clicks=0, className="btn btn-danger me-2"),
                    html.Div([
                        html.Label("Speed: "),
                        dcc.Slider(id="speed-slider", min=0.1, max=10, step=0.1, 
                                  value=1.0, marks={1: '1x', 5: '5x', 10: '10x'})
                    ], className="d-inline-block ms-3", style={'width': '200px'})
                ], className="d-flex align-items-center")
            ], className="control-panel mb-4"),
            
            # Status Overview
            html.Div([
                html.H3("📊 System Overview"),
                html.Div(id="status-cards", children=[
                    self._create_status_card("Total Trains", "10", "🚂"),
                    self._create_status_card("Inducted", "6", "✅"),
                    self._create_status_card("Available Bays", "3", "🏗️"),
                    self._create_status_card("Avg Risk", "12.4%", "⚠️")
                ], className="row")
            ], className="overview-section mb-4"),
            
            # Main Visualization Area
            html.Div([
                # Bay Layout Visualization
                html.Div([
                    html.H4("🏗️ Bay Layout & Occupancy"),
                    dcc.Graph(id="bay-layout-graph")
                ], className="col-md-6"),
                
                # Train Status Visualization
                html.Div([
                    html.H4("🚂 Train Status Distribution"),
                    dcc.Graph(id="train-status-graph")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Real-time Metrics
            html.Div([
                html.Div([
                    html.H4("📈 Real-time Performance"),
                    dcc.Graph(id="performance-timeline")
                ], className="col-md-8"),
                
                html.Div([
                    html.H4("🎯 Risk Assessment"),
                    dcc.Graph(id="risk-gauge")
                ], className="col-md-4")
            ], className="row mb-4"),
            
            # Scenario Testing Panel
            html.Div([
                html.H3("🧪 Scenario Testing"),
                html.Div([
                    html.Div([
                        html.Label("Scenario Type:"),
                        dcc.Dropdown(
                            id="scenario-type",
                            options=[
                                {'label': '🚨 Emergency Response', 'value': 'emergency'},
                                {'label': '🔧 Bay Maintenance', 'value': 'maintenance'},
                                {'label': '⚠️ Train Failures', 'value': 'failures'},
                                {'label': '📈 Peak Demand', 'value': 'peak_demand'}
                            ],
                            value='emergency'
                        )
                    ], className="col-md-3"),
                    
                    html.Div([
                        html.Label("Duration (minutes):"),
                        dcc.Input(id="scenario-duration", type="number", 
                                 value=60, min=10, max=480)
                    ], className="col-md-2"),
                    
                    html.Div([
                        html.Label("Speed Multiplier:"),
                        dcc.Input(id="scenario-speed", type="number", 
                                 value=10, min=1, max=100)
                    ], className="col-md-2"),
                    
                    html.Div([
                        html.Button("🚀 Run Scenario", id="run-scenario-btn", 
                                   n_clicks=0, className="btn btn-info mt-4")
                    ], className="col-md-2")
                ], className="row"),
                
                html.Div(id="scenario-results", className="mt-3")
            ], className="scenario-panel mb-4"),
            
            # Live Event Log
            html.Div([
                html.H3("📋 Live Event Log"),
                html.Div(id="event-log", style={
                    'height': '300px', 
                    'overflow-y': 'scroll',
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'background-color': '#f8f9fa'
                })
            ], className="event-log-section"),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),
            
            # Store for maintaining state
            dcc.Store(id='dashboard-state', data={})
        ], className="container-fluid")
    
    def _create_status_card(self, title: str, value: str, icon: str):
        """Create a status card component"""
        return html.Div([
            html.Div([
                html.Div([
                    html.Span(icon, className="status-icon"),
                    html.Div([
                        html.H4(value, className="status-value"),
                        html.P(title, className="status-title")
                    ])
                ], className="d-flex align-items-center")
            ], className="card-body")
        ], className="col-md-3 mb-3")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""
        
        @self.app.callback(
            Output('bay-layout-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_bay_layout(n):
            return self._create_bay_layout_figure()
        
        @self.app.callback(
            Output('train-status-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_train_status(n):
            return self._create_train_status_figure()
        
        @self.app.callback(
            Output('performance-timeline', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_timeline(n):
            return self._create_performance_timeline()
        
        @self.app.callback(
            Output('risk-gauge', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_gauge(n):
            return self._create_risk_gauge()
        
        @self.app.callback(
            Output('status-cards', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_cards(n):
            summary = self.current_state.get('summary', {})
            return [
                self._create_status_card("Total Trains", str(summary.get('total_trains', 0)), "🚂"),
                self._create_status_card("Inducted", str(summary.get('inducted_trains', 0)), "✅"),
                self._create_status_card("Available Bays", str(summary.get('available_bays', 0)), "🏗️"),
                self._create_status_card("Avg Risk", f"{summary.get('average_failure_risk', 0):.1%}", "⚠️")
            ]
        
        @self.app.callback(
            Output('event-log', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_event_log(n):
            return self._create_event_log()
        
        # Simulation control callbacks
        @self.app.callback(
            Output('dashboard-state', 'data'),
            [Input('start-btn', 'n_clicks'),
             Input('pause-btn', 'n_clicks'), 
             Input('stop-btn', 'n_clicks')],
            [State('speed-slider', 'value'),
             State('dashboard-state', 'data')]
        )
        def control_simulation(start_clicks, pause_clicks, stop_clicks, speed, current_state):
            ctx = callback_context
            if not ctx.triggered:
                return current_state or {}
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn' and start_clicks > 0:
                self.digital_twin.start_simulation(time_multiplier=speed)
                return {'status': 'running', 'speed': speed}
            elif button_id == 'pause-btn' and pause_clicks > 0:
                self.digital_twin.stop_simulation()
                return {'status': 'paused', 'speed': speed}
            elif button_id == 'stop-btn' and stop_clicks > 0:
                self.digital_twin.stop_simulation()
                return {'status': 'stopped', 'speed': speed}
            
            return current_state or {}
        
        # Scenario testing callback
        @self.app.callback(
            Output('scenario-results', 'children'),
            [Input('run-scenario-btn', 'n_clicks')],
            [State('scenario-type', 'value'),
             State('scenario-duration', 'value'),
             State('scenario-speed', 'value')]
        )
        def run_scenario(n_clicks, scenario_type, duration, speed):
            if n_clicks == 0:
                return html.Div()
            
            # Create scenario configuration
            scenario_config = {
                'duration_minutes': duration,
                'time_multiplier': speed
            }
            
            if scenario_type == 'emergency':
                scenario_config['emergency_type'] = 'power_outage'
            elif scenario_type == 'maintenance':
                scenario_config['bay_outages'] = {'count': 2, 'duration_hours': 4}
            elif scenario_type == 'failures':
                scenario_config['simulate_failures'] = {'count': 2}
            elif scenario_type == 'peak_demand':
                scenario_config['increased_demand'] = {'factor': 1.5}
            
            # Run scenario (simplified for demo)
            scenario_id = self.digital_twin.scenario_manager.create_scenario(
                f"Test Scenario - {scenario_type}", scenario_config
            )
            
            return html.Div([
                html.H5(f"🧪 Scenario '{scenario_type}' initiated"),
                html.P(f"Scenario ID: {scenario_id}"),
                html.P(f"Duration: {duration} minutes at {speed}x speed"),
                html.Div([
                    html.Span("⏳ Running scenario...", className="text-info")
                ])
            ], className="alert alert-info")
    
    def _create_bay_layout_figure(self):
        """Create bay layout visualization"""
        bays = self.current_state.get('bays', {})
        
        # Create a grid layout for bays
        bay_data = []
        colors = []
        texts = []
        
        for i, (bay_id, bay_info) in enumerate(bays.items()):
            row = i // 3
            col = i % 3
            
            bay_data.append([col, row])
            
            # Color based on status
            if bay_info['status'] == 'available':
                colors.append('lightgreen')
            elif bay_info['status'] == 'occupied':
                colors.append('orange')
            elif bay_info['status'] == 'partial':
                colors.append('yellow')
            else:
                colors.append('lightgray')
            
            # Text with bay info
            occupancy = len(bay_info.get('occupied_trains', []))
            capacity = bay_info.get('max_capacity', 1)
            texts.append(f"{bay_id}<br>{occupancy}/{capacity}")
        
        if bay_data:
            bay_array = np.array(bay_data)
            
            fig = go.Figure(data=go.Scatter(
                x=bay_array[:, 0],
                y=bay_array[:, 1],
                mode='markers+text',
                marker=dict(
                    size=80,
                    color=colors,
                    line=dict(width=2, color='black')
                ),
                text=texts,
                textposition="middle center",
                textfont=dict(size=10),
                name="Bays"
            ))
            
            fig.update_layout(
                title="Bay Layout and Occupancy",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
                height=300
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="No bay data available")
        
        return fig
    
    def _create_train_status_figure(self):
        """Create train status distribution chart"""
        trains = self.current_state.get('trains', {})
        
        status_counts = {}
        for train_info in trains.values():
            status = train_info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Train Status Distribution"
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="No train data available")
        
        return fig
    
    def _create_performance_timeline(self):
        """Create performance timeline chart"""
        # This would typically show historical performance data
        # For demo, we'll create a simple timeline
        
        times = pd.date_range(start=datetime.now() - timedelta(hours=2), 
                             end=datetime.now(), freq='10min')
        
        # Simulate performance metrics
        inducted_trains = np.random.randint(4, 8, len(times))
        bay_utilization = np.random.uniform(60, 95, len(times))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=inducted_trains,
            mode='lines+markers',
            name='Inducted Trains',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=bay_utilization,
            mode='lines+markers',
            name='Bay Utilization (%)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Performance Timeline',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Inducted Trains', side='left'),
            yaxis2=dict(title='Bay Utilization (%)', side='right', overlaying='y'),
            height=300
        )
        
        return fig
    
    def _create_risk_gauge(self):
        """Create risk assessment gauge"""
        summary = self.current_state.get('summary', {})
        avg_risk = summary.get('average_failure_risk', 0.1) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fleet Risk Level (%)"},
            delta = {'reference': 15},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def _create_event_log(self):
        """Create live event log"""
        # This would typically show real events from the digital twin
        # For demo, we'll show recent train state changes
        
        events = []
        trains = self.current_state.get('trains', {})
        
        for train_id, train_info in trains.items():
            recent_events = train_info.get('recent_events', [])
            for event in recent_events[-3:]:  # Last 3 events
                timestamp = event.get('timestamp', '')
                event_type = event.get('event_type', '')
                
                if timestamp:
                    try:
                        time_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = time_obj.strftime('%H:%M:%S')
                    except:
                        time_str = timestamp[:8]
                else:
                    time_str = 'Unknown'
                
                events.append(html.Div([
                    html.Span(f"[{time_str}] ", className="text-muted"),
                    html.Span(f"{train_id}: ", className="fw-bold"),
                    html.Span(f"{event.get('old_status', '')} → {event.get('new_status', '')}")
                ]))
        
        if not events:
            events = [html.Div("No recent events", className="text-muted")]
        
        return events[-10:]  # Show last 10 events
    
    def run_server(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server"""
        print(f"🌐 Starting web dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
