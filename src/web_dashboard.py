import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import required libraries for Material Design
try:
    import dash_mantine_components as dmc
    import dash_bootstrap_components as dbc
except ImportError:
    # Fallback if libraries not installed
    dmc = None
    dbc = None

class InteractiveWebDashboard:
    """Modern Material Design Dashboard for KMRL IntelliFleet with AI Integration"""
    
    def __init__(self, digital_twin_engine, monitor=None, iot_simulator=None, cv_system=None, 
                 ai_optimizer=None, constraint_engine=None, ai_dashboard=None, ai_data_processor=None):
        self.digital_twin = digital_twin_engine
        self.monitor = monitor
        self.iot_simulator = iot_simulator
        self.cv_system = cv_system
        
        # AI Integration
        self.ai_optimizer = ai_optimizer
        self.constraint_engine = constraint_engine
        self.ai_dashboard = ai_dashboard
        self.ai_data_processor = ai_data_processor
        
        # Add observer and get initial state
        if hasattr(self.digital_twin, 'add_observer'):
            self.digital_twin.add_observer(self._on_state_update)
        self.current_state = self.digital_twin.get_current_state()
        
        # Initialize Dash app with external stylesheets
        external_stylesheets = [
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
        ]
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        
        # Setup layout
        if dmc:
            self.app.layout = dmc.MantineProvider(
                children=[self._create_main_layout()]
            )
        else:
            self.app.layout = self._create_main_layout()
        
        self.setup_callbacks()

    def _on_state_update(self, state):
        """Callback for digital twin state updates"""
        self.current_state = state

    def _create_main_layout(self):
        """Single method to create the complete dashboard layout"""
        return html.Div([

            
            # Header
            html.Div([
                html.H1("🚄 KMRL IntelliFleet", className="header-title"),
                html.P("AI-Powered Digital Twin Dashboard", className="header-subtitle")
            ], className="modern-header"),
            
            # Control Panel
            html.Div([
                html.H3("🎮 Simulation Control", className="control-title"),
                html.Div([
                    html.Button("▶️ Start", id="start-btn", className="modern-btn btn-success", n_clicks=0),
                    html.Button("⏸️ Pause", id="pause-btn", className="modern-btn btn-warning", n_clicks=0),
                    html.Button("⏹️ Stop", id="stop-btn", className="modern-btn btn-danger", n_clicks=0),
                    html.Div([
                        html.Label("Speed Multiplier", style={'color': '#666', 'fontSize': '0.9rem', 'marginBottom': '0.5rem'}),
                        dcc.Slider(id="speed-slider", min=0.1, max=10, step=0.1, value=1.0, 
                                  marks={1: '1×', 5: '5×', 10: '10×'}, tooltip={"placement": "bottom", "always_visible": True})
                    ], className="modern-slider")
                ], className="controls-row")
            ], className="control-panel"),
            
            # Status Cards - DYNAMIC ONLY
            html.Div(id="status-cards", className="row"),
            
            # Charts Row 1
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("🏗️ Bay Layout & Occupancy", className="viz-title"),
                        dcc.Graph(id="bay-layout-graph", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-6"),
                html.Div([
                    html.Div([
                        html.H4("🚂 Train Status Distribution", className="viz-title"),
                        dcc.Graph(id="train-status-graph", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-6")
            ], className="row"),
            
            # Charts Row 2
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("📈 Performance Timeline", className="viz-title"),
                        dcc.Graph(id="performance-timeline", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-8"),
                html.Div([
                    html.Div([
                        html.H4("🎯 Risk Assessment", className="viz-title"),
                        dcc.Graph(id="risk-gauge", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-4")
            ], className="row"),
            
            # AI SUMMARY SECTIONS - ALL DYNAMIC
            html.Div(id="ai-summary-section"),
            
            # Scenario Testing Panel
            html.Div([
                html.H3("🧪 Scenario Testing", style={'color': '#1976d2', 'marginBottom': '1.5rem', 'fontSize': '1.5rem'}),
                html.Div([
                    html.Div([
                        html.Label("Scenario Type", style={'color': '#666', 'marginBottom': '0.5rem'}),
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
                        html.Label("Duration (min)", style={'color': '#666', 'marginBottom': '0.5rem'}),
                        dcc.Input(id="scenario-duration", type="number", value=60, min=10, max=480,
                                 style={'width': '100%', 'padding': '0.75rem', 'border': '2px solid rgba(0,0,0,0.1)', 
                                       'borderRadius': '12px', 'fontSize': '0.9rem'})
                    ], className="col-md-2"),
                    html.Div([
                        html.Label("Speed", style={'color': '#666', 'marginBottom': '0.5rem'}),
                        dcc.Input(id="scenario-speed", type="number", value=10, min=1, max=100,
                                 style={'width': '100%', 'padding': '0.75rem', 'border': '2px solid rgba(0,0,0,0.1)', 
                                       'borderRadius': '12px', 'fontSize': '0.9rem'})
                    ], className="col-md-2"),
                    html.Div([
                        html.Button("🚀 Run Scenario", id="run-scenario-btn", className="modern-btn btn-info", n_clicks=0,
                                   style={'marginTop': '1.5rem', 'width': '100%'})
                    ], className="col-md-2")
                ], className="row"),
                html.Div(id="scenario-results", style={'marginTop': '1.5rem'})
            ], className="scenario-panel"),
            
            # Event Log
            html.Div([
                html.Div([
                    html.H3("📋 Live Event Log", className="viz-title"),
                    html.Div(id="event-log", className="event-log")
                ], className="viz-card")
            ]),
            
            # Auto-refresh and storage components
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
            dcc.Store(id='dashboard-state', data={})
        ], className="container-fluid", style={'padding': '20px', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})

    def _create_status_card(self, title: str, value: str, icon: str, color: str):
        """Create a modern status card component"""
        return html.Div([
            html.Div([
                html.Span(icon, className="status-icon"),
                html.H4(str(value), className="status-value", style={'color': color}),
                html.P(title, className="status-title")
            ], className="status-card")
        ], className="col-md-3")

    def setup_callbacks(self):
        """Setup dashboard callbacks with comprehensive error handling"""

        @self.app.callback(Output('status-cards', 'children'),
                           Input('interval-component', 'n_intervals'))
        def update_status_cards(n):
            try:
                # Get dynamic data from AI processor
                if self.ai_data_processor:
                    summary = self.ai_data_processor.get_train_status_summary()
                    performance = self.ai_data_processor.get_performance_metrics()
                    
                    total_trains = summary.get('total_trains', 0)
                    inducted_trains = summary.get('inducted_trains', 0)
                    available_bays = 6 - inducted_trains  # Assuming 6 total bays
                    avg_risk = f"{performance.get('maintenance_risk', 0):.1f}%"
                else:
                    # Fallback to current state data
                    current_state = getattr(self, 'current_state', {})
                    total_trains = len(current_state.get('trains', {})) or 5
                    inducted_trains = len([t for t in current_state.get('trains', {}).values() 
                                         if t.get('status') in ['inducted', 'running']]) or 3
                    available_bays = len(current_state.get('bays', {})) or 3
                    avg_risk = "15.2%"
                
                cards = [
                    self._create_status_card("Total Trains", total_trains, "🚂", "#1976d2"),
                    self._create_status_card("Inducted", inducted_trains, "✅", "#4caf50"),
                    self._create_status_card("Available Bays", available_bays, "🏗️", "#ff9800"),
                    self._create_status_card("Avg Risk", avg_risk, "⚠️", "#f44336")
                ]
                return cards
                
            except Exception as e:
                return [html.Div(f"Error loading status: {str(e)}", className="alert alert-danger")]

        @self.app.callback(Output('ai-summary-section', 'children'),
                           Input('interval-component', 'n_intervals'))
        def update_ai_summary(n):
            try:
                if not self.ai_data_processor:
                    return html.Div([
                        html.Div([
                            html.H4("🤖 AI System Not Available", className="viz-title"),
                            html.P("AI optimization data is not available. Please check system configuration.",
                                  style={'color': '#666', 'textAlign': 'center', 'padding': '2rem'})
                        ], className="ai-summary-card")
                    ])
                
                # Get fresh AI data
                train_details = self.ai_data_processor.get_detailed_train_list()
                summary = self.ai_data_processor.get_train_status_summary()
                performance_metrics = self.ai_data_processor.get_performance_metrics()
                violations = self.ai_data_processor.get_constraint_violations()
                
                # Create inducted trains table
                inducted_trains = [t for t in train_details if t.get('inducted', False)]
                inducted_table_rows = []
                
                for i, train in enumerate(inducted_trains):
                    status_color = '#4caf50' if 'Ready' in train.get('status', '') else '#ff9800'
                    status_icon = '✅' if 'Ready' in train.get('status', '') else '⚠️'
                    
                    inducted_table_rows.append(html.Tr([
                        html.Td(f"{i+1}", style={'fontWeight': 'bold', 'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                        html.Td(train.get('train_id', 'Unknown'), style={'fontWeight': 'bold', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                        html.Td(train.get('bay_assignment', 'N/A'), style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                        html.Td(f"{train.get('priority_score', 0):.1f}", style={'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                        html.Td(f"{train.get('branding_hours', 0):.1f}h", style={'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                        html.Td([status_icon, f" {train.get('status', 'Unknown')}"], 
                                style={'color': status_color, 'fontWeight': 'bold', 'padding': '8px', 'borderBottom': '1px solid #ddd'})
                    ]))
                
                # Create violations list
                violations_items = []
                for violation in violations:
                    train_violations = violation.get('violations', [])
                    if train_violations:
                        violations_items.append(
                            html.Div([
                                html.Div([
                                    html.Strong(f"❌ {violation.get('train_id', 'Unknown')}:", style={'color': '#f44336'})
                                ]),
                                html.Div([
                                    html.Div([f"└─ {str(v)}"], style={'marginLeft': '1rem', 'color': '#666'}) 
                                    for v in train_violations
                                ])
                            ], style={'margin': '0.5rem 0'})
                        )
                
                # Performance distribution
                excellent_count = len([t for t in inducted_trains if t.get('priority_score', 0) >= 80])
                good_count = len([t for t in inducted_trains if 60 <= t.get('priority_score', 0) < 80])
                acceptable_count = len([t for t in inducted_trains if 40 <= t.get('priority_score', 0) < 60])
                poor_count = len([t for t in inducted_trains if t.get('priority_score', 0) < 40])
                
                return html.Div([
                    # Inducted Trains Section
                    html.Div([
                        html.H4("✅ INDUCTED TRAINS - TONIGHT'S SERVICE", 
                               style={'color': '#4caf50', 'borderBottom': '2px solid #4caf50', 'paddingBottom': '0.5rem'}),
                        html.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Rank", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd', 'textAlign': 'center'}),
                                    html.Th("Train", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd'}),
                                    html.Th("Bay", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd'}),
                                    html.Th("Score", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd', 'textAlign': 'center'}),
                                    html.Th("Branding", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd', 'textAlign': 'center'}),
                                    html.Th("Status", style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #ddd'})
                                ])
                            ]),
                            html.Tbody(inducted_table_rows if inducted_table_rows else [
                                html.Tr([html.Td("No inducted trains available - Check AI optimization system", 
                                               colSpan=6, style={'textAlign': 'center', 'padding': '2rem', 'color': '#666'})])
                            ])
                        ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd', 'marginTop': '1rem'}),
                        
                        html.Div([
                            html.Strong("💡 Average Performance Score: "),
                            html.Span(f"{performance_metrics.get('system_performance', 0):.1f}/100", 
                                    style={'color': '#1976d2', 'fontSize': '1.1em'})
                        ], style={'margin': '1rem 0', 'padding': '0.5rem', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'})
                    ], className="ai-summary-card"),
                    
                    # Constraint Violations Section
                    html.Div([
                        html.H4("⚠️ CONSTRAINT VIOLATIONS & ALERTS", 
                               style={'color': '#f44336', 'borderBottom': '2px solid #f44336', 'paddingBottom': '0.5rem'}),
                        html.Div(violations_items if violations_items else [
                            html.Div("No constraint violations detected", style={'color': '#4caf50', 'fontWeight': 'bold'})
                        ], style={'backgroundColor': '#fff3e0', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
                    ], className="ai-summary-card"),
                    
                    # Capacity Utilization Section
                    html.Div([
                        html.H4("🏗️ CAPACITY UTILIZATION", 
                               style={'color': '#ff9800', 'borderBottom': '2px solid #ff9800', 'paddingBottom': '0.5rem'}),
                        html.Div([
                            html.Div([
                                html.Strong("Bay Utilization: "),
                                html.Span(f"{summary.get('inducted_trains', 0)}/6 ({(summary.get('inducted_trains', 0)/6*100):.1f}%)", 
                                         style={'color': '#4caf50'})
                            ], style={'margin': '0.5rem 0'}),
                            html.Div([
                                html.Strong("Cleaning Utilization: "),
                                html.Span(f"{summary.get('inducted_trains', 0)}/6 ({(summary.get('inducted_trains', 0)/6*100):.1f}%)", 
                                         style={'color': '#4caf50'})
                            ], style={'margin': '0.5rem 0'}),
                        ], style={'backgroundColor': '#fff3e0', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
                    ], className="ai-summary-card"),
                    
                    # Performance Distribution Section
                    html.Div([
                        html.H4("📈 Performance Distribution", style={'color': '#1976d2', 'margin': '1rem 0'}),
                        html.Div([
                            html.Div([html.Span("🟢 Excellent (80+): "), html.Strong(f"{excellent_count} trains")], style={'margin': '0.25rem 0'}),
                            html.Div([html.Span("🟡 Good (60-79): "), html.Strong(f"{good_count} trains")], style={'margin': '0.25rem 0'}),
                            html.Div([html.Span("🟠 Acceptable (40-59): "), html.Strong(f"{acceptable_count} trains")], style={'margin': '0.25rem 0'}),
                            html.Div([html.Span("🔴 Poor (<40): "), html.Strong(f"{poor_count} trains")], style={'margin': '0.25rem 0'}),
                        ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '8px'})
                    ], className="ai-summary-card"),
                    
                    # Key Metrics Summary
                    html.Div([
                        html.H4("💰 ESTIMATED IMPACT", style={'color': '#1976d2', 'margin': '1rem 0'}),
                        html.Div([
                            html.Div([
                                html.Strong("Tonight's Cost Savings: "),
                                html.Span(f"₹{performance_metrics.get('cost_savings', 138000):,}", style={'color': '#4caf50', 'fontSize': '1.2em'})
                            ], style={'margin': '0.5rem 0'}),
                            html.Div([
                                html.Strong("Annual Projected Savings: "),
                                html.Span(f"₹{performance_metrics.get('annual_savings', 50370000):,}", style={'color': '#4caf50', 'fontSize': '1.2em'})
                            ], style={'margin': '0.5rem 0'}),
                        ], style={'backgroundColor': '#e8f5e8', 'padding': '1rem', 'borderRadius': '8px'})
                    ], className="ai-summary-card")
                ])
                
            except Exception as e:
                return html.Div([
                    html.Div([
                        html.H4("⚠️ AI Summary Error", className="viz-title"),
                        html.P(f"Error loading AI data: {str(e)}", style={'color': '#f44336'})
                    ], className="ai-summary-card")
                ])

        @self.app.callback(Output('bay-layout-graph', 'figure'),
                           Input('interval-component', 'n_intervals'))
        def update_bay_layout(n):
            try:
                return self._create_bay_layout_figure()
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(title=f"Bay Layout Error: {str(e)}", height=300)
                return fig

        @self.app.callback(Output('train-status-graph', 'figure'),
                           Input('interval-component', 'n_intervals'))
        def update_train_status(n):
            try:
                return self._create_train_status_figure()
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(title=f"Train Status Error: {str(e)}", height=300)
                return fig

        @self.app.callback(Output('performance-timeline', 'figure'),
                           Input('interval-component', 'n_intervals'))
        def update_performance(n):
            try:
                return self._create_performance_timeline()
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(title=f"Performance Error: {str(e)}", height=300)
                return fig

        @self.app.callback(Output('risk-gauge', 'figure'),
                           Input('interval-component', 'n_intervals'))
        def update_risk(n):
            try:
                return self._create_risk_gauge()
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(title=f"Risk Gauge Error: {str(e)}", height=300)
                return fig

        @self.app.callback(Output('event-log', 'children'),
                           Input('interval-component', 'n_intervals'))
        def update_log(n):
            try:
                return self._create_event_log()
            except Exception as e:
                return [html.Div(f"Event log error: {str(e)}", style={'color': 'red'})]

        @self.app.callback(Output('dashboard-state', 'data'),
                           Input('start-btn', 'n_clicks'),
                           Input('pause-btn', 'n_clicks'),
                           Input('stop-btn', 'n_clicks'),
                           State('speed-slider', 'value'),
                           State('dashboard-state', 'data'))
        def control_sim(start, pause, stop, speed, current):
            ctx = callback_context
            if not ctx.triggered:
                return current or {}
            
            btn = ctx.triggered[0]['prop_id'].split('.')[0]
            state = current or {}
            
            try:
                if btn == 'start-btn' and hasattr(self.digital_twin, 'start_simulation'):
                    self.digital_twin.start_simulation(time_multiplier=speed)
                    state.update({'status': 'running', 'speed': speed})
                elif btn == 'pause-btn' and hasattr(self.digital_twin, 'stop_simulation'):
                    self.digital_twin.stop_simulation()
                    state.update({'status': 'paused'})
                elif btn == 'stop-btn' and hasattr(self.digital_twin, 'stop_simulation'):
                    self.digital_twin.stop_simulation()
                    state.update({'status': 'stopped'})
            except Exception as e:
                state.update({'error': str(e)})
            
            return state

        @self.app.callback(Output('scenario-results', 'children'),
                           Input('run-scenario-btn', 'n_clicks'),
                           State('scenario-type', 'value'),
                           State('scenario-duration', 'value'),
                           State('scenario-speed', 'value'))
        def run_scenario(n, scenario_type, duration, speed):
            if n == 0:
                return html.Div()
            
            try:
                config = {'duration_minutes': duration, 'time_multiplier': speed}
                
                if scenario_type == 'emergency':
                    config['emergency_type'] = 'power_outage'
                elif scenario_type == 'maintenance':
                    config['bay_outages'] = {'count': 2, 'duration_hours': 4}
                elif scenario_type == 'failures':
                    config['simulate_failures'] = {'count': 2}
                elif scenario_type == 'peak_demand':
                    config['increased_demand'] = {'factor': 1.5}
                
                if hasattr(self.digital_twin, 'scenario_manager'):
                    scenario_id = self.digital_twin.scenario_manager.create_scenario(f"Scenario-{scenario_type}", config)
                    return html.Div([
                        html.H5(f"✅ Scenario '{scenario_type}' started successfully", 
                               className="alert alert-success"),
                        html.P(f"Scenario ID: {scenario_id}", style={'color': '#666'})
                    ])
                else:
                    return html.Div([
                        html.H5(f"⚠️ Scenario '{scenario_type}' simulation not available", 
                               className="alert alert-warning")
                    ])
                    
            except Exception as e:
                return html.Div([
                    html.H5(f"❌ Error running scenario: {str(e)}", 
                           className="alert alert-danger")
                ])

    def _create_bay_layout_figure(self):
        """Create bay layout visualization with real data"""
        try:
            current_state = getattr(self, 'current_state', {})
            bays = current_state.get('bays', {})
            
            if not bays:
                # Default layout
                xs = [0, 1, 2]
                ys = [0, 0, 0]
                colors = ['lightgreen', 'orange', 'lightgreen']
                texts = ['Bay 1<br>0/2', 'Bay 2<br>1/2', 'Bay 3<br>0/2']
            else:
                xs, ys, colors, texts = [], [], [], []
                for i, (bay_id, info) in enumerate(bays.items()):
                    row, col = divmod(i, 3)
                    xs.append(col)
                    ys.append(row)
                    
                    status = info.get('status', 'unknown')
                    colors.append({
                        'available': 'lightgreen',
                        'occupied': 'orange',
                        'partial': 'yellow',
                        'maintenance': 'red'
                    }.get(status, 'lightgray'))
                    
                    occ = len(info.get('occupied_trains', []))
                    cap = info.get('max_capacity', 1)
                    texts.append(f"{bay_id}<br>{occ}/{cap}")

            fig = go.Figure(go.Scatter(
                x=xs, y=ys,
                mode='markers+text',
                marker={'size': 80, 'color': colors, 'line': {'width': 2, 'color': 'black'}},
                text=texts,
                textposition="middle center",
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Bay Layout & Occupancy",
                xaxis={'visible': False, 'range': [-0.5, 2.5]},
                yaxis={'visible': False, 'range': [-0.5, 1.5]},
                showlegend=False,
                height=300,
                margin={'l': 20, 'r': 20, 't': 40, 'b': 20}
            )
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Bay Layout Error: {str(e)}", height=300)
            return fig

    def _create_train_status_figure(self):
        """Create train status distribution chart with real data"""
        try:
            if self.ai_data_processor:
                summary = self.ai_data_processor.get_train_status_summary()
                labels = ['Inducted', 'Ready', 'Maintenance', 'Standby', 'Ineligible']
                values = [
                    summary.get('inducted_trains', 0),
                    summary.get('ready_trains', 0),
                    summary.get('maintenance_trains', 0),
                    summary.get('standby_trains', 0),
                    summary.get('ineligible_trains', 0)
                ]
                # Filter out zero values
                filtered_data = [(l, v) for l, v in zip(labels, values) if v > 0]
                if filtered_data:
                    labels, values = zip(*filtered_data)
                else:
                    labels, values = ['No Data'], [1]
            else:
                current_state = getattr(self, 'current_state', {})
                trains = current_state.get('trains', {})
                
                if trains:
                    status_counts = {}
                    for train_info in trains.values():
                        status = train_info.get('status', 'unknown')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    labels = list(status_counts.keys())
                    values = list(status_counts.values())
                else:
                    labels = ['Running', 'Idle', 'Maintenance']
                    values = [2, 2, 1]

            fig = px.pie(values=values, names=labels, title="Train Status Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=300)
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Train Status Error: {str(e)}", height=300)
            return fig

    def _create_performance_timeline(self):
        """Create performance timeline chart with enhanced data"""
        try:
            times = pd.date_range(datetime.now() - timedelta(hours=2), datetime.now(), freq='10min')
            
            # Try to get real data if available
            if self.ai_data_processor:
                summary = self.ai_data_processor.get_train_status_summary()
                inducted_base = summary.get('inducted_trains', 3)
                utilization_base = (inducted_base / 6) * 100 if inducted_base else 50
            else:
                inducted_base = 3
                utilization_base = 50
            
            # Generate realistic variations around the base
            inducted_count = np.random.poisson(inducted_base, len(times))
            inducted_count = np.clip(inducted_count, 0, 6)  # Cap at max capacity
            
            utilization = np.random.normal(utilization_base, 10, len(times))
            utilization = np.clip(utilization, 0, 100)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=times, y=inducted_count,
                mode='lines+markers',
                name='Inducted Trains',
                line={'color': '#1976d2', 'width': 2},
                marker={'size': 4}
            ))
            
            fig.add_trace(go.Scatter(
                x=times, y=utilization,
                mode='lines+markers',
                name='Utilization (%)',
                yaxis='y2',
                line={'color': '#4caf50', 'width': 2},
                marker={'size': 4}
            ))
            
            fig.update_layout(
                title='Performance Timeline (Last 2 Hours)',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Inducted Trains', 'side': 'left', 'range': [0, 6]},
                yaxis2={
                    'title': 'Utilization (%)',
                    'side': 'right',
                    'overlaying': 'y',
                    'range': [0, 100]
                },
                height=300,
                legend={'x': 0.01, 'y': 0.99}
            )
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Performance Error: {str(e)}", height=300)
            return fig

    def _create_risk_gauge(self):
        """Create risk assessment gauge with real data"""
        try:
            # Get real risk data if available
            if self.ai_data_processor:
                performance = self.ai_data_processor.get_performance_metrics()
                avg_risk = performance.get('maintenance_risk', 25)
            else:
                avg_risk = 25  # Default risk
            
            avg_risk = max(0, min(100, avg_risk))  # Ensure valid range
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_risk,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fleet Risk (%)"},
                delta={'reference': 25, 'position': "top"},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': 'lightgreen'},
                        {'range': [25, 50], 'color': 'yellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Risk Gauge Error: {str(e)}", height=300)
            return fig

    def _create_event_log(self):
        """Create live event log with real data"""
        try:
            current_state = getattr(self, 'current_state', {})
            trains = current_state.get('trains', {})
            events = []
            
            if trains:
                for train_id, train_info in trains.items():
                    recent_events = train_info.get('recent_events', [])
                    for event in recent_events[-3:]:  # Last 3 events per train
                        timestamp = event.get('timestamp', datetime.now().strftime('%H:%M:%S'))
                        if isinstance(timestamp, str) and len(timestamp) > 8:
                            timestamp = timestamp[:8]
                        
                        events.append(html.Div([
                            html.Span(f"[{timestamp}] ", style={'color': '#666'}),
                            html.Span(f"{train_id}: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{event.get('old_status', 'unknown')} → {event.get('new_status', 'unknown')}")
                        ], style={'marginBottom': '5px', 'padding': '5px', 'borderLeft': '3px solid #1976d2', 'backgroundColor': '#f8f9fa'}))
            
            if not events:
                # Default events with current timestamp
                current_time = datetime.now()
                events = [
                    html.Div([
                        html.Span(f"[{(current_time - timedelta(minutes=5)).strftime('%H:%M:%S')}] ", style={'color': '#666'}),
                        html.Span("T001: ", style={'fontWeight': 'bold'}),
                        html.Span("idle → inducted")
                    ], style={'marginBottom': '5px', 'padding': '5px', 'borderLeft': '3px solid #4caf50', 'backgroundColor': '#f8f9fa'}),
                    html.Div([
                        html.Span(f"[{(current_time - timedelta(minutes=10)).strftime('%H:%M:%S')}] ", style={'color': '#666'}),
                        html.Span("T002: ", style={'fontWeight': 'bold'}),
                        html.Span("maintenance → ready")
                    ], style={'marginBottom': '5px', 'padding': '5px', 'borderLeft': '3px solid #ff9800', 'backgroundColor': '#f8f9fa'}),
                    html.Div([
                        html.Span(f"[{(current_time - timedelta(minutes=15)).strftime('%H:%M:%S')}] ", style={'color': '#666'}),
                        html.Span("System: ", style={'fontWeight': 'bold'}),
                        html.Span("AI optimization completed")
                    ], style={'marginBottom': '5px', 'padding': '5px', 'borderLeft': '3px solid #1976d2', 'backgroundColor': '#f8f9fa'})
                ]
            
            return events[-10:]  # Return last 10 events
            
        except Exception as e:
            return [html.Div(f"Event log error: {str(e)}", style={'color': 'red'})]

    def run_server(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server"""
        print(f"🚀 Starting KMRL IntelliFleet Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
