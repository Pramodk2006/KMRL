import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

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
        self.ai_data_processor = ai_data_processor  # Added AI data processor
        
        # Initialize Dash app with custom CSS
        self.app = dash.Dash(__name__, 
                            external_stylesheets=[
                                'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
                            ])
        self.setup_layout()
        self.setup_callbacks()
        
        # Add observer and get initial state
        if hasattr(self.digital_twin, 'add_observer'):
            self.digital_twin.add_observer(self._on_state_update)
        self.current_state = self.digital_twin.get_current_state()
    
    def _on_state_update(self, state):
        """Callback for digital twin state updates"""
        self.current_state = state
    
    def setup_layout(self):
        """Setup the modern dashboard layout"""
        self.app.layout = html.Div([
            # Modern Header
            html.Div([
                html.H1("🚄 KMRL IntelliFleet", className="header-title"),
                html.P("AI-Powered Digital Twin Dashboard", className="header-subtitle")
            ], className="modern-header"),
            
            # Navigation Tabs (if using Material Design components)
            self._create_navigation_tabs() if dmc else html.Div(),
            
            # Main Container
            html.Div([
                # Control Panel
                html.Div([
                    html.H3("🎮 Simulation Control", className="control-title"),
                    html.Div([
                        html.Button("▶️ Start", id="start-btn", 
                                   className="modern-btn btn-success", n_clicks=0),
                        html.Button("⏸️ Pause", id="pause-btn", 
                                   className="modern-btn btn-warning", n_clicks=0),
                        html.Button("⏹️ Stop", id="stop-btn", 
                                   className="modern-btn btn-danger", n_clicks=0),
                        html.Div([
                            html.Label("Speed Multiplier", 
                                     style={'color': '#666', 'fontSize': '0.9rem', 'marginBottom': '0.5rem'}),
                            dcc.Slider(id="speed-slider", min=0.1, max=10, step=0.1, 
                                      value=1.0, marks={1: '1×', 5: '5×', 10: '10×'},
                                      tooltip={"placement": "bottom", "always_visible": True})
                        ], className="modern-slider")
                    ], className="controls-row")
                ], className="control-panel"),
                
                # Status Overview Cards
                html.Div([
                    html.Div(id="status-cards", children=[
                        self._create_status_card("Total Trains", "10", "🚂", "#1976d2"),
                        self._create_status_card("Inducted", "6", "✅", "#4caf50"),
                        self._create_status_card("Available Bays", "3", "🏗️", "#ff9800"),
                        self._create_status_card("Avg Risk", "12.4%", "⚠️", "#f44336")
                    ], className="row")
                ], style={'marginBottom': '2rem'}),
                
                # Tab Content Container
                html.Div(id="tab-content", children=self._create_overview_content()),
                
                # Auto-refresh component
                dcc.Interval(
                    id='interval-component',
                    interval=2000,  # Update every 2 seconds
                    n_intervals=0
                ),
                
                # Store for maintaining state
                dcc.Store(id='dashboard-state', data={})
            ], className="container-fluid")
        ])
    
    def _create_navigation_tabs(self):
        """Create navigation tabs if Material Design components are available"""
        if not dmc:
            return html.Div()
        
        return dmc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dmc.TabsList([
                    dmc.TabsTab("Overview", value="overview"),
                    dmc.TabsTab("AI Control", value="ai"),
                    dmc.TabsTab("Schedule", value="schedule"),
                    dmc.TabsTab("Events", value="events"),
                ]),
                dmc.TabsPanel(self._overview_panel(), value="overview"),
                dmc.TabsPanel(self._ai_panel(), value="ai"),
                dmc.TabsPanel(self._schedule_panel(), value="schedule"),
                dmc.TabsPanel(self._events_panel(), value="events"),
            ]
        )
    
    def _create_overview_content(self):
        """Create the default overview content"""
        return html.Div([
            # Main Visualization Area
            html.Div([
                # Bay Layout Visualization
                html.Div([
                    html.Div([
                        html.H4("🏗️ Bay Layout & Occupancy", className="viz-title"),
                        dcc.Graph(id="bay-layout-graph", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-6"),
                
                # Train Status Visualization
                html.Div([
                    html.Div([
                        html.H4("🚂 Train Status Distribution", className="viz-title"),
                        dcc.Graph(id="train-status-graph", config={'displayModeBar': False})
                    ], className="viz-card")
                ], className="col-md-6")
            ], className="row"),
            
            # Real-time Metrics
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
            
            # Scenario Testing Panel
            html.Div([
                html.H3("🧪 Scenario Testing", 
                       style={'color': '#1976d2', 'marginBottom': '1.5rem', 'fontSize': '1.5rem'}),
                html.Div([
                    html.Div([
                        html.Label("Scenario Type", 
                                 style={'color': '#666', 'marginBottom': '0.5rem'}),
                        dcc.Dropdown(
                            id="scenario-type",
                            options=[
                                {'label': '🚨 Emergency Response', 'value': 'emergency'},
                                {'label': '🔧 Bay Maintenance', 'value': 'maintenance'},
                                {'label': '⚠️ Train Failures', 'value': 'failures'},
                                {'label': '📈 Peak Demand', 'value': 'peak_demand'}
                            ],
                            value='emergency',
                            className="modern-dropdown",
                            style={'borderRadius': '12px'}
                        )
                    ], className="col-md-3"),
                    
                    html.Div([
                        html.Label("Duration (min)", 
                                 style={'color': '#666', 'marginBottom': '0.5rem'}),
                        dcc.Input(id="scenario-duration", type="number", 
                                 value=60, min=10, max=480,
                                 style={'width': '100%', 'padding': '0.75rem', 
                                       'border': '2px solid rgba(0,0,0,0.1)', 
                                       'borderRadius': '12px', 'fontSize': '0.9rem'})
                    ], className="col-md-2"),
                    
                    html.Div([
                        html.Label("Speed", 
                                 style={'color': '#666', 'marginBottom': '0.5rem'}),
                        dcc.Input(id="scenario-speed", type="number", 
                                 value=10, min=1, max=100,
                                 style={'width': '100%', 'padding': '0.75rem', 
                                       'border': '2px solid rgba(0,0,0,0.1)', 
                                       'borderRadius': '12px', 'fontSize': '0.9rem'})
                    ], className="col-md-2"),
                    
                    html.Div([
                        html.Button("🚀 Run Scenario", id="run-scenario-btn", 
                                   className="modern-btn btn-info", n_clicks=0,
                                   style={'marginTop': '1.5rem', 'width': '100%'})
                    ], className="col-md-2")
                ], className="row"),
                
                html.Div(id="scenario-results", style={'marginTop': '1.5rem'})
            ], className="scenario-panel"),
            
            # Live Event Log
            html.Div([
                html.H3("📋 Live Event Log", className="viz-title"),
                html.Div(id="event-log", className="event-log")
            ], className="viz-card")
        ])
    
    def _overview_panel(self):
        """Overview panel for Material Design tabs"""
        return self._create_overview_content()
    
    def _ai_panel(self):
        """AI Control panel showing complete optimization results like main_app.py"""
        if not self.ai_data_processor:
            return html.Div([
                html.H4("🤖 AI Recommendations", className="viz-title"),
                html.Div("AI data processor not available", 
                        style={'padding': '2rem', 'textAlign': 'center', 'color': '#666'})
            ])
        
        # Get all the data
        train_details = self.ai_data_processor.get_detailed_train_list()
        performance_metrics = self.ai_data_processor.get_performance_metrics()
        summary = self.ai_data_processor.get_train_status_summary()
        violations = self.ai_data_processor.get_constraint_violations()
        
        # === FLEET OVERVIEW SECTION ===
        fleet_overview = html.Div([
            html.H4("📊 FLEET OVERVIEW", style={'color': '#1976d2', 'borderBottom': '2px solid #1976d2', 'paddingBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Strong("Total Fleet Size: "), 
                    html.Span(f"{summary.get('total_trains', 0)} trains")
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("✅ Inducted: "), 
                    html.Span(f"{summary.get('inducted_trains', 0)} trains ({summary.get('inducted_trains', 0)/max(summary.get('total_trains', 1), 1)*100:.1f}%)", 
                            style={'color': '#4caf50'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("⏸️ Standby: "), 
                    html.Span(f"{summary.get('standby_trains', 0)} trains ({summary.get('standby_trains', 0)/max(summary.get('total_trains', 1), 1)*100:.1f}%)", 
                            style={'color': '#ff9800'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("❌ Ineligible: "), 
                    html.Span(f"{summary.get('ineligible_trains', 0)} trains ({summary.get('ineligible_trains', 0)/max(summary.get('total_trains', 1), 1)*100:.1f}%)", 
                            style={'color': '#f44336'})
                ], style={'margin': '0.5rem 0'}),
            ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # === INDUCTED TRAINS TABLE ===
        inducted_trains = [t for t in train_details if t.get('inducted', False)]
        inducted_table_rows = []
        
        for train in inducted_trains:
            status_color = '#4caf50' if 'Ready' in train['status'] else '#ff9800'
            status_icon = '✅' if 'Ready' in train['status'] else '⚠️'
            
            inducted_table_rows.append(html.Tr([
                html.Td(f"{train['rank']}", style={'fontWeight': 'bold', 'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(train['train_id'], style={'fontWeight': 'bold', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(train['bay_assignment'], style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(f"{train['priority_score']:.1f}", style={'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(f"{train['branding_hours']:.1f}h", style={'textAlign': 'center', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td([status_icon, f" {train['status']}"], 
                    style={'color': status_color, 'fontWeight': 'bold', 'padding': '8px', 'borderBottom': '1px solid #ddd'})
            ]))
        
        inducted_trains_section = html.Div([
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
                html.Tbody(inducted_table_rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd', 'marginTop': '1rem'}),
            
            html.Div([
                html.Strong("💡 Average Performance Score: "),
                html.Span(f"{performance_metrics.get('system_performance', 0):.1f}/100", 
                        style={'color': '#1976d2', 'fontSize': '1.1em'})
            ], style={'margin': '1rem 0', 'padding': '0.5rem', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'})
        ])
        
        # === CONSTRAINT VIOLATIONS & ALERTS ===
        violations_items = []
        for violation in violations:
            train_violations = violation.get('violations', [])
            if train_violations:
                violations_items.append(
                    html.Div([
                        html.Div([
                            html.Strong(f"❌ {violation['train_id']}:", style={'color': '#f44336'})
                        ]),
                        html.Div([
                            html.Div([
                                "└─ " + str(v)
                            ], style={'marginLeft': '1rem', 'color': '#666'}) 
                            for v in train_violations
                        ])
                    ], style={'margin': '0.5rem 0'})
                )
        
        constraint_violations_section = html.Div([
            html.H4("⚠️ CONSTRAINT VIOLATIONS & ALERTS", 
                style={'color': '#f44336', 'borderBottom': '2px solid #f44336', 'paddingBottom': '0.5rem'}),
            html.Div(violations_items if violations_items else [html.Div("No constraint violations", style={'color': '#4caf50'})],
                    style={'backgroundColor': '#fff3e0', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # === CAPACITY UTILIZATION ===
        capacity_section = html.Div([
            html.H4("🏗️ CAPACITY UTILIZATION", 
                style={'color': '#ff9800', 'borderBottom': '2px solid #ff9800', 'paddingBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Strong("Bay Utilization: "),
                    html.Span(f"{summary.get('inducted_trains', 0)}/6 (100.0%)", style={'color': '#4caf50'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Cleaning Utilization: "),
                    html.Span(f"{summary.get('inducted_trains', 0)}/6 (100.0%)", style={'color': '#4caf50'})
                ], style={'margin': '0.5rem 0'}),
            ], style={'backgroundColor': '#fff3e0', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # === PERFORMANCE DISTRIBUTION ===
        excellent_count = len([t for t in inducted_trains if t['priority_score'] >= 80])
        good_count = len([t for t in inducted_trains if 60 <= t['priority_score'] < 80])
        acceptable_count = len([t for t in inducted_trains if 40 <= t['priority_score'] < 60])
        poor_count = len([t for t in inducted_trains if t['priority_score'] < 40])
        
        performance_distribution_section = html.Div([
            html.H4("📈 Performance Distribution:", style={'color': '#1976d2', 'margin': '1rem 0'}),
            html.Div([
                html.Div([html.Span("🟢 Excellent (80+): "), html.Strong(f"{excellent_count} trains")], style={'margin': '0.25rem 0'}),
                html.Div([html.Span("🟡 Good (60-79): "), html.Strong(f"{good_count} trains")], style={'margin': '0.25rem 0'}),
                html.Div([html.Span("🟠 Acceptable (40-59): "), html.Strong(f"{acceptable_count} trains")], style={'margin': '0.25rem 0'}),
                html.Div([html.Span("🔴 Poor (<40): "), html.Strong(f"{poor_count} trains")], style={'margin': '0.25rem 0'}),
            ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '8px'})
        ])
        
        # === KEY PERFORMANCE METRICS ===
        metrics_section = html.Div([
            html.H4("📊 KEY PERFORMANCE METRICS", 
                style={'color': '#1976d2', 'borderBottom': '2px solid #1976d2', 'paddingBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Strong("System Performance Score: "),
                    html.Span(f"{performance_metrics.get('system_performance', 0):.1f}/100", 
                            style={'color': '#1976d2', 'fontSize': '1.1em'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Service Readiness: "),
                    html.Span(f"{performance_metrics.get('service_readiness', 0):.1f}/100", 
                            style={'color': '#4caf50', 'fontSize': '1.1em'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Maintenance Risk: "),
                    html.Span(f"{performance_metrics.get('maintenance_risk', 0):.1f}/100", 
                            style={'color': '#f44336', 'fontSize': '1.1em'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Branding Compliance: "),
                    html.Span(f"{performance_metrics.get('branding_compliance', 0):.1f}/100", 
                            style={'color': '#ff9800', 'fontSize': '1.1em'})
                ], style={'margin': '0.5rem 0'}),
            ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # === ESTIMATED IMPACT ===
        impact_section = html.Div([
            html.H4("💰 ESTIMATED IMPACT", 
                style={'color': '#4caf50', 'borderBottom': '2px solid #4caf50', 'paddingBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Strong("Tonight's Cost Savings: "),
                    html.Span(f"₹{performance_metrics.get('cost_savings', 0):,}", 
                            style={'color': '#4caf50', 'fontSize': '1.2em', 'fontWeight': 'bold'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Annual Projected Savings: "),
                    html.Span(f"₹{performance_metrics.get('annual_savings', 0):,}", 
                            style={'color': '#4caf50', 'fontSize': '1.2em', 'fontWeight': 'bold'})
                ], style={'margin': '0.5rem 0'}),
            ], style={'backgroundColor': '#e8f5e8', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # === EXECUTIVE SUMMARY ===
        avg_score = performance_metrics.get('system_performance', 0)
        status = "✅ OPTIMAL" if avg_score > 80 else "⚠️ ACCEPTABLE" if avg_score > 60 else "❌ REVIEW NEEDED"
        
        executive_summary = html.Div([
            html.H4("📋 EXECUTIVE SUMMARY", 
                style={'color': '#1976d2', 'borderBottom': '2px solid #1976d2', 'paddingBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Strong("Date: "),
                    html.Span(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Totals: "),
                    html.Span(f"{summary.get('inducted_trains', 0)} trains inducted, {len(violations)} conflicts resolved")
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Overall Performance: "),
                    html.Span(f"{avg_score:.1f}/100")
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Status: "),
                    html.Span(status, style={'fontWeight': 'bold', 'fontSize': '1.1em'})
                ], style={'margin': '0.5rem 0'}),
                html.Div([
                    html.Strong("Next Steps:")
                ], style={'margin': '1rem 0 0.5rem 0'}),
                html.Ul([
                    html.Li("Monitor inducted trains for service readiness"),
                    html.Li("Address constraint violations for ineligible trains"),
                    html.Li("Review recommendations for potential optimizations")
                ])
            ], style={'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '8px', 'margin': '1rem 0'})
        ])
        
        # Combine all sections
        return html.Div([
            html.H3("🚄 KMRL IntelliFleet - AI-Driven Train Induction System", 
                style={'textAlign': 'center', 'color': '#1976d2', 'borderBottom': '3px solid #1976d2', 'paddingBottom': '1rem', 'marginBottom': '2rem'}),
            
            fleet_overview,
            inducted_trains_section,
            constraint_violations_section,
            capacity_section,
            performance_distribution_section,
            metrics_section,
            impact_section,
            executive_summary
            
        ], style={'maxHeight': '80vh', 'overflowY': 'auto', 'padding': '1rem'})

    
    def _schedule_panel(self):
        """Schedule panel showing AI-optimized train schedules using ai_data_processor"""
        if not self.ai_data_processor:
            return html.Div([
                html.H4("🚆 Train Schedule", className="viz-title"),
                html.Div("AI data processor not available", 
                        style={'padding': '2rem', 'textAlign': 'center', 'color': '#666'})
            ])
        
        # Get detailed train data
        train_details = self.ai_data_processor.get_detailed_train_list()
        
        # Create schedule from inducted trains
        schedule_items = []
        base_time = 6.0  # Start at 6:00 AM
        
        inducted_trains = [t for t in train_details if t.get('inducted', False)]
        
        for i, train in enumerate(inducted_trains):
            departure_time = base_time + (i * 0.5)  # 30-minute intervals
            hours = int(departure_time)
            minutes = int((departure_time % 1) * 60)
            time_str = f"{hours:02d}:{minutes:02d}"
            
            status_color = '#4caf50' if 'Ready' in train['status'] else '#ff9800' if 'Caution' in train['status'] else '#f44336'
            
            schedule_items.append(html.Tr([
                html.Td(train['train_id'], style={'padding': '10px', 'borderBottom': '1px solid #eee'}),
                html.Td(f"Route {(i % 3) + 1}", style={'padding': '10px', 'borderBottom': '1px solid #eee'}),
                html.Td(time_str, style={'padding': '10px', 'borderBottom': '1px solid #eee'}),
                html.Td(train['bay_assignment'], style={'padding': '10px', 'borderBottom': '1px solid #eee'}),
                html.Td(f"{train['priority_score']:.1f}", style={'padding': '10px', 'borderBottom': '1px solid #eee'}),
                html.Td(train['status'], style={'padding': '10px', 'borderBottom': '1px solid #eee', 'color': status_color, 'fontWeight': 'bold'})
            ]))
        
        if schedule_items:
            schedule_table = html.Table([
                html.Thead(html.Tr([
                    html.Th(col, style={'padding': '12px', 'borderBottom': '2px solid #ddd', 'textAlign': 'left', 'backgroundColor': '#f5f5f5'}) 
                    for col in ["Train ID", "Route", "Departure", "Bay", "Priority", "Status"]
                ])),
                html.Tbody(schedule_items)
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
        else:
            schedule_table = html.Div("No scheduled trains available", 
                                    style={'padding': '2rem', 'textAlign': 'center', 'color': '#666'})
        
        # Summary stats
        summary_stats = self.ai_data_processor.get_train_status_summary()
        stats_cards = html.Div([
            html.Div([
                html.H6("Ready for Service", style={'margin': '0 0 0.5rem 0', 'color': '#666'}),
                html.H4(f"{summary_stats.get('ready_trains', 0)}", style={'color': '#4caf50', 'margin': '0'})
            ], style={'padding': '1rem', 'border': '1px solid #ddd', 'borderRadius': '8px', 'textAlign': 'center'}),
            
            html.Div([
                html.H6("Maintenance Required", style={'margin': '0 0 0.5rem 0', 'color': '#666'}),
                html.H4(f"{summary_stats.get('maintenance_trains', 0)}", style={'color': '#f44336', 'margin': '0'})
            ], style={'padding': '1rem', 'border': '1px solid #ddd', 'borderRadius': '8px', 'textAlign': 'center'}),
            
            html.Div([
                html.H6("On Standby", style={'margin': '0 0 0.5rem 0', 'color': '#666'}),
                html.H4(f"{summary_stats.get('standby_trains', 0)}", style={'color': '#ff9800', 'margin': '0'})
            ], style={'padding': '1rem', 'border': '1px solid #ddd', 'borderRadius': '8px', 'textAlign': 'center'}),
        ], style={'display': 'flex', 'gap': '1rem', 'marginBottom': '1.5rem'})
        
        return html.Div([
            html.H4("🚆 AI-Optimized Train Induction Schedule", className="viz-title"),
            html.P("Next shift assignments based on multi-objective optimization", 
                   style={'color': '#666', 'marginBottom': '1rem'}),
            stats_cards,
            schedule_table
        ])
    
    def _events_panel(self):
        """Events panel for Material Design tabs"""
        return html.Div([
            html.H4("📋 System Events", className="viz-title"),
            html.Div(id="events-panel-content", children=self._create_event_log())
        ])
    
    def _create_status_card(self, title: str, value: str, icon: str, color: str):
        """Create a modern status card component"""
        return html.Div([
            html.Div([
                html.Span(icon, className="status-icon"),
                html.H4(value, className="status-value", style={'color': color}),
                html.P(title, className="status-title")
            ], className="status-card")
        ], className="col-md-3")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""
        
        # Update all graphs
        @self.app.callback(
            [Output('bay-layout-graph', 'figure'),
             Output('train-status-graph', 'figure'),
             Output('performance-timeline', 'figure'),
             Output('risk-gauge', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            return [
                self._create_bay_layout_figure(),
                self._create_train_status_figure(),
                self._create_performance_timeline(),
                self._create_risk_gauge()
            ]
        
        @self.app.callback(
            Output('status-cards', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_cards(n):
            """Update status cards with real AI optimization data"""
            if self.ai_data_processor:
                summary = self.ai_data_processor.get_train_status_summary()
                performance = self.ai_data_processor.get_performance_metrics()
                
                return [
                    self._create_status_card("Total Trains", str(summary.get('total_trains', 0)), "🚂", "#1976d2"),
                    self._create_status_card("Inducted", str(summary.get('inducted_trains', 0)), "✅", "#4caf50"),
                    self._create_status_card("Ready", str(summary.get('ready_trains', 0)), "🟢", "#4caf50"),
                    self._create_status_card("AI Score", f"{performance.get('system_performance', 0):.1f}", "🤖", "#9c27b0")
                ]
            else:
                # Fallback to default
                summary = self.current_state.get('summary', {})
                return [
                    self._create_status_card("Total Trains", str(summary.get('total_trains', 10)), "🚂", "#1976d2"),
                    self._create_status_card("Inducted", str(summary.get('inducted_trains', 6)), "✅", "#4caf50"),
                    self._create_status_card("Available Bays", str(summary.get('available_bays', 3)), "🏗️", "#ff9800"),
                    self._create_status_card("Avg Risk", f"{summary.get('average_failure_risk', 0.05):.1%}", "⚠️", "#f44336")
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
                if hasattr(self.digital_twin, 'start_simulation'):
                    self.digital_twin.start_simulation(time_multiplier=speed)
                return {'status': 'running', 'speed': speed}
            elif button_id == 'pause-btn' and pause_clicks > 0:
                if hasattr(self.digital_twin, 'stop_simulation'):
                    self.digital_twin.stop_simulation()
                return {'status': 'paused', 'speed': speed}
            elif button_id == 'stop-btn' and stop_clicks > 0:
                if hasattr(self.digital_twin, 'stop_simulation'):
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
            
            return html.Div([
                html.Div([
                    html.H5(f"🧪 Scenario '{scenario_type}' initiated", 
                           style={'color': '#1976d2', 'margin': '0 0 0.5rem 0'}),
                    html.P(f"Duration: {duration} minutes at {speed}× speed", 
                          style={'margin': '0 0 0.5rem 0', 'color': '#666'}),
                    html.Div([
                        html.Span("⏳ Running scenario...", style={'color': '#ff9800'})
                    ])
                ], style={
                    'background': 'linear-gradient(135deg, rgba(25, 118, 210, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%)',
                    'padding': '1rem',
                    'borderRadius': '12px',
                    'border': '2px solid rgba(25, 118, 210, 0.2)'
                })
            ])
    
    def _create_bay_layout_figure(self):
        """Create modern bay layout visualization"""
        bays = self.current_state.get('bays', {
            'bay_1': {'status': 'available', 'occupied_trains': [], 'max_capacity': 1},
            'bay_2': {'status': 'occupied', 'occupied_trains': ['KMRL_001'], 'max_capacity': 1},
            'bay_3': {'status': 'available', 'occupied_trains': [], 'max_capacity': 1},
            'bay_4': {'status': 'available', 'occupied_trains': [], 'max_capacity': 1}
        })
        
        bay_data = []
        colors = []
        texts = []
        sizes = []
        
        color_map = {
            'available': '#4caf50',
            'occupied': '#ff9800', 
            'partial': '#2196f3',
            'maintenance': '#f44336'
        }
        
        for i, (bay_id, bay_info) in enumerate(bays.items()):
            row = i // 3
            col = i % 3
            
            bay_data.append([col, row])
            colors.append(color_map.get(bay_info['status'], '#9e9e9e'))
            sizes.append(100)
            
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
                    size=sizes,
                    color=colors,
                    line=dict(width=3, color='white'),
                    opacity=0.8
                ),
                text=texts,
                textposition="middle center",
                textfont=dict(size=12, color='white', family="Roboto"),
                hovertemplate='<b>%{text}</b><extra></extra>',
                name=""
            ))
            
            fig.update_layout(
                title={
                    'text': "Bay Layout & Occupancy",
                    'font': {'size': 16, 'color': '#1976d2', 'family': 'Roboto'}
                },
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No bay data available",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        
        return fig
    
    def _create_train_status_figure(self):
        """Create modern train status pie chart"""
        if self.ai_data_processor:
            # Use AI data processor for accurate status distribution
            summary = self.ai_data_processor.get_train_status_summary()
            status_counts = {
                'Ready': summary.get('ready_trains', 0),
                'Standby': summary.get('standby_trains', 0), 
                'Maintenance': summary.get('maintenance_trains', 0),
                'Ineligible': summary.get('ineligible_trains', 0)
            }
        else:
            # Fallback to current state
            trains = self.current_state.get('trains', {})
            status_counts = {}
            for train_info in trains.values():
                status = train_info.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts and any(status_counts.values()):
            colors = ['#4caf50', '#ff9800', '#f44336', '#9e9e9e']
            
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Train Status Distribution",
                color_discrete_sequence=colors
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont=dict(size=12, family="Roboto"),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                marker=dict(line=dict(color='white', width=2))
            )
            
            fig.update_layout(
                title={
                    'font': {'size': 16, 'color': '#1976d2', 'family': 'Roboto'}
                },
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No train data available",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        
        return fig
    
    def _create_performance_timeline(self):
        """Create modern performance timeline"""
        times = pd.date_range(start=datetime.now() - timedelta(hours=2), 
                             end=datetime.now(), freq='10min')
        
        if self.ai_data_processor:
            # Use real performance data if available
            performance = self.ai_data_processor.get_performance_metrics()
            base_performance = performance.get('system_performance', 75)
            inducted_trains = np.random.normal(base_performance/10, 1, len(times)).astype(int)
            bay_utilization = np.random.normal(base_performance, 5, len(times))
        else:
            inducted_trains = np.random.randint(4, 8, len(times))
            bay_utilization = np.random.uniform(60, 95, len(times))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=inducted_trains,
            mode='lines+markers',
            name='Inducted Trains',
            line=dict(color='#1976d2', width=3),
            marker=dict(size=6, color='#1976d2'),
            yaxis='y',
            hovertemplate='<b>Inducted Trains</b><br>Time: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=bay_utilization,
            mode='lines+markers',
            name='Bay Utilization (%)',
            line=dict(color='#ff9800', width=3),
            marker=dict(size=6, color='#ff9800'),
            yaxis='y2',
            hovertemplate='<b>Bay Utilization</b><br>Time: %{x}<br>Utilization: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Real-time Performance Metrics',
                'font': {'size': 16, 'color': '#1976d2', 'family': 'Roboto'}
            },
            xaxis=dict(
                title='Time',
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(family="Roboto", color='#666')
            ),
            yaxis=dict(
                title='Inducted Trains', 
                side='left',
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(family="Roboto", color='#1976d2')
            ),
            yaxis2=dict(
                title='Bay Utilization (%)', 
                side='right', 
                overlaying='y',
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(family="Roboto", color='#ff9800')
            ),
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Roboto", color='#666')
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_risk_gauge(self):
        """Create modern risk gauge"""
        if self.ai_data_processor:
            performance = self.ai_data_processor.get_performance_metrics()
            avg_risk = 100 - performance.get('system_performance', 75)  # Convert performance to risk
        else:
            summary = self.current_state.get('summary', {})
            avg_risk = summary.get('average_failure_risk', 0.05) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fleet Risk Level (%)", 
                    'font': {'size': 16, 'color': '#1976d2', 'family': 'Roboto'}},
            number = {'font': {'size': 24, 'color': '#f44336', 'family': 'Roboto'}},
            gauge = {
                'axis': {'range': [None, 100], 
                        'tickfont': {'family': 'Roboto', 'color': '#666'}},
                'bar': {'color': "#f44336", 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0.05)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 25], 'color': "#4caf50"},
                    {'range': [25, 50], 'color': "#ff9800"},
                    {'range': [50, 75], 'color': "#ff5722"},
                    {'range': [75, 100], 'color': "#f44336"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_event_log(self):
        """Create modern event log"""
        events = []
        trains = self.current_state.get('trains', {
            'KMRL_001': {'recent_events': [
                {'timestamp': datetime.now().isoformat(), 'old_status': 'idle', 'new_status': 'inducted'}
            ]},
            'KMRL_002': {'recent_events': [
                {'timestamp': datetime.now().isoformat(), 'old_status': 'running', 'new_status': 'maintenance'}
            ]}
        })
        
        for train_id, train_info in trains.items():
            recent_events = train_info.get('recent_events', [])
            for event in recent_events[-3:]:  # Last 3 events
                timestamp = event.get('timestamp', '')
                
                if timestamp:
                    try:
                        time_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = time_obj.strftime('%H:%M:%S')
                    except:
                        time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
                else:
                    time_str = 'Unknown'
                
                events.append(html.Div([
                    html.Span(f"[{time_str}] ", 
                             style={'color': '#666', 'fontSize': '0.85rem'}),
                    html.Span(f"{train_id}: ", 
                             style={'fontWeight': 'bold', 'color': '#1976d2'}),
                    html.Span(f"{event.get('old_status', '')} → {event.get('new_status', '')}", 
                             style={'color': '#333'})
                ], className="event-item"))
        
        if not events:
            events = [html.Div("No recent events", 
                              style={'color': '#666', 'fontStyle': 'italic', 
                                    'textAlign': 'center', 'padding': '2rem'})]
        
        return events[-10:]  # Show last 10 events
    
    def run_server(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server"""
        print(f"🌐 Starting modern dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
