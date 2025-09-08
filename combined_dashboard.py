"""
Combined Dashboard with Animated Simulation + Classical UI Navigation Tabs
Integrates AnimatedTrainDashboard and InteractiveWebDashboard into one interface
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import logging

# Import the existing dashboard classes
from animated_web_dashboard import AnimatedTrainDashboard
from src.enhanced_web_dashboard import InteractiveWebDashboard

logger = logging.getLogger(__name__)

class CombinedKMRLDashboard:
    """Combined Dashboard with Animated Simulation + Classical UI Tabs"""

    def __init__(self, digital_twin, monitor=None, iot_simulator=None, cv_system=None,
                 ai_optimizer=None, constraint_engine=None, ai_dashboard=None, ai_data_processor=None):
        
        self.digital_twin = digital_twin
        self.monitor = monitor
        self.iot_simulator = iot_simulator
        self.cv_system = cv_system
        self.ai_optimizer = ai_optimizer
        self.constraint_engine = constraint_engine
        self.ai_dashboard = ai_dashboard
        self.ai_data_processor = ai_data_processor

        # Initialize dashboard components (no separate servers)
        self.animated_dashboard = AnimatedTrainDashboard(
            digital_twin, monitor, iot_simulator, cv_system,
            ai_optimizer=ai_optimizer, constraint_engine=constraint_engine,
            ai_dashboard=ai_dashboard, ai_data_processor=ai_data_processor
        )

        self.classic_dashboard = InteractiveWebDashboard(
            digital_twin, monitor, iot_simulator, cv_system,
            ai_optimizer=ai_optimizer, constraint_engine=constraint_engine,
            ai_dashboard=ai_dashboard, ai_data_processor=ai_data_processor
        )

        # Create combined Dash app
        external_stylesheets = [
            'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css',
            'https://codepen.io/chriddyp/pen/bWLwgP.css'
        ]

        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = "KMRL IntelliFleet - Combined Dashboard"
        
        # Build layout and register callbacks
        self.app.layout = self._build_combined_layout()
        self._register_combined_callbacks()

    def _build_combined_layout(self):
        """Creates main layout with navigation tabs and content container"""
        return html.Div([
            # Header Section
            html.Div([
                html.Div([
                    html.H1("🚄 KMRL IntelliFleet Dashboard", 
                           style={'color': '#1976d2', 'marginBottom': '5px', 'fontSize': '2.5rem'}),
                    html.P("AI-Powered Digital Twin System with Live Animation", 
                          style={'color': '#666', 'fontSize': '1.1rem', 'marginBottom': '20px'})
                ], style={'textAlign': 'center', 'padding': '20px'})
            ], style={'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #dee2e6'}),

            # Navigation Tabs
            html.Div([
                dcc.Tabs(
                    id='main-dashboard-tabs',
                    value='animated',
                    children=[
                        dcc.Tab(
                            label='🎬 Live Simulation', 
                            value='animated',
                            style={
                                'padding': '12px 24px',
                                'fontWeight': 'bold',
                                'fontSize': '1.1rem'
                            },
                            selected_style={
                                'backgroundColor': '#1976d2',
                                'color': 'white',
                                'borderTop': '3px solid #0d47a1'
                            }
                        ),
                        dcc.Tab(
                            label='📊 Analytics Dashboard', 
                            value='classic',
                            style={
                                'padding': '12px 24px',
                                'fontWeight': 'bold',
                                'fontSize': '1.1rem'
                            },
                            selected_style={
                                'backgroundColor': '#1976d2',
                                'color': 'white',
                                'borderTop': '3px solid #0d47a1'
                            }
                        )
                    ],
                    style={'marginBottom': '0px'}
                )
            ], style={'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

            # Main Content Container
            html.Div(id='main-tab-content', style={'minHeight': '80vh'})

        ], style={'backgroundColor': '#f5f5f5'})

    def _register_combined_callbacks(self):
        """Register callbacks for tab switching and dashboard updates"""

        # Main tab switching callback
        @self.app.callback(
            Output('main-tab-content', 'children'),
            Input('main-dashboard-tabs', 'value')
        )
        def render_main_tab_content(selected_tab):
            if selected_tab == 'animated':
                return html.Div([
                    self.animated_dashboard._create_animated_layout()
                ], style={'padding': '20px'})
            elif selected_tab == 'classic':
                return html.Div([
                    self.classic_dashboard._create_main_layout()
                ], style={'padding': '20px'})
            else:
                return html.Div("Unknown tab selected", style={'color': 'red', 'padding': '20px'})

        # === ANIMATED DASHBOARD CALLBACKS ===
        
        # Animated train map update
        @self.app.callback(
            Output('animated-train-map', 'figure'),
            Input('animation-interval', 'n_intervals'),
            State('animation-state', 'data'),
            prevent_initial_call=False
        )
        def update_animated_train_map(n_intervals, animation_state):
            try:
                if animation_state and animation_state.get('paused', False):
                    # If paused, return current map without updates
                    return self.animated_dashboard._create_animated_train_map()
                return self.animated_dashboard._create_animated_train_map()
            except Exception as e:
                logger.error(f"Error updating animated train map: {e}")
                return self.animated_dashboard._create_empty_map(f"Map update error: {str(e)}")

        # Live timestamp update
        @self.app.callback(
            Output('live-timestamp', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_live_timestamp(n):
            from datetime import datetime
            return datetime.now().strftime('%H:%M:%S')

        # Active trains count
        @self.app.callback(
            Output('active-trains-count', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_active_trains_count(n):
            try:
                current_state = self.digital_twin.get_current_state()
                trains = current_state.get('trains', {})
                return str(len(trains))
            except:
                return "0"

        # Moving trains count
        @self.app.callback(
            Output('moving-trains-count', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_moving_trains_count(n):
            try:
                current_state = self.digital_twin.get_current_state()
                trains = current_state.get('trains', {})
                moving_count = sum(1 for train in trains.values() if train.get('status') == 'moving')
                return str(moving_count)
            except:
                return "0"

        # Movement progress bars
        @self.app.callback(
            Output('movement-progress-bars', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_movement_progress(n):
            return self.animated_dashboard.get_movement_progress_bars()

        # Live bay status
        @self.app.callback(
            Output('live-bay-status', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_live_bay_status(n):
            return self.animated_dashboard.get_live_bay_status()

        # Movement event feed
        @self.app.callback(
            Output('movement-event-feed', 'children'),
            Input('animation-interval', 'n_intervals')
        )
        def update_movement_event_feed(n):
            return self.animated_dashboard.get_movement_event_feed()

        # Current animation speed display
        @self.app.callback(
            Output('current-speed', 'children'),
            Input('animation-speed-slider', 'value')
        )
        def update_current_speed_display(speed):
            return f"{speed}×"

        # === CLASSICAL DASHBOARD CALLBACKS ===

        # Status cards for classical dashboard
        @self.app.callback(
            Output('status-cards', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_status_cards(n):
            try:
                if self.ai_data_processor:
                    summary = self.ai_data_processor.get_train_status_summary()
                    performance = self.ai_data_processor.get_performance_metrics()
                    
                    return [
                        self.classic_dashboard._create_status_card(
                            "Total Trains", summary.get('total_trains', 0), "🚂", "#1976d2"
                        ),
                        self.classic_dashboard._create_status_card(
                            "Inducted", summary.get('inducted_trains', 0), "✅", "#4caf50"
                        ),
                        self.classic_dashboard._create_status_card(
                            "System Performance", f"{performance.get('system_performance', 0):.1f}%", "📈", "#ff9800"
                        ),
                        self.classic_dashboard._create_status_card(
                            "Cost Savings", f"₹{performance.get('cost_savings', 0):,}", "💰", "#9c27b0"
                        )
                    ]
                return []
            except Exception as e:
                return [html.Div(f"Error loading status: {e}", style={'color': 'red'})]

        # Bay layout graph
        @self.app.callback(
            Output('bay-layout-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_bay_layout(n):
            return self.classic_dashboard._create_bay_layout_figure()

        # Train status graph
        @self.app.callback(
            Output('train-status-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_train_status(n):
            return self.classic_dashboard._create_train_status_figure()

        # Performance timeline
        @self.app.callback(
            Output('performance-timeline', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_performance_timeline(n):
            return self.classic_dashboard._create_performance_timeline()

        # Risk gauge
        @self.app.callback(
            Output('risk-gauge', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_risk_gauge(n):
            return self.classic_dashboard._create_risk_gauge()

        # AI summary section
        @self.app.callback(
            Output('ai-summary-section', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_ai_summary_section(n):
            try:
                if not self.ai_data_processor:
                    return html.Div("AI Data Processor not available", style={'color': 'red'})

                summary = self.ai_data_processor.get_train_status_summary()
                train_details = self.ai_data_processor.get_detailed_train_list()
                performance = self.ai_data_processor.get_performance_metrics()
                violations = self.ai_data_processor.get_constraint_violations()

                # Get inducted trains only
                inducted_trains = [t for t in train_details if t.get('inducted', False)]

                ai_summary_layout = html.Div([
                    # Fleet Overview Section
                    html.Div([
                        html.H3("🤖 AI Fleet Overview", style={'color': '#1976d2', 'marginBottom': '20px'}),
                        html.Div([
                            html.Div([
                                html.H4(f"{summary.get('total_trains', 0)}", style={'margin': '0', 'color': '#1976d2'}),
                                html.P("Total Trains", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center"),
                            html.Div([
                                html.H4(f"{summary.get('inducted_trains', 0)}", style={'margin': '0', 'color': '#4caf50'}),
                                html.P("Inducted", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center"),
                            html.Div([
                                html.H4(f"{summary.get('ready_trains', 0)}", style={'margin': '0', 'color': '#2196f3'}),
                                html.P("Ready", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center"),
                            html.Div([
                                html.H4(f"{summary.get('maintenance_trains', 0)}", style={'margin': '0', 'color': '#ff9800'}),
                                html.P("Maintenance", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center"),
                            html.Div([
                                html.H4(f"{summary.get('standby_trains', 0)}", style={'margin': '0', 'color': '#9e9e9e'}),
                                html.P("Standby", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center"),
                            html.Div([
                                html.H4(f"{summary.get('ineligible_trains', 0)}", style={'margin': '0', 'color': '#f44336'}),
                                html.P("Ineligible", style={'margin': '0', 'color': '#666'})
                            ], className="col-md-2 text-center")
                        ], className="row")
                    ], className="card p-4 mb-4"),

                    # Inducted Trains Details
                    html.Div([
                        html.H4("📋 Inducted Trains Details", style={'color': '#1976d2', 'marginBottom': '15px'}),
                        html.Div([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Rank", style={'padding': '8px'}),
                                        html.Th("Train ID", style={'padding': '8px'}),
                                        html.Th("Bay Assignment", style={'padding': '8px'}),
                                        html.Th("Status", style={'padding': '8px'}),
                                        html.Th("Priority Score", style={'padding': '8px'})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(str(train['rank']), style={'padding': '8px'}),
                                        html.Td(train['train_id'], style={'padding': '8px', 'fontWeight': 'bold'}),
                                        html.Td(train['bay_assignment'], style={'padding': '8px'}),
                                        html.Td(train['status'], style={
                                            'padding': '8px',
                                            'color': '#4caf50' if 'Ready' in train['status'] else '#ff9800'
                                        }),
                                        html.Td(f"{train['priority_score']:.1f}", style={'padding': '8px'})
                                    ]) for train in inducted_trains[:8]  # Show top 8
                                ])
                            ], className="table table-striped", style={'fontSize': '0.9rem'})
                        ], style={'maxHeight': '300px', 'overflowY': 'auto'})
                    ], className="card p-4 mb-4") if inducted_trains else html.Div(),

                    # Performance Metrics
                    html.Div([
                        html.H4("📈 Performance Metrics", style={'color': '#1976d2', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.H5(f"{performance.get('system_performance', 0):.1f}/100", style={'color': '#4caf50'}),
                                html.P("System Performance", style={'color': '#666'})
                            ], className="col-md-3 text-center"),
                            html.Div([
                                html.H5(f"{performance.get('service_readiness', 0):.1f}%", style={'color': '#2196f3'}),
                                html.P("Service Readiness", style={'color': '#666'})
                            ], className="col-md-3 text-center"),
                            html.Div([
                                html.H5(f"₹{performance.get('cost_savings', 0):,}", style={'color': '#9c27b0'}),
                                html.P("Cost Savings", style={'color': '#666'})
                            ], className="col-md-3 text-center"),
                            html.Div([
                                html.H5(f"₹{performance.get('annual_savings', 0):,}", style={'color': '#ff9800'}),
                                html.P("Annual Savings", style={'color': '#666'})
                            ], className="col-md-3 text-center")
                        ], className="row")
                    ], className="card p-4 mb-4"),

                    # Constraint Violations (if any)
                    html.Div([
                        html.H4("⚠️ Constraint Violations", style={'color': '#f44336', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.Strong(f"{violation['train_id']}: ", style={'color': '#f44336'}),
                                html.Span(", ".join(violation['violations'][:2]), style={'color': '#666'})
                            ], style={'marginBottom': '8px'}) for violation in violations[:5]
                        ] if violations else [html.P("No violations detected", style={'color': '#4caf50'})])
                    ], className="card p-4") if violations else html.Div()
                ])

                return ai_summary_layout

            except Exception as e:
                return html.Div(f"Error loading AI summary: {e}", style={'color': 'red'})

        # Event log
        @self.app.callback(
            Output('event-log', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_event_log(n):
            return self.classic_dashboard._create_event_log()

    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the combined dashboard server"""
        print(f"🚀 Starting KMRL Combined Dashboard at http://{host}:{port}")
        print("📱 Features:")
        print("  🎬 Animated Simulation Tab - Live train tracking and movement")
        print("  📊 Analytics Dashboard Tab - AI insights and performance metrics")
        self.app.run(host=host, port=port, debug=debug)