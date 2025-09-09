"""
Modern Combined KMRL Dashboard with Enhanced Live Simulation Layout
Features professional design, improved user experience, and responsive layout
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import logging
from datetime import datetime

# Import the enhanced dashboard classes
from enhanced_animated_dashboard import EnhancedAnimatedTrainDashboard
from src.enhanced_web_dashboard import InteractiveWebDashboard

logger = logging.getLogger(__name__)

class ModernCombinedKMRLDashboard:
    """Modern Combined Dashboard with Enhanced Live Simulation and Analytics"""

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

        # Initialize enhanced dashboard components
        self.enhanced_animated_dashboard = EnhancedAnimatedTrainDashboard(
            digital_twin, monitor, iot_simulator, cv_system,
            ai_optimizer=ai_optimizer, constraint_engine=constraint_engine,
            ai_dashboard=ai_dashboard, ai_data_processor=ai_data_processor
        )

        self.classic_dashboard = InteractiveWebDashboard(
            digital_twin, monitor, iot_simulator, cv_system,
            ai_optimizer=ai_optimizer, constraint_engine=constraint_engine,
            ai_dashboard=ai_dashboard, ai_data_processor=ai_data_processor
        )

        # Create modern Dash app with enhanced styling
        external_stylesheets = [
            'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.2.3/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
            'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
        ]

        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = "KMRL IntelliFleet - Modern Dashboard"

        # Add modern CSS styling
        self.app.index_string = self._get_enhanced_index_template()
        
        # Build layout and register callbacks
        self.app.layout = self._build_modern_layout()
        self._register_enhanced_callbacks()

    def _get_enhanced_index_template(self):
        """Get enhanced HTML template with modern CSS"""
        return '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    /* Modern CSS Variables */
                    :root {
                        --primary-color: #1976d2;
                        --primary-dark: #0d47a1;
                        --secondary-color: #4caf50;
                        --accent-color: #ff9800;
                        --error-color: #f44336;
                        --warning-color: #ff9800;
                        --info-color: #2196f3;
                        --success-color: #4caf50;
                        --background-color: #f5f7fa;
                        --surface-color: #ffffff;
                        --text-primary: #212121;
                        --text-secondary: #666666;
                        --border-color: #e0e0e0;
                        --shadow-light: 0 2px 8px rgba(0,0,0,0.1);
                        --shadow-medium: 0 4px 16px rgba(0,0,0,0.15);
                        --shadow-heavy: 0 8px 32px rgba(0,0,0,0.2);
                        --border-radius: 12px;
                        --border-radius-small: 8px;
                        --transition-fast: 0.2s ease;
                        --transition-normal: 0.3s ease;
                        --transition-slow: 0.5s ease;
                    }

                    /* Global Styles */
                    * {
                        box-sizing: border-box;
                    }

                    body {
                        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background-color: var(--background-color);
                        color: var(--text-primary);
                        margin: 0;
                        padding: 0;
                        line-height: 1.6;
                    }

                    /* Panel Cards */
                    .panel-card {
                        background: var(--surface-color);
                        border-radius: var(--border-radius);
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                        box-shadow: var(--shadow-light);
                        border: 1px solid var(--border-color);
                        transition: box-shadow var(--transition-normal);
                    }

                    .panel-card:hover {
                        box-shadow: var(--shadow-medium);
                    }

                    /* Control Buttons */
                    .control-btn {
                        padding: 0.75rem 1.5rem;
                        border: none;
                        border-radius: var(--border-radius-small);
                        font-weight: 500;
                        font-size: 0.9rem;
                        cursor: pointer;
                        transition: all var(--transition-fast);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        text-decoration: none;
                    }

                    .control-btn.primary {
                        background: var(--primary-color);
                        color: white;
                    }

                    .control-btn.primary:hover {
                        background: var(--primary-dark);
                        transform: translateY(-2px);
                        box-shadow: var(--shadow-medium);
                    }

                    .control-btn.secondary {
                        background: var(--accent-color);
                        color: white;
                    }

                    .control-btn.secondary:hover {
                        background: #f57c00;
                        transform: translateY(-2px);
                        box-shadow: var(--shadow-medium);
                    }

                    .control-btn.danger {
                        background: var(--error-color);
                        color: white;
                    }

                    .control-btn.danger:hover {
                        background: #d32f2f;
                        transform: translateY(-2px);
                        box-shadow: var(--shadow-medium);
                    }

                    /* Control Section */
                    .control-section {
                        margin-bottom: 1.5rem;
                        padding-bottom: 1.5rem;
                        border-bottom: 1px solid var(--border-color);
                    }

                    .control-section:last-child {
                        border-bottom: none;
                        margin-bottom: 0;
                        padding-bottom: 0;
                    }

                    /* Stat Cards */
                    .stat-card {
                        background: var(--surface-color);
                        border-radius: var(--border-radius-small);
                        padding: 1rem;
                        text-align: center;
                        border: 1px solid var(--border-color);
                        transition: all var(--transition-fast);
                    }

                    .stat-card:hover {
                        transform: translateY(-2px);
                        box-shadow: var(--shadow-medium);
                    }

                    /* Health Status */
                    .health-item {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 0.75rem 0;
                        border-bottom: 1px solid #f0f0f0;
                    }

                    .health-item:last-child {
                        border-bottom: none;
                    }

                    .health-status {
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.8rem;
                        font-weight: 500;
                    }

                    .health-status.good {
                        background: #e8f5e8;
                        color: var(--success-color);
                    }

                    .health-status.warning {
                        background: #fff3e0;
                        color: var(--warning-color);
                    }

                    .health-status.error {
                        background: #ffebee;
                        color: var(--error-color);
                    }

                    /* Legend */
                    .legend-item {
                        display: flex;
                        align-items: center;
                    }

                    /* Event Feed */
                    .event-item {
                        margin-bottom: 0.75rem;
                        padding: 0.75rem;
                        border-radius: var(--border-radius-small);
                        border-left: 4px solid var(--primary-color);
                        background: var(--surface-color);
                        transition: all var(--transition-fast);
                    }

                    .event-item:hover {
                        transform: translateX(4px);
                        box-shadow: var(--shadow-light);
                    }

                    .event-item.running {
                        border-left-color: var(--success-color);
                        background: #f8fff8;
                    }

                    .event-item.inducted {
                        border-left-color: var(--info-color);
                        background: #f3f8ff;
                    }

                    .event-item.system {
                        border-left-color: var(--accent-color);
                        background: #fffbf0;
                    }

                    .event-time {
                        font-family: 'Courier New', monospace;
                        font-size: 0.8rem;
                        color: var(--text-secondary);
                        margin-right: 0.5rem;
                    }

                    .event-icon {
                        margin-right: 0.5rem;
                    }

                    .event-train {
                        font-weight: 600;
                        color: var(--primary-color);
                        margin-right: 0.5rem;
                    }

                    .event-desc {
                        color: var(--text-primary);
                    }

                    /* Tab Navigation */
                    .tab-navigation {
                        background: var(--surface-color);
                        border-bottom: 1px solid var(--border-color);
                        padding: 0 2rem;
                    }

                    .tab-item {
                        padding: 1rem 2rem;
                        margin-right: 1rem;
                        border: none;
                        background: transparent;
                        color: var(--text-secondary);
                        font-weight: 500;
                        cursor: pointer;
                        border-bottom: 3px solid transparent;
                        transition: all var(--transition-fast);
                    }

                    .tab-item:hover {
                        color: var(--primary-color);
                        background: #f8f9fa;
                    }

                    .tab-item.active {
                        color: var(--primary-color);
                        border-bottom-color: var(--primary-color);
                        background: #f8f9fa;
                    }

                    /* Responsive Design */
                    @media (max-width: 768px) {
                        .left-column, .right-column {
                            width: 100% !important;
                            margin-bottom: 1rem;
                        }
                        
                        .center-column {
                            width: 100% !important;
                        }
                        
                        .panel-card {
                            padding: 1rem;
                        }
                        
                        .control-btn {
                            padding: 0.5rem 1rem;
                            font-size: 0.85rem;
                        }
                    }

                    /* Loading States */
                    .loading {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 2rem;
                        color: var(--text-secondary);
                    }

                    .loading::before {
                        content: '';
                        display: inline-block;
                        width: 20px;
                        height: 20px;
                        border: 2px solid var(--border-color);
                        border-left-color: var(--primary-color);
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-right: 0.5rem;
                    }

                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }

                    /* Error States */
                    .error-message {
                        color: var(--error-color);
                        background: #ffebee;
                        padding: 1rem;
                        border-radius: var(--border-radius-small);
                        border-left: 4px solid var(--error-color);
                        margin: 0.5rem 0;
                    }

                    .no-data {
                        color: var(--text-secondary);
                        text-align: center;
                        padding: 2rem;
                        font-style: italic;
                    }

                    /* Animations */
                    .fade-in {
                        animation: fadeIn 0.5s ease-in;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }

                    /* Scrollbar Styling */
                    ::-webkit-scrollbar {
                        width: 8px;
                    }

                    ::-webkit-scrollbar-track {
                        background: #f1f1f1;
                        border-radius: 4px;
                    }

                    ::-webkit-scrollbar-thumb {
                        background: #c1c1c1;
                        border-radius: 4px;
                    }

                    ::-webkit-scrollbar-thumb:hover {
                        background: #a1a1a1;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    def _build_modern_layout(self):
        """Create modern layout with enhanced navigation"""
        return html.Div([
            # Enhanced Header
            html.Div([
                html.Div([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-train", style={'marginRight': '1rem', 'color': '#1976d2'}),
                            "KMRL IntelliFleet"
                        ], style={
                            'margin': '0', 'color': '#1976d2', 'fontSize': '2.2rem', 
                            'fontWeight': '700', 'display': 'flex', 'alignItems': 'center'
                        }),
                        html.P("Next-Generation Railway Management System", style={
                            'margin': '0', 'color': '#666', 'fontSize': '1.1rem', 'marginTop': '0.5rem'
                        })
                    ]),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-clock", style={'marginRight': '0.5rem'}),
                            html.Span(id='header-live-time', style={'fontFamily': 'monospace', 'fontSize': '1.1rem'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'color': '#333'})
                    ])
                ], style={
                    'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                    'padding': '2rem', 'backgroundColor': '#fff', 'borderBottom': '1px solid #e0e0e0'
                })
            ]),

            # Enhanced Tab Navigation
            html.Div([
                dcc.Tabs(
                    id='modern-dashboard-tabs',
                    value='enhanced-simulation',
                    children=[
                        dcc.Tab(
                            label='🎬 Live Simulation',
                            value='enhanced-simulation',
                            className='tab-item',
                            selected_className='tab-item active',
                            style={'border': 'none'}
                        ),
                        dcc.Tab(
                            label='📊 Analytics Dashboard', 
                            value='analytics',
                            className='tab-item',
                            selected_className='tab-item active',
                            style={'border': 'none'}
                        )
                    ],
                    style={'height': '60px'}
                )
            ], className='tab-navigation'),

            # Main Content
            html.Div(id='modern-tab-content', className='fade-in'),
            
            # Update intervals
            dcc.Interval(id='modern-animation-interval', interval=1500, n_intervals=0),
            dcc.Interval(id='modern-analytics-interval', interval=2000, n_intervals=0),
            dcc.Store(id='modern-dashboard-state', data={'active_tab': 'enhanced-simulation'})
        ])

    def _register_enhanced_callbacks(self):
        """Register enhanced callbacks with improved error handling"""

        # Header live time update
        @self.app.callback(
            Output('header-live-time', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_header_time(n):
            return datetime.now().strftime('%H:%M:%S')

        # Main tab content rendering
        @self.app.callback(
            Output('modern-tab-content', 'children'),
            Input('modern-dashboard-tabs', 'value')
        )
        def render_modern_tab_content(selected_tab):
            try:
                if selected_tab == 'enhanced-simulation':
                    return self.enhanced_animated_dashboard._create_enhanced_animated_layout()
                elif selected_tab == 'analytics':
                    return html.Div([
                        self.classic_dashboard._create_main_layout()
                    ], style={'padding': '0'})
                else:
                    return html.Div([
                        html.Div([
                            html.H3("Page Not Found", style={'color': '#f44336'}),
                            html.P("The requested page could not be found.")
                        ], className='error-message')
                    ])
            except Exception as e:
                logger.error(f"Error rendering tab content: {e}")
                return html.Div([
                    html.Div([
                        html.H3("Loading Error", style={'color': '#f44336'}),
                        html.P(f"Error: {str(e)}")
                    ], className='error-message')
                ])

        # === ENHANCED ANIMATED DASHBOARD CALLBACKS ===

        # Enhanced animated train map
        @self.app.callback(
            Output('enhanced-animated-train-map', 'figure'),
            Input('modern-animation-interval', 'n_intervals'),
            State('modern-dashboard-state', 'data'),
            prevent_initial_call=False
        )
        def update_enhanced_animated_map(n_intervals, state):
            try:
                return self.enhanced_animated_dashboard._create_enhanced_animated_train_map()
            except Exception as e:
                logger.error(f"Error updating enhanced map: {e}")
                return self.enhanced_animated_dashboard._create_enhanced_animated_train_map()

        # Live timestamp
        @self.app.callback(
            Output('live-timestamp', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_live_timestamp(n):
            return datetime.now().strftime('%H:%M:%S')

        # Active trains count
        @self.app.callback(
            Output('active-trains-count', 'children'),
            Input('modern-animation-interval', 'n_intervals')
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
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_moving_trains_count(n):
            try:
                current_state = self.digital_twin.get_current_state()
                trains = current_state.get('trains', {})
                moving_count = sum(1 for train in trains.values() 
                                 if train.get('status') in ['running', 'moving'])
                return str(moving_count)
            except:
                return "0"

        # Stats panels
        @self.app.callback(
            [Output('stats-total-trains', 'children'),
             Output('stats-running-trains', 'children'),
             Output('stats-depot-trains', 'children'),
             Output('stats-system-health', 'children')],
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_stats_panels(n):
            try:
                current_state = self.digital_twin.get_current_state()
                trains = current_state.get('trains', {})
                
                total = len(trains)
                running = sum(1 for t in trains.values() if t.get('status') in ['running', 'moving'])
                depot = sum(1 for t in trains.values() if 'depot' in t.get('location', '').lower())
                health = "98%"  # Simulated health percentage
                
                return str(total), str(running), str(depot), health
            except:
                return "0", "0", "0", "N/A"

        # System health indicators
        @self.app.callback(
            [Output('health-digital-twin', 'children'),
             Output('health-iot-sensors', 'children'),
             Output('health-ai-engine', 'children'),
             Output('health-network', 'children')],
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_health_indicators(n):
            # Simulated health status
            return "Online", "Active", "Optimal", "Connected"

        # Enhanced event feed
        @self.app.callback(
            Output('enhanced-movement-event-feed', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_enhanced_event_feed(n):
            return self.enhanced_animated_dashboard.get_enhanced_movement_event_feed()

        # Enhanced progress bars
        @self.app.callback(
            Output('enhanced-movement-progress-bars', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_enhanced_progress_bars(n):
            return self.enhanced_animated_dashboard.get_enhanced_movement_progress_bars()

        # Enhanced bay status
        @self.app.callback(
            Output('enhanced-live-bay-status', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_enhanced_bay_status(n):
            return self.enhanced_animated_dashboard.get_enhanced_live_bay_status()

        # Current animation speed display
        @self.app.callback(
            Output('current-speed', 'children'),
            Input('animation-speed-slider', 'value'),
            prevent_initial_call=False
        )
        def update_current_speed_display(speed):
            return f"{speed}×"

        # Map last update timestamp
        @self.app.callback(
            Output('map-last-update', 'children'),
            Input('modern-animation-interval', 'n_intervals')
        )
        def update_map_timestamp(n):
            return datetime.now().strftime('%H:%M:%S')

        # === ANALYTICS DASHBOARD CALLBACKS (existing ones) ===

        # Status cards
        @self.app.callback(
            Output('status-cards', 'children'),
            Input('modern-analytics-interval', 'n_intervals')
        )
        def update_analytics_status_cards(n):
            try:
                return self.classic_dashboard.setup_callbacks() or []
            except:
                return []

    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the modern combined dashboard"""
        print("\n" + "="*80)
        print("🚀 STARTING MODERN KMRL INTELLIFLEET DASHBOARD")
        print("="*80)
        print(f"🌐 URL: http://{host}:{port}")
        print("✨ Features:")
        print("   🎬 Enhanced Live Simulation - Modern 3-column layout")
        print("   📊 Analytics Dashboard - AI insights and metrics")
        print("   🎨 Material Design 3.0 - Professional styling")
        print("   📱 Fully Responsive - Mobile, tablet, desktop ready")
        print("   ⚡ Real-time Updates - 1.5s refresh rate")
        print("   🎯 Enhanced UX - Improved navigation and controls")
        print("="*80)
        print("🎮 NAVIGATION:")
        print("   • Use tabs to switch between Live Simulation and Analytics")
        print("   • Live Simulation: 3-column layout with controls, map, and activity feed")
        print("   • Interactive map with real KMRL route visualization")
        print("   • Enhanced animation controls and system monitoring")
        print("="*80)
        
        self.app.run(host=host, port=port, debug=debug)