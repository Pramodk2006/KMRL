"""
Enhanced Animated Dashboard for KMRL IntelliFleet Live Simulation
Modern, responsive layout with improved user experience and visual design
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnhancedAnimatedTrainDashboard:
    """Enhanced Live Train Simulation Dashboard with Modern Layout"""

    def __init__(self, digital_twin_engine, monitor=None, iot_simulator=None, cv_system=None,
                 ai_optimizer=None, constraint_engine=None, ai_dashboard=None, ai_data_processor=None):
        self.digital_twin = digital_twin_engine
        self.monitor = monitor
        self.iot_simulator = iot_simulator
        self.cv_system = cv_system
        self.ai_optimizer = ai_optimizer
        self.constraint_engine = constraint_engine
        self.ai_dashboard = ai_dashboard
        self.ai_data_processor = ai_data_processor

        # KMRL Route data with GPS coordinates
        self.kmrl_stations = {
            'Muttom Depot': {'lat': 9.9312, 'lon': 76.2673, 'type': 'depot'},
            'Ernakulam South': {'lat': 9.9816, 'lon': 76.2999, 'type': 'station'},
            'Kadavanthra': {'lat': 9.9648, 'lon': 76.3028, 'type': 'station'},
            'Town Hall': {'lat': 9.9749, 'lon': 76.2854, 'type': 'station'},
            'Maharajas College': {'lat': 9.9393, 'lon': 76.2856, 'type': 'station'},
            'Kaloor': {'lat': 9.9594, 'lon': 76.2818, 'type': 'station'},
            'JLN Stadium': {'lat': 9.9404, 'lon': 76.2758, 'type': 'station'}
        }

    def _create_enhanced_animated_layout(self):
        """Create the enhanced animated layout with modern design"""
        return html.Div([
            # Modern Header Section
            self._create_header_section(),
            
            # Main Content Grid
            html.Div([
                # Left Column - Controls & Status
                html.Div([
                    self._create_control_panel(),
                    self._create_live_stats_panel(),
                    self._create_system_health_panel()
                ], className="left-column", style={
                    'width': '25%', 'padding': '1rem', 'backgroundColor': '#fff',
                    'boxShadow': '2px 0 10px rgba(0,0,0,0.1)', 'height': 'calc(100vh - 120px)',
                    'overflowY': 'auto'
                }),
                
                # Center Column - Interactive Map
                html.Div([
                    self._create_map_container()
                ], className="center-column", style={
                    'width': '50%', 'padding': '1rem', 'backgroundColor': '#fafafa'
                }),
                
                # Right Column - Activity Feed & Progress
                html.Div([
                    self._create_activity_feed(),
                    self._create_progress_tracking(),
                    self._create_bay_status_panel()
                ], className="right-column", style={
                    'width': '25%', 'padding': '1rem', 'backgroundColor': '#fff',
                    'boxShadow': '-2px 0 10px rgba(0,0,0,0.1)', 'height': 'calc(100vh - 120px)',
                    'overflowY': 'auto'
                })
            ], style={
                'display': 'flex', 'width': '100%', 'minHeight': 'calc(100vh - 120px)'
            }),

            # Auto-refresh interval
            dcc.Interval(id='animation-interval', interval=1500, n_intervals=0),
            dcc.Store(id='animation-state', data={'paused': False, 'speed': 1.0})
        ], style={'backgroundColor': '#f5f7fa', 'minHeight': '100vh'})

    def _create_header_section(self):
        """Create modern header with live information"""
        return html.Div([
            html.Div([
                # Left - Title and Status
                html.Div([
                    html.H2("🎬 Live Train Simulation", style={
                        'margin': '0', 'color': '#1976d2', 'fontSize': '1.8rem', 'fontWeight': '600'
                    }),
                    html.Div([
                        html.Span("🟢", id="system-status-icon"),
                        html.Span("System Active", id="system-status-text", style={
                            'marginLeft': '0.5rem', 'color': '#4caf50', 'fontWeight': '500'
                        })
                    ])
                ], style={'display': 'flex', 'flexDirection': 'column'}),
                
                # Center - Live Clock
                html.Div([
                    html.Div([
                        html.H3(id='live-timestamp', style={
                            'margin': '0', 'color': '#333', 'fontSize': '2rem', 'fontFamily': 'monospace'
                        }),
                        html.P("Live Time", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'})
                    ], style={'textAlign': 'center'})
                ]),
                
                # Right - Quick Stats
                html.Div([
                    html.Div([
                        html.H4(id='active-trains-count', style={
                            'margin': '0', 'color': '#1976d2', 'fontSize': '1.5rem'
                        }),
                        html.P("Active Trains", style={'margin': '0', 'color': '#666', 'fontSize': '0.85rem'})
                    ], style={'textAlign': 'center', 'marginRight': '1.5rem'}),
                    html.Div([
                        html.H4(id='moving-trains-count', style={
                            'margin': '0', 'color': '#4caf50', 'fontSize': '1.5rem'
                        }),
                        html.P("Moving", style={'margin': '0', 'color': '#666', 'fontSize': '0.85rem'})
                    ], style={'textAlign': 'center'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                'padding': '1.5rem 2rem', 'backgroundColor': '#fff',
                'borderBottom': '1px solid #e0e0e0', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ])

    def _create_control_panel(self):
        """Create enhanced animation control panel"""
        return html.Div([
            html.H4("🎮 Animation Controls", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            
            # Play/Pause/Reset Controls
            html.Div([
                html.Button([
                    html.I(className="fas fa-play", style={'marginRight': '0.5rem'}),
                    "Play"
                ], id="animation-play-btn", className="control-btn primary", n_clicks=0),
                html.Button([
                    html.I(className="fas fa-pause", style={'marginRight': '0.5rem'}),
                    "Pause"
                ], id="animation-pause-btn", className="control-btn secondary", n_clicks=0),
                html.Button([
                    html.I(className="fas fa-stop", style={'marginRight': '0.5rem'}),
                    "Reset"
                ], id="animation-reset-btn", className="control-btn danger", n_clicks=0)
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '0.5rem', 'marginBottom': '1.5rem'}),
            
            # Speed Control
            html.Div([
                html.Label("Animation Speed", style={'fontWeight': '500', 'marginBottom': '0.5rem', 'display': 'block'}),
                html.Div([
                    dcc.Slider(
                        id='animation-speed-slider',
                        min=0.1, max=5.0, step=0.1, value=1.0,
                        marks={0.1: '0.1×', 1: '1×', 2: '2×', 3: '3×', 5: '5×'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span("Current Speed: ", style={'color': '#666'}),
                    html.Span(id='current-speed', style={'fontWeight': 'bold', 'color': '#1976d2'})
                ])
            ], className="control-section"),
            
            # View Options
            html.Div([
                html.H5("🔧 View Options", style={'color': '#666', 'marginBottom': '0.75rem'}),
                dcc.Checklist(
                    id='view-options',
                    options=[
                        {'label': ' Show Route Lines', 'value': 'routes'},
                        {'label': ' Show Station Names', 'value': 'stations'},
                        {'label': ' Show Train IDs', 'value': 'train_ids'},
                        {'label': ' Real-time Updates', 'value': 'realtime'}
                    ],
                    value=['routes', 'stations', 'train_ids', 'realtime'],
                    style={'color': '#333'}
                )
            ], className="control-section")
            
        ], className="panel-card")

    def _create_live_stats_panel(self):
        """Create live statistics panel"""
        return html.Div([
            html.H4("📊 Live Statistics", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            
            # Stats Grid
            html.Div([
                # Total Trains
                html.Div([
                    html.Div("🚂", style={'fontSize': '2rem', 'marginBottom': '0.5rem'}),
                    html.H3(id='stats-total-trains', style={'margin': '0', 'color': '#1976d2'}),
                    html.P("Total Trains", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'})
                ], className="stat-card"),
                
                # Running Trains
                html.Div([
                    html.Div("🏃", style={'fontSize': '2rem', 'marginBottom': '0.5rem'}),
                    html.H3(id='stats-running-trains', style={'margin': '0', 'color': '#4caf50'}),
                    html.P("Running", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'})
                ], className="stat-card"),
                
                # At Depot
                html.Div([
                    html.Div("🏠", style={'fontSize': '2rem', 'marginBottom': '0.5rem'}),
                    html.H3(id='stats-depot-trains', style={'margin': '0', 'color': '#ff9800'}),
                    html.P("At Depot", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'})
                ], className="stat-card"),
                
                # System Health
                html.Div([
                    html.Div("❤️", style={'fontSize': '2rem', 'marginBottom': '0.5rem'}),
                    html.H3(id='stats-system-health', style={'margin': '0', 'color': '#e91e63'}),
                    html.P("Health", style={'margin': '0', 'color': '#666', 'fontSize': '0.9rem'})
                ], className="stat-card")
            ], style={
                'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '0.75rem'
            })
        ], className="panel-card")

    def _create_system_health_panel(self):
        """Create system health monitoring panel"""
        return html.Div([
            html.H4("🔍 System Health", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            
            # Health Indicators
            html.Div([
                html.Div([
                    html.Div(["🔄", html.Span(" Digital Twin", style={'marginLeft': '0.5rem'})]),
                    html.Div(id='health-digital-twin', className="health-status good")
                ], className="health-item"),
                
                html.Div([
                    html.Div(["📡", html.Span(" IoT Sensors", style={'marginLeft': '0.5rem'})]),
                    html.Div(id='health-iot-sensors', className="health-status good")
                ], className="health-item"),
                
                html.Div([
                    html.Div(["🤖", html.Span(" AI Engine", style={'marginLeft': '0.5rem'})]),
                    html.Div(id='health-ai-engine', className="health-status good")
                ], className="health-item"),
                
                html.Div([
                    html.Div(["🌐", html.Span(" Network", style={'marginLeft': '0.5rem'})]),
                    html.Div(id='health-network', className="health-status good")
                ], className="health-item")
            ])
        ], className="panel-card")

    def _create_map_container(self):
        """Create the enhanced interactive map container"""
        return html.Div([
            # Map Header
            html.Div([
                html.H3("🗺️ KMRL Route Map", style={
                    'margin': '0', 'color': '#1976d2', 'fontSize': '1.4rem', 'fontWeight': '600'
                }),
                html.Div([
                    html.Span("Last Updated: "),
                    html.Span(id='map-last-update', style={'fontWeight': 'bold', 'color': '#4caf50'})
                ], style={'fontSize': '0.85rem', 'color': '#666'})
            ], style={
                'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                'marginBottom': '1rem', 'padding': '0 0.5rem'
            }),
            
            # Interactive Map
            html.Div([
                dcc.Graph(
                    id='enhanced-animated-train-map',
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'KMRL_live_map',
                            'height': 600,
                            'width': 800,
                            'scale': 2
                        }
                    },
                    style={'height': '60vh'}
                )
            ], style={
                'backgroundColor': '#fff', 'borderRadius': '12px',
                'boxShadow': '0 4px 16px rgba(0,0,0,0.1)', 'padding': '1rem'
            }),
            
            # Map Legend
            html.Div([
                html.H5("🏷️ Legend", style={'margin': '0 0 0.75rem 0', 'color': '#333'}),
                html.Div([
                    html.Div([
                        html.Div(style={
                            'width': '16px', 'height': '16px', 'backgroundColor': '#4caf50',
                            'borderRadius': '50%', 'marginRight': '0.5rem'
                        }),
                        html.Span("Running Trains", style={'fontSize': '0.85rem'})
                    ], className="legend-item"),
                    html.Div([
                        html.Div(style={
                            'width': '16px', 'height': '16px', 'backgroundColor': '#ff9800',
                            'borderRadius': '50%', 'marginRight': '0.5rem'
                        }),
                        html.Span("At Station", style={'fontSize': '0.85rem'})
                    ], className="legend-item"),
                    html.Div([
                        html.Div(style={
                            'width': '16px', 'height': '16px', 'backgroundColor': '#1976d2',
                            'borderRadius': '50%', 'marginRight': '0.5rem'
                        }),
                        html.Span("At Depot", style={'fontSize': '0.85rem'})
                    ], className="legend-item"),
                    html.Div([
                        html.Div(style={
                            'width': '16px', 'height': '16px', 'backgroundColor': '#f44336',
                            'borderRadius': '50%', 'marginRight': '0.5rem'
                        }),
                        html.Span("Maintenance", style={'fontSize': '0.85rem'})
                    ], className="legend-item")
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '1rem'})
            ], style={
                'marginTop': '1rem', 'padding': '1rem', 'backgroundColor': '#f8f9fa',
                'borderRadius': '8px', 'border': '1px solid #e0e0e0'
            })
        ])

    def _create_activity_feed(self):
        """Create live activity feed"""
        return html.Div([
            html.H4("📡 Live Activity", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            html.Div(id='enhanced-movement-event-feed', style={
                'maxHeight': '250px', 'overflowY': 'auto',
                'border': '1px solid #e0e0e0', 'borderRadius': '8px',
                'backgroundColor': '#fafafa', 'padding': '0.75rem'
            })
        ], className="panel-card")

    def _create_progress_tracking(self):
        """Create progress tracking panel"""
        return html.Div([
            html.H4("⏱️ Journey Progress", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            html.Div(id='enhanced-movement-progress-bars', style={
                'maxHeight': '200px', 'overflowY': 'auto'
            })
        ], className="panel-card")

    def _create_bay_status_panel(self):
        """Create bay status monitoring panel"""
        return html.Div([
            html.H4("🏗️ Bay Status", style={
                'color': '#1976d2', 'marginBottom': '1rem', 'fontSize': '1.2rem', 'fontWeight': '600'
            }),
            html.Div(id='enhanced-live-bay-status')
        ], className="panel-card")

    def _create_enhanced_animated_train_map(self):
        """Create enhanced animated train map with better visualization"""
        try:
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            
            # Create figure
            fig = go.Figure()
            
            # Add KMRL route lines
            route_coords = self._get_route_coordinates()
            for route_name, coords in route_coords.items():
                fig.add_trace(go.Scattermapbox(
                    lat=[coord[0] for coord in coords],
                    lon=[coord[1] for coord in coords],
                    mode='lines',
                    line=dict(width=4, color='#1976d2'),
                    name=route_name,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add stations
            for station_name, station_info in self.kmrl_stations.items():
                station_color = '#ff6b6b' if station_info['type'] == 'depot' else '#4ecdc4'
                station_size = 15 if station_info['type'] == 'depot' else 10
                
                fig.add_trace(go.Scattermapbox(
                    lat=[station_info['lat']],
                    lon=[station_info['lon']],
                    mode='markers+text',
                    marker=dict(size=station_size, color=station_color),
                    text=[station_name],
                    textposition="top center",
                    textfont=dict(size=10, color='#333'),
                    name=f"{station_info['type'].title()}s",
                    showlegend=False,
                    hovertemplate=f"<b>{station_name}</b><br>Type: {station_info['type'].title()}<extra></extra>"
                ))
            
            # Add trains with enhanced visualization
            train_lats, train_lons, train_colors, train_texts, train_hovers = [], [], [], [], []
            
            for train_id, train_info in trains.items():
                location = train_info.get('location', 'muttom_depot')
                status = train_info.get('status', 'idle')
                
                # Get coordinates with slight randomization for realistic movement
                lat, lon = self._get_train_coordinates(location, train_id)
                
                # Enhanced status-based styling
                if status in ['running', 'moving']:
                    color = '#4caf50'
                    symbol = 'arrow'
                    size = 12
                elif status in ['inducted', 'ready']:
                    color = '#2196f3'
                    symbol = 'circle'
                    size = 10
                elif status == 'maintenance':
                    color = '#f44336'
                    symbol = 'x'
                    size = 10
                else:
                    color = '#ff9800'
                    symbol = 'circle'
                    size = 8
                
                train_lats.append(lat)
                train_lons.append(lon)
                train_colors.append(color)
                train_texts.append(train_id)
                
                # Enhanced hover information
                hover_text = f"""
                <b>{train_id}</b><br>
                Status: {status.title()}<br>
                Location: {location.replace('_', ' ').title()}<br>
                Mileage: {train_info.get('mileage_km', 'N/A')} km<br>
                Last Update: {datetime.now().strftime('%H:%M:%S')}
                """
                train_hovers.append(hover_text)
            
            # Add train markers
            if train_lats:
                fig.add_trace(go.Scattermapbox(
                    lat=train_lats,
                    lon=train_lons,
                    mode='markers+text',
                    marker=dict(
                        size=[12 if trains[train_texts[i]].get('status') in ['running', 'moving'] else 10 
                              for i in range(len(train_texts))],
                        color=train_colors,
                        opacity=0.8
                    ),
                    text=train_texts,
                    textposition="middle center",
                    textfont=dict(size=8, color='white'),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=train_hovers,
                    name="Trains",
                    showlegend=False
                ))
            
            # Enhanced layout
            fig.update_layout(
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoidGVzdCIsImEiOiJjbGh6N2NubXQwMHl6M2RwZmt3bWdpNWl2In0.sample_token',
                    style='open-street-map',
                    center=dict(lat=9.9592, lon=76.2895),  # Centered on Kochi
                    zoom=12,
                    bearing=0,
                    pitch=0
                ),
                title={
                    'text': f"🚄 KMRL Live Train Tracking - {datetime.now().strftime('%H:%M:%S')}",
                    'x': 0.5,
                    'font': {'size': 16, 'color': '#1976d2'}
                },
                height=480,
                margin={'l': 0, 'r': 0, 't': 40, 'b': 0},
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            # Enhanced error handling
            error_fig = go.Figure()
            error_fig.add_annotation(
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                text=f"🚫 Map Loading Error<br><br>Error: {str(e)}<br><br>Please check system connection",
                showarrow=False,
                font={'size': 14, 'color': '#f44336'},
                bgcolor='rgba(255,235,238,0.8)',
                bordercolor='#f44336',
                borderwidth=2
            )
            error_fig.update_layout(
                title="KMRL Route Map - Error",
                height=480,
                plot_bgcolor='#fafafa'
            )
            return error_fig

    def _get_route_coordinates(self):
        """Get KMRL route coordinates for line drawing"""
        return {
            'Main Line': [
                [9.9312, 76.2673],  # Muttom Depot
                [9.9393, 76.2856],  # Maharajas College
                [9.9404, 76.2758],  # JLN Stadium
                [9.9594, 76.2818],  # Kaloor
                [9.9749, 76.2854],  # Town Hall
                [9.9648, 76.3028],  # Kadavanthra
                [9.9816, 76.2999]   # Ernakulam South
            ]
        }

    def _get_train_coordinates(self, location, train_id):
        """Get train coordinates with realistic variation"""
        base_coords = self.kmrl_stations.get(location.replace('_', ' ').title(), 
                                           self.kmrl_stations['Muttom Depot'])
        
        # Add small random variation for realistic movement
        variation = 0.001  # Small variation in degrees
        hash_seed = hash(train_id) % 1000
        lat_offset = (hash_seed % 100 - 50) * variation / 50
        lon_offset = ((hash_seed // 100) % 100 - 50) * variation / 50
        
        return base_coords['lat'] + lat_offset, base_coords['lon'] + lon_offset

    def get_enhanced_movement_event_feed(self):
        """Get enhanced movement event feed"""
        try:
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            events = []
            
            current_time = datetime.now()
            
            for train_id, train_info in trains.items():
                status = train_info.get('status', 'idle')
                location = train_info.get('location', 'depot')
                
                # Create realistic events
                if status == 'running':
                    event_time = (current_time - timedelta(minutes=2)).strftime('%H:%M:%S')
                    events.append(html.Div([
                        html.Div([
                            html.Span(f"[{event_time}]", className="event-time"),
                            html.Span("🚂", className="event-icon"),
                            html.Span(f"{train_id}", className="event-train"),
                            html.Span(f"Running to {location.replace('_', ' ').title()}", className="event-desc")
                        ], className="event-item running")
                    ]))
                elif status == 'inducted':
                    event_time = (current_time - timedelta(minutes=5)).strftime('%H:%M:%S')
                    events.append(html.Div([
                        html.Div([
                            html.Span(f"[{event_time}]", className="event-time"),
                            html.Span("✅", className="event-icon"),
                            html.Span(f"{train_id}", className="event-train"),
                            html.Span("Inducted for service", className="event-desc")
                        ], className="event-item inducted")
                    ]))
            
            # Add system events
            events.append(html.Div([
                html.Div([
                    html.Span(f"[{current_time.strftime('%H:%M:%S')}]", className="event-time"),
                    html.Span("🔄", className="event-icon"),
                    html.Span("SYSTEM", className="event-train"),
                    html.Span("Live tracking active", className="event-desc")
                ], className="event-item system")
            ]))
            
            return events[-10:]  # Return last 10 events
            
        except Exception as e:
            return [html.Div(f"Event feed error: {str(e)}", className="error-message")]

    def get_enhanced_movement_progress_bars(self):
        """Get enhanced movement progress bars"""
        try:
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            progress_bars = []
            
            for train_id, train_info in trains.items():
                status = train_info.get('status', 'idle')
                
                if status in ['running', 'moving']:
                    # Simulate progress based on time
                    progress = (hash(train_id) % 100) + (datetime.now().minute % 20)
                    progress = min(progress % 100, 95)  # Cap at 95%
                    
                    progress_bars.append(html.Div([
                        html.Div([
                            html.Span(train_id, style={'fontWeight': 'bold'}),
                            html.Span(f"{progress}%", style={'fontSize': '0.85rem', 'color': '#666'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '0.25rem'}),
                        html.Div([
                            html.Div(style={
                                'width': f'{progress}%',
                                'height': '8px',
                                'backgroundColor': '#4caf50',
                                'borderRadius': '4px',
                                'transition': 'width 0.3s ease'
                            })
                        ], style={
                            'width': '100%',
                            'height': '8px',
                            'backgroundColor': '#e0e0e0',
                            'borderRadius': '4px'
                        })
                    ], style={'marginBottom': '1rem'}))
            
            if not progress_bars:
                return [html.Div("No trains currently in motion", className="no-data")]
                
            return progress_bars
            
        except Exception as e:
            return [html.Div(f"Progress tracking error: {str(e)}", className="error-message")]

    def get_enhanced_live_bay_status(self):
        """Get enhanced live bay status"""
        try:
            bay_statuses = []
            
            # Simulate bay data
            bays = ['Bay 1', 'Bay 2', 'Bay 4']
            statuses = ['Available', 'Occupied (T001)', 'Maintenance']
            colors = ['#4caf50', '#ff9800', '#f44336']
            
            for i, bay in enumerate(bays):
                status = statuses[i % len(statuses)]
                color = colors[i % len(colors)]
                
                bay_statuses.append(html.Div([
                    html.Div([
                        html.Div(style={
                            'width': '12px', 'height': '12px',
                            'backgroundColor': color, 'borderRadius': '50%'
                        }),
                        html.Span(bay, style={'fontWeight': '500', 'marginLeft': '0.5rem'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '0.25rem'}),
                    html.Div(status, style={
                        'fontSize': '0.85rem', 'color': '#666',
                        'marginLeft': '1.25rem'
                    })
                ], style={'marginBottom': '0.75rem'}))
                
            return bay_statuses
            
        except Exception as e:
            return [html.Div(f"Bay status error: {str(e)}", className="error-message")]