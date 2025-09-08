"""
Enhanced Animated Web Dashboard for Combined System
Modified to work as a component within the combined dashboard
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class AnimatedTrainDashboard:
    """Enhanced animated train dashboard for combined system"""
    
    def __init__(self, digital_twin_engine, monitor=None, iot_simulator=None, cv_system=None,
                 ai_optimizer=None, constraint_engine=None, ai_dashboard=None, ai_data_processor=None):
        
        # Store system components
        self.digital_twin = digital_twin_engine
        self.monitor = monitor
        self.iot_simulator = iot_simulator
        self.cv_system = cv_system
        self.ai_optimizer = ai_optimizer
        self.constraint_engine = constraint_engine
        self.ai_dashboard = ai_dashboard
        self.ai_data_processor = ai_data_processor
        
        # Animation state
        self.is_animating = True
        self.animation_speed = 1.0
        
        # KMRL coordinates and locations
        self.kmrl_locations = {
            'muttom_depot': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Muttom Depot'},
            'maharajas_college': {'lat': 9.9380, 'lon': 76.2828, 'name': 'Maharajas College'},
            'ernakulam_south': {'lat': 9.9816, 'lon': 76.2999, 'name': 'Ernakulam South'},
            'kadavanthra': {'lat': 10.0090, 'lon': 76.3048, 'name': 'Kadavanthra'},
            'mg_road': {'lat': 9.9816, 'lon': 76.2999, 'name': 'MG Road'},
        }
        
        # Service bay positions (around Muttom depot)
        self.bay_positions = {
            'Bay1': {'lat': 9.9315, 'lon': 76.2670},
            'Bay2': {'lat': 9.9318, 'lon': 76.2672},
            'Bay3': {'lat': 9.9321, 'lon': 76.2674},
            'Bay4': {'lat': 9.9324, 'lon': 76.2676},
        }
        
        # Map center and zoom
        self.map_center = {'lat': 9.9816, 'lon': 76.2999}
        self.map_zoom = 12
    
    def _create_animated_layout(self):
        """Create the animated simulation dashboard layout"""
        return html.Div([
            # Animation Controls
            html.Div([
                html.Div([
                    html.H4("🎬 Animation Controls", className="text-primary mb-3"),
                    html.Div([
                        html.Button("▶️ Play", id="play-btn", className="btn btn-success me-2", n_clicks=0),
                        html.Button("⏸️ Pause", id="pause-btn", className="btn btn-warning me-2", n_clicks=0),
                        html.Button("⏹️ Stop", id="stop-btn", className="btn btn-danger me-2", n_clicks=0),
                        html.Button("🔄 Reset", id="reset-btn", className="btn btn-info", n_clicks=0),
                    ], className="d-flex mb-3"),
                    
                    html.Div([
                        html.Label("Animation Speed", className="form-label"),
                        dcc.Slider(
                            id="animation-speed-slider",
                            min=0.1, max=5.0, step=0.1, value=1.0,
                            marks={0.5: '0.5×', 1: '1×', 2: '2×', 5: '5×'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3")
                ], className="card p-3")
            ], className="col-md-4 mb-4"),
            
            # Live Statistics
            html.Div([
                html.Div([
                    html.H4("📊 Live Statistics", className="text-primary mb-3"),
                    html.Div([
                        html.Div([
                            html.H5("Current Time", className="text-muted mb-1"),
                            html.H3(id="live-timestamp", className="text-dark mb-0")
                        ], className="text-center"),
                        html.Hr(),
                        html.Div([
                            html.Div([
                                html.H6("Active Trains", className="text-muted mb-1"),
                                html.H4(id="active-trains-count", className="text-primary mb-0")
                            ], className="col-6 text-center"),
                            html.Div([
                                html.H6("Moving Trains", className="text-muted mb-1"),
                                html.H4(id="moving-trains-count", className="text-success mb-0")
                            ], className="col-6 text-center")
                        ], className="row")
                    ])
                ], className="card p-3")
            ], className="col-md-4 mb-4"),
            
            # Animation Status
            html.Div([
                html.Div([
                    html.H4("🔄 Animation Status", className="text-primary mb-3"),
                    html.Div([
                        html.Div([
                            html.Span("●", className="live-indicator me-2"),
                            html.Span("LIVE", className="fw-bold")
                        ], id="animation-status", className="mb-2"),
                        html.P("Real-time train movement simulation", className="text-muted mb-2"),
                        html.Div([
                            html.Small("Speed: ", className="text-muted"),
                            html.Small(id="current-speed", className="fw-bold text-primary")
                        ])
                    ])
                ], className="card p-3")
            ], className="col-md-4 mb-4"),
            
            # Main animated map
            html.Div([
                html.Div([
                    html.H4("🗺️ Live Train Tracking Map", className="text-primary mb-3"),
                    html.Div([
                        dcc.Graph(
                            id="animated-train-map",
                            config={'displayModeBar': False},
                            style={'height': '600px'}
                        )
                    ])
                ], className="card p-3")
            ], className="col-12 mb-4"),
            
            # Movement Progress and Bay Status
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("🚂 Train Movement Progress", className="text-primary mb-3"),
                        html.Div(id="movement-progress-bars")
                    ], className="card p-3")
                ], className="col-md-6 mb-4"),
                
                html.Div([
                    html.Div([
                        html.H5("🏗️ Live Bay Occupancy", className="text-primary mb-3"),
                        html.Div(id="live-bay-status")
                    ], className="card p-3")
                ], className="col-md-6 mb-4")
            ], className="row"),
            
            # Real-time Event Feed
            html.Div([
                html.Div([
                    html.H5("📡 Live Movement Feed", className="text-primary mb-3"),
                    html.Div([
                        html.Div(id="movement-event-feed", 
                                style={'height': '300px', 'overflowY': 'auto'})
                    ])
                ], className="card p-3")
            ], className="col-12 mb-4"),
            
            # Animation interval and state storage
            dcc.Interval(id='animation-interval', interval=1500, n_intervals=0),  # 1.5 second updates
            dcc.Store(id='animation-state', data={'paused': False, 'speed': 1.0})
            
        ], className="row")
    
    def _create_animated_train_map(self):
        """Create the main animated train map with live positions"""
        try:
            # Get current state from digital twin
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            bays = current_state.get('bays', {})
            
            # Create the map figure
            fig = go.Figure()
            
            # Add train markers
            train_lats, train_lons, train_texts, train_colors, train_symbols = [], [], [], [], []
            
            for train_id, train_info in trains.items():
                # Get train position (use random movement for demo)
                position = self._get_train_position(train_id, train_info)
                
                train_lats.append(position['lat'])
                train_lons.append(position['lon'])
                train_texts.append(f"{train_id}<br>Status: {train_info.get('status', 'Unknown')}")
                
                # Color code by status
                status = train_info.get('status', 'idle')
                if status == 'service':
                    train_colors.append('#4CAF50')  # Green - in service
                    train_symbols.append('circle')
                elif status == 'moving':
                    train_colors.append('#FF9800')  # Orange - moving
                    train_symbols.append('triangle-right')
                else:
                    train_colors.append('#9E9E9E')  # Gray - idle
                    train_symbols.append('circle')
            
            # Add train markers to map
            if train_lats:
                fig.add_trace(go.Scattermapbox(
                    lat=train_lats,
                    lon=train_lons,
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=train_colors,
                        symbol=train_symbols
                    ),
                    text=train_texts,
                    hoverinfo='text',
                    name='Trains'
                ))
            
            # Add bay markers
            bay_lats, bay_lons, bay_texts, bay_colors = [], [], [], []
            
            for bay_id, position in self.bay_positions.items():
                bay_info = bays.get(bay_id, {})
                status = bay_info.get('status', 'available')
                occupied_trains = bay_info.get('occupied_trains', [])
                
                bay_lats.append(position['lat'])
                bay_lons.append(position['lon'])
                bay_texts.append(f"{bay_id}<br>Status: {status}<br>Trains: {len(occupied_trains)}")
                
                # Color code by status
                if status == 'occupied':
                    bay_colors.append('#F44336')  # Red - occupied
                elif status == 'partial':
                    bay_colors.append('#FF9800')  # Orange - partial
                else:
                    bay_colors.append('#4CAF50')  # Green - available
            
            # Add bay markers to map
            if bay_lats:
                fig.add_trace(go.Scattermapbox(
                    lat=bay_lats,
                    lon=bay_lons,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=bay_colors,
                        symbol='square'
                    ),
                    text=bay_texts,
                    hoverinfo='text',
                    name='Service Bays'
                ))
            
            # Add KMRL stations
            station_lats, station_lons, station_texts = [], [], []
            for location_id, location_info in self.kmrl_locations.items():
                station_lats.append(location_info['lat'])
                station_lons.append(location_info['lon'])
                station_texts.append(f"{location_info['name']}<br>KMRL Station")
            
            fig.add_trace(go.Scattermapbox(
                lat=station_lats,
                lon=station_lons,
                mode='markers',
                marker=dict(
                    size=10,
                    color='#2196F3',
                    symbol='rail'
                ),
                text=station_texts,
                hoverinfo='text',
                name='KMRL Stations'
            ))
            
            # Configure map layout
            fig.update_layout(
                mapbox=dict(
                    accesstoken=None,  # Use OpenStreetMap (no token needed)
                    style="open-street-map",
                    center=dict(lat=self.map_center['lat'], lon=self.map_center['lon']),
                    zoom=self.map_zoom
                ),
                showlegend=True,
                height=600,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_map(f"Error creating map: {str(e)}")
    
    def _get_train_position(self, train_id, train_info):
        """Get current position of a train (simulated movement)"""
        try:
            # Get base position from train location
            location = train_info.get('location', 'muttom_depot')
            
            if location in self.kmrl_locations:
                base_pos = self.kmrl_locations[location]
            else:
                base_pos = self.kmrl_locations['muttom_depot']  # Default
            
            # Add some realistic movement simulation
            time_factor = (datetime.now().timestamp() / 100) % (2 * math.pi)
            
            # Create slight movement for trains (simulate real GPS variation)
            lat_offset = math.sin(time_factor + hash(train_id) % 100) * 0.002
            lon_offset = math.cos(time_factor + hash(train_id) % 100) * 0.002
            
            return {
                'lat': base_pos['lat'] + lat_offset,
                'lon': base_pos['lon'] + lon_offset
            }
            
        except Exception:
            # Fallback to depot position
            return self.kmrl_locations['muttom_depot']
    
    def _create_empty_map(self, error_message="No data available"):
        """Create an empty map with error message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=error_message,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=self.map_center['lat'], lon=self.map_center['lon']),
                zoom=self.map_zoom
            ),
            showlegend=False,
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        
        return fig
    
    def get_movement_progress_bars(self):
        """Create progress bars for train movements"""
        try:
            current_state = self.digital_twin.get_current_state()
            trains = current_state.get('trains', {})
            
            progress_items = []
            
            for train_id, train_info in trains.items():
                status = train_info.get('status', 'idle')
                
                if status == 'moving':
                    # Simulate progress (in real system, this would be actual progress)
                    progress = (hash(train_id) + int(datetime.now().timestamp())) % 100
                    
                    progress_items.append(
                        html.Div([
                            html.Div([
                                html.Strong(f"{train_id}", className="text-primary"),
                                html.Small(f" → {train_info.get('assigned_bay', 'Unknown')}", 
                                         className="text-muted ms-2")
                            ], className="mb-1"),
                            html.Div([
                                html.Div(
                                    style={
                                        'width': f'{progress}%',
                                        'height': '8px',
                                        'backgroundColor': '#4CAF50',
                                        'borderRadius': '4px',
                                        'transition': 'width 0.5s ease'
                                    }
                                )
                            ], style={
                                'backgroundColor': '#E0E0E0',
                                'borderRadius': '4px',
                                'height': '8px',
                                'width': '100%'
                            }),
                            html.Small(f"{progress}% complete", className="text-muted")
                        ], className="mb-3")
                    )
            
            if not progress_items:
                progress_items = [
                    html.Div("No trains currently in movement", 
                           className="text-muted text-center p-3")
                ]
            
            return progress_items
            
        except Exception as e:
            return [html.Div(f"Error loading progress: {str(e)}", className="text-danger")]
    
    def get_live_bay_status(self):
        """Get live bay status display"""
        try:
            current_state = self.digital_twin.get_current_state()
            bays = current_state.get('bays', {})
            
            bay_items = []
            
            for bay_id, bay_info in bays.items():
                if bay_id.startswith('Bay'):  # Only show service bays
                    status = bay_info.get('status', 'available')
                    occupied_trains = bay_info.get('occupied_trains', [])
                    max_capacity = bay_info.get('max_capacity', 2)
                    
                    # Status color and icon
                    if status == 'available':
                        status_color = '#4CAF50'
                        status_icon = '🟢'
                    elif status == 'partial':
                        status_color = '#FF9800'
                        status_icon = '🟡'
                    elif status == 'occupied':
                        status_color = '#F44336'
                        status_icon = '🔴'
                    else:
                        status_color = '#9E9E9E'
                        status_icon = '⚪'
                    
                    bay_items.append(
                        html.Div([
                            html.Div([
                                html.Span(status_icon, className="me-2"),
                                html.Strong(bay_id, className="text-primary"),
                                html.Span(f" ({len(occupied_trains)}/{max_capacity})", 
                                         className="text-muted ms-2")
                            ], className="d-flex align-items-center mb-1"),
                            html.Div([
                                html.Small(status.upper(), 
                                         style={'color': status_color, 'fontWeight': 'bold'})
                            ]),
                            html.Hr(className="my-2")
                        ])
                    )
            
            if not bay_items:
                bay_items = [
                    html.Div("No bay information available", 
                           className="text-muted text-center p-3")
                ]
            
            return bay_items
            
        except Exception as e:
            return [html.Div(f"Error loading bay status: {str(e)}", className="text-danger")]
    
    def get_movement_event_feed(self):
        """Get live movement event feed"""
        try:
            # Generate simulated events (in real system, these would be actual events)
            current_time = datetime.now()
            events = []
            
            # Recent events simulation
            for i in range(5):
                event_time = current_time - timedelta(minutes=i*2)
                events.append(
                    html.Div([
                        html.Div([
                            html.Small(event_time.strftime('%H:%M:%S'), className="text-muted"),
                            html.Span(" • ", className="text-muted mx-2"),
                            html.Strong(f"KMRL_{100+i}", className="text-primary"),
                            html.Span(" started moving to Bay1", className="ms-2")
                        ], className="mb-2 p-2", 
                        style={'backgroundColor': '#F8F9FA', 'borderRadius': '4px'})
                    ])
                )
            
            return events
            
        except Exception as e:
            return [html.Div(f"Error loading events: {str(e)}", className="text-danger")]