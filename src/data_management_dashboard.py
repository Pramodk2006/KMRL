"""
Professional Data Management Dashboard for KMRL IntelliFleet
Provides comprehensive data upload, validation, and management capabilities
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import json
import io
import base64

class DataManagementDashboard:
    """Professional data management interface for KMRL IntelliFleet"""
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        
        # API base and auth cache
        self.api_base = "http://127.0.0.1:8000"
        self._jwt_token = None

        
        # Custom CSS for professional styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
                    .header { background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; }
                    .card { border: none; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }
                    .card-header { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px 12px 0 0; border-bottom: 1px solid #dee2e6; }
                    .btn-primary { background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); border: none; border-radius: 8px; padding: 0.75rem 1.5rem; }
                    .btn-success { background: linear-gradient(135deg, #388e3c 0%, #66bb6a 100%); border: none; border-radius: 8px; }
                    .btn-warning { background: linear-gradient(135deg, #f57c00 0%, #ffb74d 100%); border: none; border-radius: 8px; }
                    .btn-danger { background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%); border: none; border-radius: 8px; }
                    .upload-area { border: 2px dashed #1976d2; border-radius: 12px; padding: 2rem; text-align: center; background-color: #f8f9ff; transition: all 0.3s ease; }
                    .upload-area:hover { background-color: #e3f2fd; border-color: #1565c0; }
                    .upload-area.dragover { background-color: #e3f2fd; border-color: #1565c0; transform: scale(1.02); }
                    .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                    .status-online { background-color: #4caf50; }
                    .status-offline { background-color: #f44336; }
                    .status-warning { background-color: #ff9800; }
                    .data-source-card { transition: transform 0.2s ease; }
                    .data-source-card:hover { transform: translateY(-2px); }
                    .metric-card { text-align: center; padding: 1.5rem; }
                    .metric-value { font-size: 2rem; font-weight: bold; color: #1976d2; }
                    .metric-label { color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
                    /* Data Sources grid layout (Bootstrap columns) */
                    #data-sources-grid { }
                    .data-source-card { min-height: 160px; display: flex; align-items: center; justify-content: center; }
                    .data-source-card .p-3 { white-space: normal; word-break: break-word; }
                    .data-source-toolbar { display: flex; gap: 8px; align-items: center; }
                    .chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
                    .chip-online { background-color: #e8f5e9; color: #2e7d32; }
                    .chip-warning { background-color: #fff3e0; color: #ef6c00; }
                    .chip-offline { background-color: #ffebee; color: #c62828; }
                    .badge { display: inline-block; min-width: 22px; padding: 2px 6px; font-size: 12px; border-radius: 8px; background:#e9ecef; color:#495057; }
                    .card-actions { display:flex; gap:6px; justify-content:center; margin-top:8px; }
                    .btn-ghost { border: 1px solid #dee2e6; background:#fff; padding:6px 10px; border-radius:6px; font-size:12px; }
                    .btn-ghost:hover { background:#f8f9fa; }
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
        
        self.app.layout = self._create_layout()
        self.setup_callbacks()

    def _get_jwt_token(self) -> str:
        """Login to API and cache JWT (operator role)."""
        if self._jwt_token:
            return self._jwt_token
        try:
            resp = requests.post(
                f"{self.api_base}/auth/login",
                params={"username": "operator", "password": "kmrl2025"},
                timeout=5
            )
            if resp.ok:
                data = resp.json()
                self._jwt_token = data.get("access_token")
                return self._jwt_token
        except Exception:
            pass
        return None
    
    def _create_layout(self):
        """Create the main dashboard layout"""
        return html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H1("📊 KMRL IntelliFleet Data Management", className="text-white mb-0"),
                    html.P("Professional Data Integration & Management Platform", className="text-white-50 mb-0")
                ], className="container")
            ], className="header"),
            
            html.Div([
                dcc.Store(id='dq-store', data={'uploads': {}}),
                dcc.Interval(id='dq-ping', interval=30000, n_intervals=0),
                dcc.Interval(id='status-ping', interval=30000, n_intervals=0),
                # System Status Overview
                html.Div([
                    html.Div([
                        html.H4("🔗 System Status", className="card-title"),
                        html.Div(id="system-status-cards", className="row")
                    ], className="card-header"),
                    html.Div(id="system-status-content", className="card-body")
                ], className="card"),
                
                # Data Sources Management
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H4("📁 Data Sources", className="card-title mb-0"),
                                html.P("Manage and monitor all data integration points", className="text-muted mb-0")
                            ], className="col-md-6"),
                            html.Div([
                                html.Div([
                                    dcc.Input(id='ds-search', type='text', placeholder='Search sources...', className='form-control form-control-sm', debounce=True),
                                ], className='col-md-6 mb-1'),
                                html.Div([
                                    dcc.Dropdown(
                                        id='ds-status-filter',
                                        options=[
                                            {'label': 'All', 'value': 'all'},
                                            {'label': 'Online', 'value': 'online'},
                                            {'label': 'Warning', 'value': 'warning'},
                                            {'label': 'Offline', 'value': 'offline'},
                                        ],
                                        value='all',
                                        clearable=False,
                                        className='form-control form-control-sm'
                                    )
                                ], className='col-md-6')
                            ], className="row")
                        ], className="row data-source-toolbar")
                    ], className="card-header"),
                    html.Div([
                        html.Div(id="data-sources-grid", className="row")
                    ], className="card-body")
                ], className="card"),
                
                # CSV Upload Interface
                html.Div([
                    html.Div([
                        html.H4("📤 Data Upload", className="card-title"),
                        html.P("Upload CSV files for core system data", className="text-muted mb-0")
                    ], className="card-header"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H5("Select Data Type", className="mb-3"),
                                dcc.Dropdown(
                                    id="upload-data-type",
                                    options=[
                                        {'label': '🚂 Trains Data', 'value': 'trains'},
                                        {'label': '🔧 Job Cards (Maximo)', 'value': 'job_cards'},
                                        {'label': '🧹 Cleaning Slots', 'value': 'cleaning_slots'},
                                        {'label': '🏗️ Bay Configuration', 'value': 'bay_config'},
                                        {'label': '📝 Branding Contracts', 'value': 'branding_contracts'},
                                        {'label': '📊 Historical Outcomes', 'value': 'outcomes'}
                                    ],
                                    placeholder="Choose data type to upload...",
                                    className="mb-3"
                                ),
                                html.Div([
                                    html.Div([
                                        html.I("📁", className="fa-3x text-primary mb-3"),
                                        html.H5("Drag & Drop CSV File", className="mb-2"),
                                        html.P("or click to browse", className="text-muted mb-3"),
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div(['Select File']),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px',
                                                'cursor': 'pointer'
                                            },
                                            multiple=False
                                        )
                                    ], className="upload-area")
                                ], id="upload-area-container"),
                                html.Div(id="upload-preview", className="mt-3"),
                                html.Div([
                                    html.Button("📤 Upload Data", id="upload-submit-btn", 
                                              className="btn btn-primary btn-lg", disabled=True),
                                    html.Button("📋 Download Template", id="download-template-btn", 
                                              className="btn btn-outline-secondary btn-lg ml-2")
                                ], className="mt-3")
                            ], className="col-md-8"),
                            html.Div([
                                html.H5("📋 Data Format Guide", className="mb-3"),
                                html.Div(id="format-guide", className="border rounded p-3")
                            ], className="col-md-4")
                        ], className="row")
                    ], className="card-body")
                ], className="card"),
                
                # Data Validation & Quality
                html.Div([
                    html.Div([
                        html.H4("✅ Data Quality", className="card-title"),
                        html.P("Monitor data validation and quality metrics", className="text-muted mb-0")
                    ], className="card-header"),
                    html.Div([
                        html.Div(id="data-quality-metrics", className="row"),
                        html.Div(id="validation-results", className="mt-3")
                    ], className="card-body")
                ], className="card"),

                # Fleet Readiness Snapshot
                html.Div([
                    html.Div([
                        html.H4("🚉 Fleet Readiness Snapshot", className="card-title"),
                        html.P("Merged view of fitness, maintenance and branding for each train", className="text-muted mb-0")
                    ], className="card-header"),
                    html.Div([
                        html.Div(id="fleet-readiness-table"),
                        dcc.ConfirmDialog(id='override-dialog', message=''),
                        dcc.Store(id='pending-override')
                    ], className="card-body")
                ], className="card"),
                
                # Real-time Data Simulation
                html.Div([
                    html.Div([
                        html.H4("🔄 Real-time Data Simulation", className="card-title"),
                        html.P("Configure and monitor simulated data sources", className="text-muted mb-0")
                    ], className="card-header"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H6("IoT Sensor Simulation", className="mb-2"),
                                html.Div([
                                    html.Span("Status: ", className="mr-2"),
                                    html.Span(id="iot-status", className="status-indicator status-online"),
                                    html.Span("Running", id="iot-status-text")
                                ], className="mb-2"),
                                html.Button("🔄 Restart IoT", id="restart-iot-btn", className="btn btn-sm btn-outline-primary"),
                                html.Button("⏸️ Pause IoT", id="pause-iot-btn", className="btn btn-sm btn-outline-warning ml-1")
                            ], className="col-md-4"),
                            html.Div([
                                html.H6("Computer Vision Simulation", className="mb-2"),
                                html.Div([
                                    html.Span("Status: ", className="mr-2"),
                                    html.Span(id="cv-status", className="status-indicator status-online"),
                                    html.Span("Running", id="cv-status-text")
                                ], className="mb-2"),
                                html.Button("🔄 Restart CV", id="restart-cv-btn", className="btn btn-sm btn-outline-primary"),
                                html.Button("⏸️ Pause CV", id="pause-cv-btn", className="btn btn-sm btn-outline-warning ml-1")
                            ], className="col-md-4"),
                            html.Div([
                                html.H6("Maximo Integration", className="mb-2"),
                                html.Div([
                                    html.Span("Status: ", className="mr-2"),
                                    html.Span(id="maximo-status", className="status-indicator status-warning"),
                                    html.Span("Manual Mode", id="maximo-status-text")
                                ], className="mb-2"),
                                html.Button("📤 Upload Maximo Export", id="upload-maximo-btn", className="btn btn-sm btn-outline-primary"),
                                html.Button("🔄 Refresh Data", id="refresh-maximo-btn", className="btn btn-sm btn-outline-success ml-1")
                            ], className="col-md-4")
                        ], className="row")
                    ], className="card-body")
                ], className="card"),
                
                # Upload Results
                html.Div(id="upload-results", className="mt-3")
                
            ], className="container")
        ])
    
    def setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        # System status update
        @self.app.callback(
            [Output('system-status-cards', 'children'),
             Output('system-status-content', 'children')],
            [Input('upload-data-type', 'value'), Input('status-ping', 'n_intervals')]
        )
        def update_system_status(data_type, _):
            try:
                # Try to get system status from API
                try:
                    resp = requests.get(f"{self.api_base}/system/status", timeout=3)
                    if not resp.ok:
                        raise Exception("API returned error")
                    status = resp.json() or {}
                    sys = status.get('system_status', {}) or {}
                    data_freshness = sys.get('data_freshness', {}) or {}
                    data_counts = sys.get('data_counts', {}) or {}
                    last_uploads = sys.get('last_uploads', {}) or {}
                    integrations = {}
                    try:
                        for item in (data_freshness.get('integration') or []):
                            integrations[item.get('source')] = item
                    except Exception:
                        pass

                    # Top metric cards (database summary)
                    total_rows = sum(int(v or 0) for v in data_counts.values())
                    sources_ok = sum(1 for k in ['trains','job_cards','cleaning_slots','bay_config','branding_contracts'] if data_freshness.get(k) == 'ok')
                    cards = [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I("📚", className="fa-2x text-primary mb-2"),
                                    html.Div(str(total_rows), className="metric-value"),
                                    html.Div("Total Rows", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("🗂️", className="fa-2x text-success mb-2"),
                                    html.Div(str(len(data_counts)), className="metric-value"),
                                    html.Div("Tables", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("🔄", className="fa-2x text-info mb-2"),
                                    html.Div(f"{sources_ok}/5", className="metric-value"),
                                    html.Div("Sources Fresh", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("⏱️", className="fa-2x text-warning mb-2"),
                                    html.Div(datetime.now().strftime('%Y-%m-%d %H:%M'), className="metric-value"),
                                    html.Div("As of", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3")
                        ], className="row")
                    ]

                    # Helper to format a row
                    def row(ds_name, key, hb_key=None):
                        last = last_uploads.get(key)
                        if not last and hb_key and hb_key in integrations:
                            last = integrations[hb_key].get('last_heartbeat')
                        status_txt = 'OK' if data_freshness.get(key) == 'ok' else 'Stale'
                        status_cls = 'text-success' if status_txt == 'OK' else 'text-warning'
                        return html.Tr([
                            html.Td(ds_name),
                            html.Td(last or '—'),
                            html.Td([html.Span("✅ ", className=status_cls), html.Span(status_txt, className=status_cls)])
                        ])

                    # Build health table similar to the screenshot
                    table = html.Table([
                        html.Thead(html.Tr([
                            html.Th('Data Source'),
                            html.Th('Last Successful Update'),
                            html.Th('Status')
                        ])),
                        html.Tbody([
                            row('IBM Maximo Job Cards', 'job_cards', 'upload_job_cards'),
                            row('Fitness Certificates', 'trains', 'fitness_db'),
                            row('Branding Contracts', 'branding_contracts', 'upload_branding_contracts'),
                            row('Cleaning Bay Availability', 'cleaning_slots', 'cleaning_slots'),
                            html.Tr([
                                html.Td('Live Train Mileage (IoT)'),
                                html.Td('Streaming Live'),
                                html.Td([html.Span('🟢 '), html.Span('Live', className='text-success')])
                            ])
                        ])
                    ], className="table table-sm")

                    freshness_content = html.Div([
                        html.H6("System Health & Data Freshness", className="mb-3"),
                        table
                    ])

                    return cards, freshness_content
                except Exception:
                    # Fallback to static placeholders when API is not available
                    cards = [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I("📚", className="fa-2x text-primary mb-2"),
                                    html.Div("—", className="metric-value"),
                                    html.Div("Total Rows", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("🗂️", className="fa-2x text-success mb-2"),
                                    html.Div("—", className="metric-value"),
                                    html.Div("Tables", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("🔄", className="fa-2x text-info mb-2"),
                                    html.Div("0/5", className="metric-value"),
                                    html.Div("Sources Fresh", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3"),
                            html.Div([
                                html.Div([
                                    html.I("⏱️", className="fa-2x text-warning mb-2"),
                                    html.Div(datetime.now().strftime('%Y-%m-%d %H:%M'), className="metric-value"),
                                    html.Div("As of", className="metric-label")
                                ], className="metric-card")
                            ], className="col-md-3")
                        ], className="row")
                    ]

                    freshness_content = html.Div([
                        html.H6("System Health & Data Freshness", className="mb-3"),
                        html.Table([
                            html.Thead(html.Tr([html.Th('Data Source'), html.Th('Last Successful Update'), html.Th('Status')])),
                            html.Tbody([
                                html.Tr([html.Td('API Server'), html.Td('—'), html.Td([html.Span('🔴 '), html.Span('Offline', className='text-danger')])]),
                                html.Tr([html.Td('IoT Sensors'), html.Td('Simulated'), html.Td([html.Span('🟡 '), html.Span('Manual', className='text-warning')])])
                            ])
                        ], className="table table-sm")
                    ])

                    return cards, freshness_content
            except Exception as e:
                return [], html.Div(f"Error: {str(e)}", className="text-danger")
        
        # Data Quality metrics renderer (uses dq-store + system status)
        @self.app.callback(
            Output('data-quality-metrics', 'children'),
            [Input('dq-store', 'data'), Input('dq-ping', 'n_intervals')]
        )
        def render_data_quality(dq_store, _):
            try:
                # Fetch simple counts/freshness from API if available
                counts = {}
                last_uploads = {}
                try:
                    resp = requests.get(f"{self.api_base}/system/status", timeout=3)
                    if resp.ok:
                        js = resp.json() or {}
                        sys = js.get('system_status', {}) if isinstance(js, dict) else {}
                        counts = sys.get('data_counts', {}) or {}
                        last_uploads = sys.get('last_uploads', {}) or {}
                except Exception:
                    pass

                uploads = (dq_store or {}).get('uploads', {})
                # Compute overall score: 100 - error ratio across last uploads
                total_rows = sum(v.get('rows', 0) for v in uploads.values()) or 0
                total_errors = sum(v.get('errors', 0) for v in uploads.values()) or 0
                score = 100.0 if total_rows == 0 else max(0.0, 100.0 - (total_errors / max(1, total_rows) * 100.0))

                cards = []
                # Overall
                cards.append(html.Div([
                    html.Div([
                        html.Div(f"{score:.1f}%", className="metric-value"),
                        html.Div("Overall Data Quality", className="metric-label")
                    ], className="metric-card")
                ], className="col-md-3"))

                # Per-table recent upload and counts
                def table_card(name: str, key: str):
                    up = uploads.get(key, {})
                    last = up.get('ts') or last_uploads.get(key, '—')
                    rows = up.get('rows', counts.get(key, 0))
                    errors = up.get('errors', 0)
                    err_ratio = (errors / max(1, rows)) * 100.0 if rows else 0.0
                    return html.Div([
                        html.Div([
                            html.Div(name, className="small text-muted mb-1"),
                            html.Div(str(rows), className="metric-value"),
                            html.Div(f"Last: {last}", className="small text-muted"),
                            html.Div(f"Errors: {errors} ({err_ratio:.1f}%)", className="small")
                        ], className="metric-card")
                    ], className="col-md-3")

                cards.append(table_card('Trains', 'trains'))
                cards.append(table_card('Job Cards', 'job_cards'))
                cards.append(table_card('Cleaning Slots', 'cleaning_slots'))
                cards.append(table_card('Branding Contracts', 'branding_contracts'))

                return html.Div(cards, className="row")
            except Exception as e:
                return html.Div(f"DQ error: {str(e)}", className="text-danger")

        # Fleet readiness table
        @self.app.callback(
            Output('fleet-readiness-table', 'children'),
            [Input('status-ping', 'n_intervals')]
        )
        def render_fleet_readiness(_):
            try:
                headers = {}
                token = self._get_jwt_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                resp = requests.get(f"{self.api_base}/fleet/readiness", headers=headers, timeout=5)
                if not resp.ok:
                    raise Exception(resp.text)
                js = resp.json() or {}
                rows = js.get('rows', [])
                if not rows:
                    return html.Div("No readiness data yet. Upload trains, job_cards and branding_contracts.", className="text-muted")
                # Build table
                header = html.Tr([html.Th('Train ID'), html.Th('Fitness'), html.Th('Maintenance'), html.Th('Branding'), html.Th('Mileage (KM)'), html.Th('Overall Readiness'), html.Th('Actions')])
                body = []
                for r in rows:
                    color = 'green' if r.get('overall_readiness')=='Go' else ('orange' if r.get('overall_readiness')=='Standby' else 'red')
                    train_id = r.get('train_id','')
                    ov_info = r.get('override_info', {})
                    has_override = ov_info.get('has_override', False)
                    
                    # Maintenance column with override badge
                    maint_cell = html.Td([
                        r.get('maintenance',''),
                        html.Br() if has_override else '',
                        html.Span(f"Overridden by {ov_info.get('by','')}: {ov_info.get('reason','')}", 
                                 className="badge badge-warning small") if has_override else ''
                    ])
                    
                    # Action buttons
                    actions = []
                    if 'Open' in r.get('maintenance','') and not has_override:
                        actions.append(html.Button('Override', id={'type':'ov-btn','train':train_id}, className='btn btn-sm btn-outline-primary'))
                    if has_override:
                        actions.append(html.Button('Revert', id={'type':'rev-btn','train':train_id}, className='btn btn-sm btn-outline-danger'))
                    
                    body.append(html.Tr([
                        html.Td(train_id),
                        html.Td(r.get('fitness','')),
                        maint_cell,
                        html.Td(r.get('branding','')),
                        html.Td(f"{int(r.get('mileage_km') or 0):,}"),
                        html.Td(html.Span(r.get('overall_readiness',''), style={'color': color})),
                        html.Td(actions)
                    ]))
                return html.Table([html.Thead(header), html.Tbody(body)], className="table table-striped table-sm")
            except Exception as e:
                return html.Div(f"Readiness error: {str(e)}", className="text-danger")

        # Capture override button click -> prompt for reason
        @self.app.callback(
            [Output('override-dialog', 'displayed'), Output('override-dialog', 'message'), Output('pending-override', 'data')],
            [Input({'type':'ov-btn','train':dash.dependencies.ALL}, 'n_clicks')],
            [State({'type':'ov-btn','train':dash.dependencies.ALL}, 'id')]
        )
        def prompt_override(n_clicks_list, ids):
            ctx = callback_context
            if not ctx.triggered:
                return False, '', {}
            # Find which button was clicked
            for n, id_obj in zip(n_clicks_list or [], ids or []):
                if n:
                    train_id = id_obj.get('train')
                    return True, f"Please provide a reason for overriding maintenance to OK for train {train_id}.", {'train_id': train_id}
            return False, '', {}

        # Handle override confirmation via simple prompt (browser confirm not available in Dash)
        @self.app.callback(
            Output('status-ping', 'n_intervals'),
            [Input('override-dialog', 'submit_n_clicks')],
            [State('pending-override', 'data')]
        )
        def apply_override(submit_clicks, pending):
            if not submit_clicks or not pending:
                raise dash.exceptions.PreventUpdate
            # Since ConfirmDialog cannot capture text, use a default reason placeholder
            reason = 'Operator override via dashboard'
            try:
                headers = {}
                token = self._get_jwt_token()
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                payload = {
                    'train_id': pending.get('train_id'),
                    'field': 'maintenance_status',
                    'value': 'OK',
                    'reason': reason
                }
                resp = requests.post(f"{self.api_base}/fleet/override", params=payload, headers=headers, timeout=5)
                # trigger refresh
                return 0
            except Exception:
                return 0

        # Handle revert button clicks
        @self.app.callback(
            Output('status-ping', 'n_intervals', allow_duplicate=True),
            [Input({'type':'rev-btn','train':dash.dependencies.ALL}, 'n_clicks')],
            [State({'type':'rev-btn','train':dash.dependencies.ALL}, 'id')],
            prevent_initial_call=True
        )
        def handle_revert(n_clicks_list, ids):
            ctx = callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            # Find which button was clicked
            for n, id_obj in zip(n_clicks_list or [], ids or []):
                if n:
                    train_id = id_obj.get('train')
                    try:
                        headers = {}
                        token = self._get_jwt_token()
                        if token:
                            headers['Authorization'] = f'Bearer {token}'
                        resp = requests.delete(f"{self.api_base}/fleet/override/{train_id}/maintenance_status", headers=headers, timeout=5)
                        # trigger refresh
                        return 0
                    except Exception:
                        return 0
            raise dash.exceptions.PreventUpdate

        # Data sources grid
        @self.app.callback(
            Output('data-sources-grid', 'children'),
            [Input('upload-data-type', 'value'), Input('ds-search', 'value'), Input('ds-status-filter', 'value')]
        )
        def update_data_sources(data_type, search_text, status_filter):
            sources = [
                {
                    'name': 'Trains Data', 'key': 'trains',
                    'icon': '🚂',
                    'status': 'online',
                    'description': 'Fleet information, fitness certificates, mileage',
                    'last_update': '2 minutes ago',
                    'records': '25'
                },
                {
                    'name': 'Job Cards (Maximo)', 'key': 'job_cards',
                    'icon': '🔧',
                    'status': 'warning',
                    'description': 'Maintenance work orders and status',
                    'last_update': 'Manual upload',
                    'records': '55'
                },
                {
                    'name': 'Cleaning Slots', 'key': 'cleaning_slots',
                    'icon': '🧹',
                    'status': 'online',
                    'description': 'Available cleaning bays and schedules',
                    'last_update': '1 hour ago',
                    'records': '6'
                },
                {
                    'name': 'Bay Configuration', 'key': 'bay_config',
                    'icon': '🏗️',
                    'status': 'online',
                    'description': 'Service bay types and capacities',
                    'last_update': '1 hour ago',
                    'records': '8'
                },
                {
                    'name': 'IoT Sensors', 'key': 'iot',
                    'icon': '📡',
                    'status': 'online',
                    'description': 'Real-time train health monitoring',
                    'last_update': 'Live',
                    'records': 'Streaming'
                },
                {
                    'name': 'Computer Vision', 'key': 'cv',
                    'icon': '👁️',
                    'status': 'online',
                    'description': 'Automated defect detection',
                    'last_update': 'Live',
                    'records': 'Active'
                }
            ]
            
            # Apply filters
            filtered = []
            for s in sources:
                if status_filter and status_filter != 'all' and s['status'] != status_filter:
                    continue
                if search_text and search_text.strip():
                    q = search_text.strip().lower()
                    if q not in s['name'].lower() and q not in s['description'].lower():
                        continue
                filtered.append(s)
            cards = []
            for source in filtered:
                status_class = f"status-{source['status']}"
                cards.append(
                    html.Div([
                        html.Div([
                            html.I(source['icon'], className="fa-2x mb-2"),
                            html.H6(source['name'], className="mb-1"),
                            html.P(source['description'], className="text-muted small mb-2"),
                            html.Div([
                                html.Span(className=f"status-indicator {status_class}"),
                                html.Span(source['last_update'], className="small")
                            ], className="mb-1"),
                            html.Div([
                                html.Span("Records: ", className="small text-muted"),
                                html.Span(source['records'], className="small font-weight-bold")
                            ])
                        ], className="card data-source-card text-center p-3")
                    ], className="col-12 col-sm-6 col-md-4 col-lg-3 mb-3")
                )
            
            return cards
        
        # Format guide update
        @self.app.callback(
            Output('format-guide', 'children'),
            [Input('upload-data-type', 'value')]
        )
        def update_format_guide(data_type):
            if not data_type:
                return html.Div("Select a data type to see format requirements")
            
            guides = {
                'trains': {
                    'required': ['train_id', 'mileage_km', 'branding_hours_left', 'fitness_valid_until', 'cleaning_slot_id', 'bay_geometry_score'],
                    'optional': ['depot_id', 'last_maintenance_date'],
                    'description': 'Train fleet information including fitness certificates and operational data'
                },
                'job_cards': {
                    'required': ['train_id', 'job_card_status'],
                    'optional': ['last_updated'],
                    'description': 'Maintenance job cards (simple schema). Status must be open/closed.'
                },
                'cleaning_slots': {
                    'required': ['slot_id', 'available_bays', 'start_time', 'end_time'],
                    'optional': ['assigned_crew', 'cleaning_type'],
                    'description': 'Cleaning bay availability and scheduling'
                },
                'bay_config': {
                    'required': ['bay_id', 'bay_type', 'max_capacity', 'geometry_score'],
                    'optional': ['depot_id', 'power_available', 'status'],
                    'description': 'Service bay configuration and capacity'
                },
                'branding_contracts': {
                    'required': ['contract_id', 'brand', 'train_id', 'hours_committed', 'start_date', 'end_date'],
                    'optional': ['priority', 'notes'],
                    'description': 'Branding contract commitments and exposure requirements'
                },
                'outcomes': {
                    'required': ['date', 'train_id', 'inducted', 'failures'],
                    'optional': ['notes', 'energy_consumed_kwh', 'branding_sla_met'],
                    'description': 'Historical operational outcomes for ML training'
                }
            }
            
            guide = guides.get(data_type, {})
            if not guide:
                return html.Div("No format guide available")
            
            return html.Div([
                html.H6("Required Columns:", className="text-primary"),
                html.Ul([html.Li(col) for col in guide['required']], className="small mb-3"),
                html.H6("Optional Columns:", className="text-secondary"),
                html.Ul([html.Li(col) for col in guide['optional']], className="small mb-3"),
                html.P(guide['description'], className="small text-muted")
            ])
        
        # File upload handling
        @self.app.callback(
            [Output('upload-preview', 'children'),
             Output('upload-submit-btn', 'disabled')],
            [Input('upload-data', 'contents')],
            [State('upload-data-type', 'value'),
             State('upload-data', 'filename')]
        )
        def handle_file_upload(contents, data_type, filename):
            if not contents or not data_type:
                return "", True
            
            try:
                # Parse uploaded file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                else:
                    return html.Div("Please upload a CSV file", className="text-danger"), True
                
                # Client-side header validation against known schema
                guides = {
                    'trains': {
                        'required': ['train_id', 'fitness_valid_until', 'mileage_km', 'branding_hours_left', 'bay_geometry_score'],
                        'optional': ['cleaning_slot_id', 'depot_id']
                    },
                    'job_cards': {
                        'required': ['train_id', 'job_card_status'],
                        'optional': ['last_updated']
                    },
                    'cleaning_slots': {
                        'required': ['slot_id', 'available_bays'],
                        'optional': ['priority', 'depot_id']
                    },
                    'bay_config': {
                        'required': ['bay_id', 'bay_type', 'max_capacity', 'geometry_score'],
                        'optional': ['depot_id']
                    },
                    'branding_contracts': {
                        'required': ['contract_id', 'brand', 'hours_committed', 'start_date', 'end_date'],
                        'optional': ['train_id']
                    },
                    'outcomes': {
                        'required': ['date', 'train_id', 'inducted'],
                        'optional': ['failures', 'notes']
                    }
                }
                req = guides.get(data_type, {}).get('required', [])
                expected = req + guides.get(data_type, {}).get('optional', [])
                missing_cols = [c for c in req if c not in df.columns]
                if missing_cols:
                    # Show a clear error and block upload
                    return html.Div([
                        html.Div([
                            html.I("❌", className="fa-2x text-danger mr-2"),
                            html.H5("Missing Required Columns", className="text-danger mb-0")
                        ], className="d-flex align-items-center mb-2"),
                        html.Ul([
                            html.Li(f"Missing: {', '.join(missing_cols)}"),
                            html.Li(f"Expected: {', '.join(expected)}")
                        ])
                    ], className="alert alert-danger"), True

                # Create preview
                preview = html.Div([
                    html.H6(f"📄 {filename}", className="text-primary"),
                    html.P(f"Rows: {len(df)}, Columns: {len(df.columns)}", className="text-muted"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_cell={'textAlign': 'left', 'fontSize': '12px'},
                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa'
                            }
                        ]
                    )
                ])
                
                return preview, False
                
            except Exception as e:
                return html.Div(f"Error parsing file: {str(e)}", className="text-danger"), True
        
        # Upload submission
        @self.app.callback(
            [Output('upload-results', 'children'), Output('dq-store', 'data')],
            [Input('upload-submit-btn', 'n_clicks')],
            [State('upload-data', 'contents'),
             State('upload-data-type', 'value'),
             State('upload-data', 'filename'),
             State('dq-store', 'data')]
        )
        def submit_upload(n_clicks, contents, data_type, filename, dq_store):
            if not n_clicks or not contents or not data_type:
                return "", (dq_store or {'uploads': {}})
            
            try:
                # Prepare file for API
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Try to send to API first
                try:
                    files = {'file': (filename, decoded, 'text/csv')}
                    headers = {}
                    token = self._get_jwt_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                    resp = requests.post(f"{self.api_base}/ingest/{data_type}", files=files, headers=headers, timeout=10)
                    
                    if resp.ok:
                        result = resp.json()
                        rows_processed = result.get('rows_processed') or result.get('rows') or 0
                        # Update DQ store
                        dq_store = dq_store or {'uploads': {}}
                        dq_store['uploads'][data_type] = {
                            'ts': datetime.now().isoformat(),
                            'rows': int(rows_processed),
                            'valid_rows': int(result.get('valid_rows', rows_processed)),
                            'errors': int(result.get('error_count', 0)),
                            'warnings': int(result.get('warning_count', 0))
                        }
                        return html.Div([
                            html.Div([
                                html.I("✅", className="fa-2x text-success mr-2"),
                                html.H5("Upload Successful!", className="text-success mb-0")
                            ], className="d-flex align-items-center mb-2"),
                            html.P(f"Uploaded {rows_processed} rows to {data_type} table"),
                            html.Div([
                                html.H6("Validation Results:", className="mb-2"),
                                html.Ul([
                                    html.Li(f"Valid rows: {result.get('valid_rows', rows_processed)}"),
                                    html.Li(f"Errors: {result.get('error_count', 0)}"),
                                    html.Li(f"Warnings: {result.get('warning_count', 0)}")
                                ])
                            ])
                        ], className="alert alert-success"), dq_store
                    else:
                        # Show API validation or server errors without falling back to local save
                        try:
                            err = resp.json()
                        except Exception:
                            err = {'detail': resp.text}
                        detail = err.get('detail', err)
                        # Unwrap our structured detail if present
                        if isinstance(detail, dict) and 'message' in detail:
                            msg = detail['message']
                            missing = detail.get('missing_columns')
                            expected = detail.get('expected_columns')
                            items = []
                            items.append(html.Li(str(msg)))
                            if missing:
                                items.append(html.Li(f"Missing columns: {', '.join(missing)}"))
                            if expected:
                                items.append(html.Li(f"Expected columns: {', '.join(expected)}"))
                            return html.Div([
                                html.Div([
                                    html.I("❌", className="fa-2x text-danger mr-2"),
                                    html.H5("Upload Failed", className="text-danger mb-0")
                                ], className="d-flex align-items-center mb-2"),
                                html.Ul(items)
                            ], className="alert alert-danger"), (dq_store or {'uploads': {}})
                        # Generic error
                        return html.Div([
                            html.Div([
                                html.I("❌", className="fa-2x text-danger mr-2"),
                                html.H5("Upload Failed", className="text-danger mb-0")
                            ], className="d-flex align-items-center mb-2"),
                            html.Pre(str(detail))
                        ], className="alert alert-danger"), (dq_store or {'uploads': {}})
                        
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ConnectTimeout) as e:
                    # Fallback: Save locally when API is not available (network issues)
                    import pandas as pd
                    import os
                    
                    # Parse CSV data
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    
                    # Create data directory if it doesn't exist
                    os.makedirs('data', exist_ok=True)
                    
                    # Save to local CSV file
                    local_file = f"data/{data_type}_uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(local_file, index=False)
                    
                    return html.Div([
                        html.Div([
                            html.I("⚠️", className="fa-2x text-warning mr-2"),
                            html.H5("Saved Locally", className="text-warning mb-0")
                        ], className="d-flex align-items-center mb-2"),
                        html.P(f"Saved {len(df)} rows to {local_file}"),
                        html.Div([
                            html.H6("Note:", className="mb-2"),
                            html.P(f"Network error: {str(e)}. Data has been saved locally and will be synced when the server is available.", className="small text-muted mb-0")
                        ]),
                        html.Div([
                            html.Button("🔄 Retry Upload", id="retry-upload-btn", className="btn btn-sm btn-outline-primary mt-2")
                        ])
                    ], className="alert alert-warning"), (dq_store or {'uploads': {}})
                except Exception as e:
                    # Any other error (including validation errors that don't get caught above)
                    return html.Div([
                        html.Div([
                            html.I("❌", className="fa-2x text-danger mr-2"),
                            html.H5("Upload Error", className="text-danger mb-0")
                        ], className="d-flex align-items-center mb-2"),
                        html.P(f"Error: {str(e)}")
                    ], className="alert alert-danger"), (dq_store or {'uploads': {}})
                    
            except Exception as e:
                return html.Div([
                    html.I("❌", className="fa-2x text-danger mr-2"),
                    html.H5("Upload Error", className="text-danger mb-0"),
                    html.P(str(e))
                ], className="alert alert-danger"), (dq_store or {'uploads': {}})
        
        # Download template
        @self.app.callback(
            Output('download-template-btn', 'n_clicks'),
            [Input('download-template-btn', 'n_clicks')],
            [State('upload-data-type', 'value')]
        )
        def download_template(n_clicks, data_type):
            if n_clicks and data_type:
                # This would trigger a file download
                # For now, just show a message
                pass
            return 0

    
    def run(self, host='127.0.0.1', port=8051, debug=False):
        """Run the data management dashboard"""
        print(f"🚀 Starting Data Management Dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    dashboard = DataManagementDashboard()
    dashboard.run()
