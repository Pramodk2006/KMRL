"""
React Dashboard Backend for KMRL IntelliFleet
Serves React frontend and provides API endpoints
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import json
import logging

logger = logging.getLogger("ReactDashboard")

class KMRLReactDashboard:
    def __init__(self, ai_data_processor, digital_twin=None, optimizer=None, constraint_engine=None):
        self.ai_data_processor = ai_data_processor
        self.digital_twin = digital_twin
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        # In-memory supervisor decision state (deterministic derivation)
        # _decisions: train_id -> 'approve' | 'reject'
        self._decisions = {}
        self._base_plan = None   # ordered train_ids of inducted from optimizer
        self._standby_ids = None # ordered standby train_ids by score desc
        
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for development
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def dashboard():
            return self.get_html_template()
        
        
        @self.app.route('/api/induction_plan.csv')
        def download_induction_plan():
            try:
                # Build CSV of inducted + backup + maintenance using effective details
                details = self._get_effective_train_details()
                inducted = [t for t in details if t.get('inducted')]
                backup = [t for t in details if not t.get('inducted') and str(t.get('status','')).lower() == 'standby']
                maintenance = [t for t in details if str(t.get('status','')).lower() in ['maintenance', 'ineligible']]
                import io, csv
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(['status', 'rank', 'train_id', 'bay', 'score'])
                for t in inducted:
                    writer.writerow(['Inducted', t.get('rank',''), t.get('train_id',''), t.get('bay_assignment',''), t.get('priority_score','')])
                for t in backup:
                    writer.writerow(['Backup', '', t.get('train_id',''), t.get('bay_assignment','N/A'), t.get('priority_score','')])
                for t in maintenance:
                    writer.writerow(['Maintenance', '', t.get('train_id',''), 'N/A', t.get('priority_score','')])
                csv_data = buf.getvalue()
                from flask import Response
                return Response(
                    csv_data,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=induction_plan.csv'}
                )
            except Exception as e:
                logger.error(f"CSV download error: {e}")
                return jsonify({'error': str(e)}), 500

        # Supervisor actions
        @self.app.route('/api/supervisor/approve', methods=['POST'])
        def supervisor_approve():
            try:
                data = request.get_json() or {}
                train_id = str(data.get('trainId', '')).strip()
                if not train_id:
                    return jsonify({'error': 'Missing trainId'}), 400
                self._init_baseline()
                self._decisions[train_id] = 'approve'
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Approve error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/supervisor/reject', methods=['POST'])
        def supervisor_reject():
            try:
                data = request.get_json() or {}
                train_id = str(data.get('trainId', '')).strip()
                if not train_id:
                    return jsonify({'error': 'Missing trainId'}), 400
                self._init_baseline()
                self._decisions[train_id] = 'reject'
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Reject error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/supervisor/undo', methods=['POST'])
        def supervisor_undo():
            try:
                data = request.get_json() or {}
                train_id = str(data.get('trainId', '')).strip()
                if not train_id:
                    return jsonify({'error': 'Missing trainId'}), 400

                if train_id in self._decisions:
                    del self._decisions[train_id]
                self._init_baseline()

                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Undo error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/enriched')
        def get_dashboard_data_enriched():
            try:
                data = {
                    'train_status_summary': self.ai_data_processor.get_train_status_summary(),
                    'detailed_train_list': self._get_effective_train_details(),
                    'performance_metrics': self.ai_data_processor.get_performance_metrics(),
                    'constraint_violations': self.ai_data_processor.get_constraint_violations()
                }
                # Enrich with resources + branding tracker derived from data loader
                try:
                    dl = getattr(self.ai_data_processor, 'data_loader', None)
                    ds = getattr(dl, 'data_sources', {}) if dl else {}
                    cleaning = ds.get('cleaning_slots')
                    bay_cfg = ds.get('bay_config')

                    cleaning_bays_total = int(cleaning['available_bays'].sum()) if cleaning is not None and not cleaning.empty else 0
                    service_bays_total = int(bay_cfg[bay_cfg['bay_type'] == 'service']['max_capacity'].sum()) if bay_cfg is not None and not bay_cfg.empty else 0

                    inducted = data['train_status_summary'].get('inducted_trains', 0)
                    service_bays_free = max(0, service_bays_total - inducted)

                    trains_df = ds.get('trains')
                    scheduled_cleaning = 0
                    if trains_df is not None and not trains_df.empty and 'cleaning_slot_id' in trains_df.columns:
                        scheduled_cleaning = int(trains_df['cleaning_slot_id'].notna().sum())

                    data['resource_summary'] = {
                        'cleaning_bays_total': cleaning_bays_total,
                        'service_bays_total': service_bays_total,
                        'service_bays_free': service_bays_free,
                        'scheduled_cleaning': scheduled_cleaning,
                        'projected_bottleneck': scheduled_cleaning > cleaning_bays_total
                    }

                    # Branding tracker: use branding_hours from train details
                    details = data['detailed_train_list'] or []
                    sla_target = 10.0
                    tracker = []
                    for t in details:
                        bh = float(t.get('branding_hours', 0) or 0)
                        remaining = max(0.0, bh)
                        exposure = max(0.0, sla_target - remaining)
                        # Risk buckets
                        if remaining <= 2:
                            risk = 'Green'
                        elif remaining <= 5:
                            risk = 'Yellow'
                        else:
                            risk = 'Red'
                        tracker.append({
                            'train_id': t.get('train_id'),
                            'required_hours': round(remaining, 1),
                            'current_exposure': round(exposure, 1),
                            'risk': risk
                        })
                    tracker.sort(key=lambda x: (-x['required_hours'], x['train_id']))
                    data['branding_tracker'] = tracker[:12]
                except Exception as _:
                    pass

                return jsonify(data)
            except Exception as e:
                logger.error(f"Error fetching dashboard data: {e}")
                return jsonify({'error': str(e)}), 500

    def _get_effective_train_details(self):
        """Derive effective plan from baseline + decisions; build full details list."""
        details = self.ai_data_processor.get_detailed_train_list() or []
        self._init_baseline()
        plan_ids = self._derive_effective_plan()

        by_id = {t.get('train_id'): t for t in details}
        # Build inducted rows first using derived plan
        inducted_list = []
        for rank, tid in enumerate(plan_ids, start=1):
            t = (by_id.get(tid) or {'train_id': tid}).copy()
            t['inducted'] = True
            t['status'] = 'Inducted'
            t['rank'] = rank
            inducted_list.append(t)

        # Add remaining trains (standby/maintenance) excluding those in plan
        others = [t for t in details if t.get('train_id') not in plan_ids]
        return inducted_list + others

    def _init_baseline(self):
        if self._base_plan is not None and self._standby_ids is not None:
            return
        res = getattr(self.optimizer, 'optimized_result', {}) or {}
        inducted = res.get('inducted_trains', [])
        standby = res.get('standby_trains', [])
        # Preserve optimizer inducted order
        self._base_plan = [t.get('train_id') for t in inducted if t.get('train_id')]
        # Standby sorted by composite score desc
        standby_sorted = sorted(standby, key=lambda x: x.get('composite_score', 0), reverse=True)
        self._standby_ids = [t.get('train_id') for t in standby_sorted if t.get('train_id')]

    def _derive_effective_plan(self):
        base = self._base_plan or []
        size = len(base)
        if size == 0:
            return []

        approved = [t for t in base if self._decisions.get(t) == 'approve']
        rejected = {t for t in base if self._decisions.get(t) == 'reject'}
        originals = [t for t in base if t not in approved and t not in rejected]
        plan = approved + originals

        if len(plan) < size:
            used = set(plan) | rejected
            for tid in self._standby_ids or []:
                if len(plan) >= size:
                    break
                if tid in used:
                    continue
                if tid in base and tid not in rejected:
                    continue
                plan.append(tid)

        return plan[:size]

        # Lightweight health endpoint for uptime checks / probes
        @self.app.route('/health')
        def health():
            return 'ok', 200
        
        @self.app.route('/api/dashboard/data')
        def get_dashboard_data():
            try:
                data = {
                    'train_status_summary': self.ai_data_processor.get_train_status_summary(),
                    'detailed_train_list': self.ai_data_processor.get_detailed_train_list(),
                    'performance_metrics': self.ai_data_processor.get_performance_metrics(),
                    'constraint_violations': self.ai_data_processor.get_constraint_violations()
                }
                return jsonify(data)
            except Exception as e:
                logger.error(f"Error fetching dashboard data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/supervisor/action', methods=['POST'])
        def supervisor_action():
            try:
                data = request.get_json()
                train_id = data.get('trainId')
                action = data.get('action')
                
                # Here you would integrate with your digital twin
                # For now, just log the action
                logger.info(f"Supervisor action: {action} for train {train_id}")
                
                return jsonify({'success': True, 'message': f'Action {action} applied to {train_id}'})
            except Exception as e:
                logger.error(f"Supervisor action error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/optimization/trigger', methods=['POST'])
        def trigger_optimization():
            try:
                data = request.get_json()
                branding_priority = data.get('brandingPriority', 75)
                mileage_balancing = data.get('mileageBalancing', 60)
                
                # Here you would trigger re-optimization with new parameters
                logger.info(f"Optimization triggered: branding={branding_priority}, mileage={mileage_balancing}")
                
                return jsonify({'success': True, 'message': 'Optimization triggered'})
            except Exception as e:
                logger.error(f"Optimization trigger error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def get_html_template(self):
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="data:;base64,iVBORw0KGgo=" />
    <title>KMRL IntelliFleet Dashboard</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <!-- Recharts UMD requires PropTypes on window -->
    <script src="https://unpkg.com/prop-types@15.8.1/prop-types.min.js"></script>
    <!-- Load Recharts after dependencies -->
    <script src="https://unpkg.com/recharts@2.8.0/umd/Recharts.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <!-- Tailwind via CDN for dev only -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .slider-green::-webkit-slider-thumb {
            appearance: none;
            height: 20px;
            width: 20px;
            border-radius: 50%;
            background: #00ff88;
            cursor: pointer;
            border: 2px solid #00ff88;
        }
        
        .slider-blue::-webkit-slider-thumb {
            appearance: none;
            height: 20px;
            width: 20px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
            border: 2px solid #00d4ff;
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            background: transparent;
            cursor: pointer;
        }
        
        input[type="range"]::-webkit-slider-track {
            background: #374151;
            height: 8px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;
        const { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, LineChart, Line, Tooltip } = window.Recharts || {};

        const KMRLDashboard = () => {
            const [currentTime, setCurrentTime] = useState(new Date());
            const [dashboardData, setDashboardData] = useState(null);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [brandingPriority, setBrandingPriority] = useState(75);
            const [mileageBalancing, setMileageBalancing] = useState(60);

            useEffect(() => {
                const timer = setInterval(() => setCurrentTime(new Date()), 1000);
                return () => clearInterval(timer);
            }, []);

            const fetchDashboardData = async () => {
                try {
                    setLoading(true);
                    // Prefer enriched endpoint if available
                    let response = await fetch('/api/dashboard/enriched');
                    if (!response.ok) {
                        response = await fetch('/api/dashboard/data');
                    }
                    const data = await response.json();
                    setDashboardData(data);
                    setError(null);
                } catch (err) {
                    setError('Failed to fetch dashboard data');
                    console.error('Dashboard data fetch error:', err);
                } finally {
                    setLoading(false);
                }
            };

            useEffect(() => {
                fetchDashboardData();
                const interval = setInterval(fetchDashboardData, 30000);
                return () => clearInterval(interval);
            }, []);

            const handleParameterChange = async (parameter, value) => {
                if (parameter === 'branding') {
                    setBrandingPriority(value);
                } else if (parameter === 'mileage') {
                    setMileageBalancing(value);
                }
                
                try {
                    await fetch('/api/optimization/trigger', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            brandingPriority: parameter === 'branding' ? value : brandingPriority,
                            mileageBalancing: parameter === 'mileage' ? value : mileageBalancing
                        })
                    });
                    
                    setTimeout(() => {
                        fetchDashboardData();
                    }, 2000);
                } catch (err) {
                    console.error('Parameter change failed:', err);
                }
            };

            const handleSupervisorAction = async (trainId, action) => {
                try {
                    await fetch('/api/supervisor/action', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ trainId, action })
                    });
                    fetchDashboardData();
                } catch (err) {
                    console.error('Supervisor action failed:', err);
                }
            };

            if (loading) {
                return (
                    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
                        <div className="text-white text-xl">Loading KMRL IntelliFleet...</div>
                    </div>
                );
            }

            if (error) {
                return (
                    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
                        <div className="text-red-400 text-xl">{error}</div>
                    </div>
                );
            }

            const { train_status_summary: summary, detailed_train_list: trainDetails, performance_metrics: performance, constraint_violations: violations, resource_summary: resources = {}, branding_tracker: branding = [] } = dashboardData;

            // Prepare fleet mileage data for chart
            // Deduplicate by train_id; last occurrence wins
            const byId = new Map();
            (trainDetails || []).forEach(t => {
                const id = String(t.train_id || '').trim();
                if (!id) return;
                byId.set(id, { 
                    name: id, 
                    mileage: Number(t.mileage_km || 0),
                    bay: t.bay_assignment || 'N/A',
                    score: Number(t.priority_score || 0)
                });
            });
            const mileageData = Array.from(byId.values()).sort((a, b) => b.mileage - a.mileage);

            const getScoreColor = (score) => {
                if (score >= 99) return '#00ff88';
                if (score >= 95) return '#ffaa00';
                return '#ff4444';
            };

            const getReasonFromScore = (train) => {
                if (train.induction_reason) return train.induction_reason;
                // Fallback heuristics if backend reason missing
                const parts = [];
                if (typeof train.service_readiness === 'number' && train.service_readiness >= 80) parts.push('High readiness');
                if (typeof train.maintenance_penalty === 'number' && train.maintenance_penalty <= 20) parts.push('Low maintenance risk');
                if (typeof train.branding_priority === 'number' && train.branding_priority >= 60) parts.push('Branding priority');
                if (typeof train.mileage_balance === 'number' && train.mileage_balance >= 70) parts.push('Mileage balanced');
                if (typeof train.shunting_cost === 'number' && train.shunting_cost <= 20) parts.push('Low shunting');
                return parts.length ? parts.slice(0,2).join(', ') : 'Balanced overall score';
            };

            return (
                <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
                    {/* Header */}
                    <div className="flex justify-between items-center mb-8">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-cyan-400 rounded-lg flex items-center justify-center">
                                <span className="text-slate-900 font-bold">K</span>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold">KMRL IntelliFleet - Planning Mode</h1>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <button 
                                onClick={fetchDashboardData}
                                className="p-2 rounded-lg bg-slate-700/50 hover:bg-slate-600/50 transition-colors"
                            >
                                ⟳
                            </button>
                            <div className="text-right">
                                <div className="text-2xl font-mono">
                                    {currentTime.toLocaleTimeString('en-IN', { 
                                        timeZone: 'Asia/Kolkata',
                                        hour12: false 
                                    })} IST
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-12 gap-6">
                        {/* Left Column */}
                        <div className="col-span-3 space-y-6">
                            {/* Summary */}
                            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                <h3 className="text-lg font-semibold mb-4">Summary</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-slate-700/30 rounded-lg p-4 text-center">
                                        <div className="text-sm text-slate-400 mb-1">Total Trains</div>
                                        <div className="text-3xl font-bold text-sky-300">{summary.total_trains}</div>
                                    </div>
                                    <div className="bg-cyan-500/20 rounded-lg p-4 text-center">
                                        <div className="text-sm text-slate-400 mb-1">Proposed for Service</div>
                                        <div className="text-3xl font-bold text-cyan-400">{summary.inducted_trains}</div>
                                    </div>
                                    <div className="bg-green-500/20 rounded-lg p-4 text-center">
                                        <div className="text-sm text-slate-400 mb-1">Backup</div>
                                        <div className="text-3xl font-bold text-green-400">{summary.ready_trains}</div>
                                    </div>
                                    <div className="bg-red-500/20 rounded-lg p-4 text-center">
                                        <div className="text-sm text-slate-400 mb-1">Maintenance</div>
                                        <div className="text-3xl font-bold text-red-400">{summary.maintenance_trains}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Standby / Backup Trains */}
                            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                <h3 className="text-lg font-semibold mb-4">Backup / Standby Trains</h3>
                                {(() => {
                                    const standby = (trainDetails || []).filter(t => (t.status || '').toLowerCase() === 'standby');
                                    if (!standby.length) {
                                        return <div className="text-slate-400 text-sm">No standby trains right now</div>;
                                    }
                                    return (
                                        <ul className="space-y-2 max-h-64 overflow-auto pr-1">
                                            {standby.slice(0, 12).map((t, i) => (
                                                <li key={t.train_id + i} className="flex items-center justify-between bg-slate-900/40 rounded px-3 py-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-slate-400 text-xs">{i+1}</span>
                                                        <span className="font-medium">{t.train_id}</span>
                                                    </div>
                                                    <span className="text-slate-300 text-sm">Score: {(t.priority_score || 0).toFixed(1)}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    );
                                })()}
                            </div>
                        </div>

                        {/* Middle Column - Main Content */}
                        <div className="col-span-9 space-y-6">
                            {/* CSV Download */}
                            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 flex items-center justify-between">
                                <h3 className="text-lg font-semibold">Tonight's Induction Plan</h3>
                                <button onClick={() => window.location.href='/api/induction_plan.csv'} className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 rounded text-sm">Download CSV</button>
                            </div>
                            {/* Proposed Induction Plan */}
                            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                <div className="flex justify-between items-center mb-6">
                                    <h3 className="text-lg font-semibold">Proposed Induction Plan</h3>
                                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                                        Review and apply actions to tonight's plan
                                    </div>
                                </div>
                                
                                <div className="overflow-hidden rounded-lg bg-slate-900/50">
                                    <table className="w-full">
                                        <thead className="bg-slate-700/50">
                                            <tr className="text-left">
                                                <th className="px-4 py-3 text-sm font-medium text-slate-300">Rank</th>
                                                <th className="px-4 py-3 text-sm font-medium text-slate-300">Train</th>
                                                <th className="px-4 py-3 text-sm font-medium text-slate-300">Bay</th>
                                                <th className="px-4 py-3 text-sm font-medium text-slate-300">Score</th>
                                                <th className="px-4 py-3 text-sm font-medium text-slate-300">Supervisor Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {trainDetails.slice(0, 10).map((train, index) => (
                                                <tr key={`${train.train_id}-${train.rank ?? index}`} className="border-t border-slate-700/30">
                                                    <td className="px-4 py-3">
                                                        <div className="flex items-center gap-2">
                                                            <input type="checkbox" defaultChecked className="w-4 h-4" />
                                                            <span className="font-medium">{train.rank}</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-4 py-3 font-medium">{train.train_id}</td>
                                                    <td className="px-4 py-3 text-slate-300">{train.bay_assignment}</td>
                                                    <td className="px-4 py-3">
                                                        <span 
                                                            className="px-3 py-1 rounded-full text-sm font-semibold"
                                                            style={{ 
                                                                backgroundColor: getScoreColor(train.priority_score) + '20',
                                                                color: getScoreColor(train.priority_score)
                                                            }}
                                                        >
                                                            {train.priority_score.toFixed(1)}
                                                        </span>
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        {(() => {
                                                            const d = (window.__decisions && window.__decisions[train.train_id]) || {};
                                                            if (d.last_action) {
                                                                return (
                                                                    <button
                                                                        onClick={async () => { await fetch('/api/supervisor/undo', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ trainId: train.train_id })}); fetchDashboardData(); }}
                                                                        className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-sm"
                                                                    >Undo</button>
                                                                );
                                                            }
                                                            return (
                                                                <div className="flex items-center gap-2">
                                                                    <button
                                                                        title="Approve"
                                                                        onClick={async () => { await fetch('/api/supervisor/approve', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ trainId: train.train_id })}); window.__decisions = window.__decisions||{}; window.__decisions[train.train_id] = { last_action:'approve' }; fetchDashboardData(); }}
                                                                        className="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 rounded text-sm"
                                                                    >✓</button>
                                                                    <button
                                                                        title="Reject"
                                                                        onClick={async () => { await fetch('/api/supervisor/reject', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ trainId: train.train_id })}); window.__decisions = window.__decisions||{}; window.__decisions[train.train_id] = { last_action:'reject' }; fetchDashboardData(); }}
                                                                        className="px-2 py-1 bg-rose-600 hover:bg-rose-500 rounded text-sm"
                                                                    >✗</button>
                                                                </div>
                                                            );
                                                        })()}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Branding SLA Tracker */}
                            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-lg font-semibold">Branding SLA Status</h3>
                                    <span className="text-slate-400 text-sm">Remaining hours only</span>
                                </div>
                                {branding && branding.length ? (
                                    <div className="overflow-hidden rounded-lg bg-slate-900/50">
                                        <table className="w-full">
                                            <thead className="bg-slate-700/50">
                                                <tr className="text-left">
                                                    <th className="px-4 py-3 text-sm font-medium text-slate-300">Train</th>
                                                    <th className="px-4 py-3 text-sm font-medium text-slate-300">Remaining Hours</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                            {branding.map((b, i) => (
                                                    <tr key={`${b.train_id}-${i}`} className="border-t border-slate-700/30">
                                                        <td className="px-4 py-3 font-medium">{b.train_id}</td>
                                                        <td className="px-4 py-3">{b.required_hours.toFixed(1)} h</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <div className="text-slate-400 text-sm">No branding SLA risks detected.</div>
                                )}
                            </div>

                            {/* Ineligible Trains */}
                            {violations.length > 0 && (
                                <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                    <h3 className="text-lg font-semibold mb-4">Ineligible Trains</h3>
                                    {violations.map((violation, index) => (
                                        <div key={index} className="flex items-center gap-3 text-red-400">
                                            <span className="text-red-500">✗</span>
                                            <span>{violation.train_id}: {violation.violations.join(', ')}</span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Fleet Mileage Balancing Chart */}
                            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-lg font-semibold">Fleet Mileage Balancing</h3>
                                    <span className="text-slate-400 text-sm">km per train (sorted)</span>
                                </div>
                                <div style={{ width: '100%', height: 320 }}>
                                    <ResponsiveContainer>
                                        <BarChart data={mileageData} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                            <XAxis dataKey="name" angle={-45} textAnchor="end" height={60} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                            <YAxis tick={{ fill: '#94a3b8' }} />
                                            <Tooltip 
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', color: '#e2e8f0' }}
                                                formatter={(value, key, payload) => {
                                                    if (key === 'mileage') {
                                                        return [Intl.NumberFormat('en-IN').format(value) + ' km', 'Mileage'];
                                                    }
                                                    return [value, key];
                                                }}
                                                labelFormatter={(label, payload) => {
                                                    if (payload && payload.length) {
                                                        const p = payload[0].payload;
                                                        return `Train ${label}  |  Bay ${p.bay}  |  Score ${p.score.toFixed(1)}`;
                                                    }
                                                    return `Train ${label}`;
                                                }}
                                            />
                                            <Bar dataKey="mileage" fill="#38bdf8" radius={[4,4,0,0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        };

        ReactDOM.render(<KMRLDashboard />, document.getElementById('root'));
    </script>
</body>
</html>
        """
    
    def run_server(self, host='127.0.0.1', port=8061, debug=False):
        # Reduce noisy 404 probing logs from werkzeug in dev
        try:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
        except Exception:
            pass
        print(f"🚀 Starting KMRL React Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)