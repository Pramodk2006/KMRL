
# --- Dynamic Dashboard Integration ---
import sys
sys.path.append('..')
from src.data_loader import DataLoader
from src.constraint_engine import CustomConstraintEngine
from src.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dashboard import InductionDashboard
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

# Run backend pipeline
loader = DataLoader()
data = loader.get_integrated_data()
constraint_engine = CustomConstraintEngine(data)
constraint_result = constraint_engine.run_constraint_optimization()
optimizer = MultiObjectiveOptimizer(constraint_result, data)
optimized_result = optimizer.optimize_induction_ranking()
optimizer.optimized_result = optimized_result
dashboard = InductionDashboard(loader, constraint_engine, optimizer)

# Extract dashboard data
current_result = optimizer.optimized_result
improvements = current_result.get('optimization_improvements', {})
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
total_trains = len(loader.data_sources.get('trains', []))
inducted = current_result.get('total_inducted', 0)
standby = len(current_result.get('standby_trains', []))
ineligible = len(constraint_engine.ineligible_trains)
inducted_pct = (inducted/total_trains*100) if total_trains else 0
ineligible_pct = (ineligible/total_trains*100) if total_trains else 0

inducted_trains = []
for i, train in enumerate(current_result.get('inducted_trains', []), 1):
	inducted_trains.append({
		'rank': i,
		'train_id': train['train_id'],
		'bay': train.get('assigned_bay', 'N/A'),
		'score': train.get('composite_score', 0),
		'branding': train.get('branding_hours_left', 0),
		'status': "Ready" if train.get('composite_score', 0) > 80 else "Caution" if train.get('composite_score', 0) > 60 else "Maintenance"
	})

violations = []
for train in constraint_engine.ineligible_trains:
	violations.append({
		'train_id': train['train_id'],
		'issues': train['conflicts']
	})

service_bays = loader.data_sources.get('bay_config', None)
if service_bays is not None and not service_bays.empty:
	service_bays = service_bays[service_bays['bay_type'] == 'service']
	total_bay_capacity = service_bays['max_capacity'].sum()
else:
	total_bay_capacity = inducted

cleaning_slots = loader.data_sources.get('cleaning_slots', None)
if cleaning_slots is not None and not cleaning_slots.empty:
	cleaning_capacity = cleaning_slots['available_bays'].sum()
else:
	cleaning_capacity = inducted

bay_utilization = f"{inducted}/{total_bay_capacity} ({(inducted/total_bay_capacity*100) if total_bay_capacity else 0:.1f}%)"
cleaning_utilization = f"{inducted}/{cleaning_capacity} ({(inducted/cleaning_capacity*100) if cleaning_capacity else 0:.1f}%)"

distribution = improvements.get('score_distribution', {})
performance_dist = {
	'excellent': distribution.get('excellent', 0),
	'good': distribution.get('good', 0),
	'acceptable': distribution.get('acceptable', 0),
	'poor': distribution.get('poor', 0),
}

performance_metrics = {
	'avg_score': improvements.get('avg_composite_score', 0),
	'service_readiness': improvements.get('avg_service_readiness', 0),
	'maintenance_risk': improvements.get('avg_maintenance_penalty', 0),
	'branding_compliance': improvements.get('avg_branding_priority', 0),
	'cost_savings': inducted * 23000,
	'annual_savings': inducted * 23000 * 365,
}

summary = {
	'date': now,
	'inducted': inducted,
	'conflicts': len(constraint_engine.conflicts),
	'overall': improvements.get('avg_composite_score', 0),
	'status': '✅ OPTIMAL' if improvements.get('avg_composite_score', 0) > 80 else '⚠️ ACCEPTABLE' if improvements.get('avg_composite_score', 0) > 60 else '❌ REVIEW NEEDED',
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
	html.Hr(),
	html.H2("🚄 KMRL IntelliFleet - AI-Driven Train Induction System", style={"textAlign": "center", "color": "#0074D9"}),
	html.Hr(),
	dbc.Row([
		dbc.Col([
			html.H5("📅 Date: " + now),
			html.H5("🎯 Status: Multi-objective optimization complete"),
			html.H5(f"📊 Data Quality: {loader.data_quality_score:.1f}%"),
		], width=4),
		dbc.Col([
			dbc.Card([
				dbc.CardHeader("Fleet Overview", style={"backgroundColor": "#F0F8FF"}),
				dbc.CardBody([
					html.Div([
						html.Span("Total Fleet Size: ", style={"fontWeight": "bold"}),
						html.Span(total_trains),
					]),
					html.Div([
						html.Span("✅ Inducted: ", style={"color": "green", "fontWeight": "bold"}),
						html.Span(f"{inducted} trains ({inducted_pct:.1f}%)"),
					]),
					html.Div([
						html.Span("⏸️  Standby: ", style={"color": "#FFA500", "fontWeight": "bold"}),
						html.Span(f"{standby} trains ({(standby/total_trains*100) if total_trains else 0:.1f}%)"),
					]),
					html.Div([
						html.Span("❌ Ineligible: ", style={"color": "red", "fontWeight": "bold"}),
						html.Span(f"{ineligible} trains ({ineligible_pct:.1f}%)"),
					]),
				])
			], color="primary", outline=True)
		], width=8)
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("✅ Inducted Trains - Tonight's Service", style={"backgroundColor": "#E6FFED"}),
		dbc.CardBody([
			dbc.Table([
				html.Thead(html.Tr([
					html.Th("Rank"), html.Th("Train"), html.Th("Bay"), html.Th("Score"), html.Th("Branding"), html.Th("Status")
				])),
				html.Tbody([
					html.Tr([
						html.Td(train['rank']),
						html.Td(train['train_id']),
						html.Td(train['bay']),
						html.Td(f"{train['score']:.1f}"),
						html.Td(f"{train['branding']:.1f}h"),
						html.Td(train['status'], style={"color": "green" if train['status']=='Ready' else "#FFA500"})
					]) for train in inducted_trains
				])
			], bordered=True, hover=True, responsive=True, striped=True),
			html.Br(),
			html.H6(f"💡 Average Performance Score: {performance_metrics['avg_score']}/100", style={"color": "#0074D9"})
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("⚠️ Constraint Violations & Alerts", style={"backgroundColor": "#FFF3CD"}),
		dbc.CardBody([
			html.Ul([
				html.Li([
					html.Span(f"❌ {v['train_id']}:", style={"color": "red", "fontWeight": "bold"}),
					html.Ul([html.Li(issue) for issue in v['issues']])
				]) for v in violations
			])
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("🏗️ Capacity Utilization", style={"backgroundColor": "#E3F2FD"}),
		dbc.CardBody([
			html.Div([
				html.Span("Bay Utilization: ", style={"fontWeight": "bold"}),
				html.Span(bay_utilization)
			]),
			html.Div([
				html.Span("Cleaning Utilization: ", style={"fontWeight": "bold"}),
				html.Span(cleaning_utilization)
			])
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("📈 Performance Distribution", style={"backgroundColor": "#F8F9FA"}),
		dbc.CardBody([
			html.Div([
				html.Span("🟢 Excellent (80+): ", style={"color": "green", "fontWeight": "bold"}),
				html.Span(performance_dist['excellent'])
			]),
			html.Div([
				html.Span("🟡 Good (60-79): ", style={"color": "#FFD700", "fontWeight": "bold"}),
				html.Span(performance_dist['good'])
			]),
			html.Div([
				html.Span("🟠 Acceptable (40-59): ", style={"color": "#FFA500", "fontWeight": "bold"}),
				html.Span(performance_dist['acceptable'])
			]),
			html.Div([
				html.Span("🔴 Poor (<40): ", style={"color": "red", "fontWeight": "bold"}),
				html.Span(performance_dist['poor'])
			])
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("📊 Key Performance Metrics", style={"backgroundColor": "#E6E6FA"}),
		dbc.CardBody([
			html.Div([
				html.Span("System Performance Score: ", style={"fontWeight": "bold"}),
				html.Span(f"{performance_metrics['avg_score']}/100")
			]),
			html.Div([
				html.Span("Service Readiness: ", style={"fontWeight": "bold"}),
				html.Span(f"{performance_metrics['service_readiness']}/100")
			]),
			html.Div([
				html.Span("Maintenance Risk: ", style={"fontWeight": "bold"}),
				html.Span(f"{performance_metrics['maintenance_risk']}/100")
			]),
			html.Div([
				html.Span("Branding Compliance: ", style={"fontWeight": "bold"}),
				html.Span(f"{performance_metrics['branding_compliance']}/100")
			])
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("💰 Estimated Impact", style={"backgroundColor": "#E8F5E9"}),
		dbc.CardBody([
			html.Div([
				html.Span("Tonight's Cost Savings: ", style={"fontWeight": "bold"}),
				html.Span(f"₹{performance_metrics['cost_savings']:,}")
			]),
			html.Div([
				html.Span("Annual Projected Savings: ", style={"fontWeight": "bold"}),
				html.Span(f"₹{performance_metrics['annual_savings']:,}")
			])
		])
	]),
	html.Hr(),
	dbc.Card([
		dbc.CardHeader("📋 Executive Summary", style={"backgroundColor": "#F3E5F5"}),
		dbc.CardBody([
			html.Div([
				html.Span("Date: ", style={"fontWeight": "bold"}),
				html.Span(summary['date'])
			]),
			html.Div([
				html.Span("Totals: ", style={"fontWeight": "bold"}),
				html.Span(f"{summary['inducted']} trains inducted, {summary['conflicts']} conflicts resolved")
			]),
			html.Div([
				html.Span("Overall Performance: ", style={"fontWeight": "bold"}),
				html.Span(f"{summary['overall']}/100")
			]),
			html.Div([
				html.Span("Status: ", style={"fontWeight": "bold"}),
				html.Span(summary['status'])
			]),
			html.Br(),
			html.Div([
				html.Span("Next Steps:", style={"fontWeight": "bold"}),
				html.Ul([
					html.Li("Monitor inducted trains for service readiness"),
					html.Li("Address constraint violations for ineligible trains"),
					html.Li("Review recommendations for potential optimizations")
				])
			])
		])
	]),
	html.Hr(),
	html.Div([
		html.Span(f"💾 Report saved to: induction_report_{now.replace('-', '').replace(':', '').replace(' ', '_')}.txt", style={"fontWeight": "bold", "color": "#0074D9"})
	], style={"textAlign": "center"}),
	html.Hr(),
], fluid=True)

if __name__ == "__main__":
	app.run(debug=True)
# ...existing code...
