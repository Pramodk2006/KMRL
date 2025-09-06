import pandas as pd
from datetime import datetime
from typing import Dict, List

class InductionDashboard:
    """Basic UI Framework for displaying KMRL IntelliFleet results"""
    
    def __init__(self, data_loader, constraint_engine, optimizer):
        self.data_loader = data_loader
        self.constraint_engine = constraint_engine
        self.optimizer = optimizer
        self.current_result = optimizer.optimized_result
        
    def display_header(self):
        print("=" * 80)
        print("🚄 KMRL IntelliFleet - AI-Driven Train Induction System")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Status: {self.current_result.get('status', 'Unknown')}")
        print(f"📊 Data Quality: {self.data_loader.data_quality_score:.1f}%")
        print("-" * 80)
    
    def display_fleet_overview(self):
        total_trains = len(self.data_loader.data_sources.get('trains', []))
        inducted = self.current_result.get('total_inducted', 0)
        standby = len(self.current_result.get('standby_trains', []))
        ineligible = len(self.constraint_engine.ineligible_trains)
        
        print("📊 FLEET OVERVIEW")
        print("-" * 40)
        print(f"Total Fleet Size:     {total_trains:2d} trains")
        print(f"✅ Inducted:          {inducted:2d} trains ({(inducted/total_trains*100):4.1f}%)")
        print(f"⏸️  Standby:           {standby:2d} trains ({(standby/total_trains*100):4.1f}%)")
        print(f"❌ Ineligible:        {ineligible:2d} trains ({(ineligible/total_trains*100):4.1f}%)")
        print()
    
    def display_inducted_trains(self):
        inducted = self.current_result.get('inducted_trains', [])
        
        print("✅ INDUCTED TRAINS - TONIGHT'S SERVICE")
        print("-" * 70)
        print(f"{'Rank':<4} {'Train':<6} {'Bay':<6} {'Score':<6} {'Branding':<9} {'Status':<15}")
        print("-" * 70)
        
        for i, train in enumerate(inducted[:10]):
            train_id = train['train_id']
            bay = train.get('assigned_bay', 'N/A')
            score = train.get('composite_score', 0)
            branding = train.get('branding_hours_left', 0)
            status = "Ready" if score > 80 else "Acceptable" if score > 60 else "Caution"
            
            print(f"{i+1:3d}. {train_id:<6} {bay:<6} {score:5.1f} {branding:8.1f}h {status:<15}")
        
        avg_score = self.current_result['optimization_improvements']['avg_composite_score']
        print(f"\n💡 Average Performance Score: {avg_score:.1f}/100")
        print()
    
    def display_conflict_alerts(self):
        conflicts = self.constraint_engine.conflicts
        ineligible = self.constraint_engine.ineligible_trains
        
        if not conflicts:
            print("✅ NO CONSTRAINT VIOLATIONS DETECTED")
            return
            
        print("⚠️  CONSTRAINT VIOLATIONS & ALERTS")
        print("-" * 50)
        
        train_issues = {}
        for train in ineligible:
            train_id = train['train_id']
            train_issues[train_id] = train['conflicts']
        
        for train_id, issues in list(train_issues.items())[:8]:
            print(f"❌ {train_id}:")
            for issue in issues:
                print(f"   └─ {issue}")
        
        if len(train_issues) > 8:
            print(f"   ... and {len(train_issues) - 8} more trains with issues")
        print()
    
    def display_capacity_analysis(self):
        bay_config = self.data_loader.data_sources.get('bay_config', pd.DataFrame())
        cleaning_slots = self.data_loader.data_sources.get('cleaning_slots', pd.DataFrame())
        
        if bay_config.empty:
            return
            
        print("🏗️ CAPACITY UTILIZATION")
        print("-" * 40)
        
        service_bays = bay_config[bay_config['bay_type'] == 'service']
        total_bay_capacity = service_bays['max_capacity'].sum()
        inducted_count = self.current_result.get('total_inducted', 0)
        
        bay_utilization = (inducted_count / total_bay_capacity * 100) if total_bay_capacity > 0 else 0
        print(f"Bay Utilization:      {inducted_count}/{total_bay_capacity} ({bay_utilization:.1f}%)")
        
        if not cleaning_slots.empty:
            cleaning_capacity = cleaning_slots['available_bays'].sum()
            cleaning_utilization = (inducted_count / cleaning_capacity * 100) if cleaning_capacity > 0 else 0
            print(f"Cleaning Utilization: {inducted_count}/{cleaning_capacity} ({cleaning_utilization:.1f}%)")
        
        improvements = self.current_result.get('optimization_improvements', {})
        distribution = improvements.get('score_distribution', {})
        
        print("\n📈 Performance Distribution:")
        print(f"   🟢 Excellent (80+): {distribution.get('excellent', 0)} trains")
        print(f"   🟡 Good (60-79):    {distribution.get('good', 0)} trains")
        print(f"   🟠 Acceptable (40-59): {distribution.get('acceptable', 0)} trains")
        print(f"   🔴 Poor (<40):      {distribution.get('poor', 0)} trains")
        print()
    
    def display_key_metrics(self):
        improvements = self.current_result.get('optimization_improvements', {})
        
        print("📊 KEY PERFORMANCE METRICS")
        print("-" * 40)
        print(f"System Performance Score: {improvements.get('avg_composite_score', 0):.1f}/100")
        print(f"Service Readiness:        {improvements.get('avg_service_readiness', 0):.1f}/100")
        print(f"Maintenance Risk:         {improvements.get('avg_maintenance_penalty', 0):.1f}/100")
        print(f"Branding Compliance:      {improvements.get('avg_branding_priority', 0):.1f}/100")
        
        inducted_count = self.current_result.get('total_inducted', 0)
        estimated_savings = inducted_count * 23000
        
        print(f"\n💰 ESTIMATED IMPACT")
        print(f"Tonight's Cost Savings:   ₹{estimated_savings:,}")
        print(f"Annual Projected Savings: ₹{estimated_savings * 365:,}")
        print()
    
    def generate_summary_report(self) -> str:
        inducted = self.current_result.get('total_inducted', 0)
        conflicts = len(self.constraint_engine.conflicts)
        avg_score = self.current_result.get('optimization_improvements', {}).get('avg_composite_score', 0)
        
        status = '✅ OPTIMAL' if avg_score > 80 else '⚠️ ACCEPTABLE' if avg_score > 60 else '❌ REVIEW NEEDED'
        
        summary = f"""
📋 EXECUTIVE SUMMARY
==================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Totals: {inducted} trains inducted, {conflicts} conflicts resolved
Overall Performance: {avg_score:.1f}/100
Status: {status}

Next Steps: 
- Monitor inducted trains for service readiness
- Address constraint violations for ineligible trains
- Review recommendations for potential optimizations
"""
        return summary
    
    def display_complete_dashboard(self):
        self.display_header()
        self.display_fleet_overview()
        self.display_inducted_trains()
        self.display_conflict_alerts()
        self.display_capacity_analysis()
        self.display_key_metrics()
        
        summary = self.generate_summary_report()
        print(summary)
        
        return summary
