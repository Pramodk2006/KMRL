import pandas as pd
from datetime import datetime
from typing import Dict, List
from .dashboard import InductionDashboard

class AIEnhancedDashboard(InductionDashboard):
    """Enhanced dashboard with AI insights"""
    
    def display_ai_insights(self):
        """Display AI-powered insights"""
        ai_insights = self.current_result.get('ai_insights', {})
        risk_insights = self.current_result.get('risk_insights', {})
        
        print("🤖 AI INTELLIGENCE & PREDICTIONS")
        print("-" * 50)
        
        if not ai_insights.get('models_ready', False):
            print("⚠️ AI models not fully initialized")
            print("   Run with historical data to enable full AI capabilities")
            return
        
        # Risk distribution
        risk_dist = risk_insights.get('risk_distribution', {})
        total_trains = sum(risk_dist.values()) if risk_dist else 0
        
        if total_trains > 0:
            print("📊 Fleet Risk Assessment:")
            print(f"   🟢 Low Risk:     {risk_dist.get('low', 0):2d} trains ({risk_dist.get('low', 0)/total_trains*100:4.1f}%)")
            print(f"   🟡 Medium Risk:  {risk_dist.get('medium', 0):2d} trains ({risk_dist.get('medium', 0)/total_trains*100:4.1f}%)")
            print(f"   🟠 High Risk:    {risk_dist.get('high', 0):2d} trains ({risk_dist.get('high', 0)/total_trains*100:4.1f}%)")
            print(f"   🔴 Critical:     {risk_dist.get('critical', 0):2d} trains ({risk_dist.get('critical', 0)/total_trains*100:4.1f}%)")
        
        # Average risk
        avg_risk = risk_insights.get('average_risk', 0)
        print(f"\n🎯 Average Fleet Risk: {avg_risk:.1%}")
        
        # High-risk trains alert
        high_risk_trains = risk_insights.get('high_risk_trains', [])
        if high_risk_trains:
            print(f"\n⚠️  HIGH-RISK TRAINS ALERT:")
            for train_id, risk_prob in high_risk_trains[:5]:
                status = "🔴 CRITICAL" if risk_prob > 0.8 else "🟠 HIGH"
                print(f"   {status}: {train_id} - {risk_prob:.1%} failure probability")
        
        print()
    
    def display_seasonal_intelligence(self):
        """Display seasonal intelligence and patterns"""
        ai_insights = self.current_result.get('ai_insights', {})
        seasonal_rec = ai_insights.get('seasonal_recommendations', {})
        
        if not seasonal_rec:
            return
        
        print("🌍 SEASONAL INTELLIGENCE")
        print("-" * 40)
        
        current_month = seasonal_rec.get('month', datetime.now().month)
        month_name = datetime.now().strftime('%B')
        
        print(f"📅 Current Season: {month_name}")
        
        # Expected metrics
        failure_rate = seasonal_rec.get('expected_failure_rate', 0.1)
        energy_consumption = seasonal_rec.get('expected_energy_consumption', 180)
        
        print(f"📊 Expected Metrics:")
        print(f"   - Failure Rate: {failure_rate:.1%}")
        print(f"   - Energy Usage: {energy_consumption:.0f} kWh/train")
        
        # Recommendations
        recommendations = seasonal_rec.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Seasonal Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Weight adjustments
        weight_adjustments = seasonal_rec.get('weight_adjustments', {})
        if weight_adjustments:
            print(f"\n⚖️  Active Weight Adjustments:")
            for weight, adjustment in weight_adjustments.items():
                symbol = "↗️" if adjustment > 0 else "↘️"
                print(f"   {symbol} {weight}: {adjustment:+.3f}")
        
        print()
    
    def display_ai_recommendations(self):
        """Display AI-powered recommendations"""
        recommendations = self.current_result.get('recommendations', [])
        
        if not recommendations:
            print("✅ NO AI RECOMMENDATIONS")
            print("   All systems operating optimally.\n")
            return
        
        print("🧠 AI RECOMMENDATIONS")
        print("-" * 40)
        
        # Group by priority
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
        low_priority = [r for r in recommendations if r.get('priority') == 'low']
        
        for priority_level, recs, icon in [
            ('HIGH PRIORITY', high_priority, '🔴'),
            ('MEDIUM PRIORITY', medium_priority, '🟡'),
            ('LOW PRIORITY', low_priority, '🟢')
        ]:
            if recs:
                print(f"{icon} {priority_level}:")
                for rec in recs:
                    rec_type = rec.get('type', 'general').upper()
                    message = rec.get('message', 'No message')
                    print(f"   [{rec_type}] {message}")
                print()
    
    def display_predictive_performance(self):
        """Display predictive model performance metrics"""
        improvements = self.current_result.get('optimization_improvements', {})
        
        if not improvements.get('ai_enhancement_active', False):
            return
        
        print("📈 PREDICTIVE PERFORMANCE")
        print("-" * 40)
        
        # AI-specific metrics
        avg_failure_risk = improvements.get('avg_failure_risk', 0.1)
        avg_predictive_score = improvements.get('avg_predictive_score', 50)
        
        print(f"🎯 Predictive Metrics:")
        print(f"   - Average Failure Risk: {avg_failure_risk:.1%}")
        print(f"   - Predictive Score: {avg_predictive_score:.1f}/100")
        
        # Risk distribution for inducted trains
        risk_dist = improvements.get('risk_distribution', {})
        if risk_dist:
            print(f"\n📊 Inducted Trains Risk Profile:")
            total = sum(risk_dist.values())
            if total > 0:
                for risk_level, count in risk_dist.items():
                    percentage = count / total * 100
                    print(f"   - {risk_level.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print()
    
    def display_enhanced_dashboard(self):
        """Display the complete enhanced dashboard"""
        # Standard dashboard sections
        self.display_header()
        self.display_fleet_overview()
        self.display_inducted_trains()
        
        # AI-enhanced sections
        self.display_ai_insights()
        self.display_seasonal_intelligence() 
        self.display_ai_recommendations()
        self.display_predictive_performance()
        
        # Standard sections
        self.display_conflict_alerts()
        self.display_capacity_analysis()
        self.display_key_metrics()
        
        # Generate enhanced summary
        summary = self.generate_enhanced_summary()
        print(summary)
        
        return summary
    
    def generate_enhanced_summary(self) -> str:
        """Generate enhanced executive summary with AI insights"""
        inducted = self.current_result.get('total_inducted', 0)
        conflicts = len(self.constraint_engine.conflicts)
        avg_score = self.current_result.get('optimization_improvements', {}).get('avg_composite_score', 0)
        
        # AI metrics
        improvements = self.current_result.get('optimization_improvements', {})
        avg_risk = improvements.get('avg_failure_risk', 0.1)
        ai_active = improvements.get('ai_enhancement_active', False)
        
        status = '✅ OPTIMAL' if avg_score > 80 else '⚠️ ACCEPTABLE' if avg_score > 60 else '❌ REVIEW NEEDED'
        
        ai_status = ""
        if ai_active:
            risk_level = "LOW" if avg_risk < 0.3 else "MEDIUM" if avg_risk < 0.6 else "HIGH"
            ai_status = f"\nAI Enhancement: ACTIVE | Fleet Risk: {risk_level} ({avg_risk:.1%})"
        
        summary = f"""
📋 EXECUTIVE SUMMARY - AI ENHANCED
=================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Totals: {inducted} trains inducted, {conflicts} conflicts resolved
Performance: {avg_score:.1f}/100 | Status: {status}{ai_status}

🤖 AI INSIGHTS:
- Predictive failure modeling: {'ACTIVE' if ai_active else 'OFFLINE'}
- Seasonal pattern recognition: {'ACTIVE' if ai_active else 'OFFLINE'}
- Reinforcement learning: {'TRAINING' if ai_active else 'OFFLINE'}

💡 NEXT STEPS: 
- Monitor high-risk trains for early intervention
- Apply seasonal weight adjustments as recommended
- Review AI recommendations for optimization opportunities
- Continue historical data collection for improved predictions
"""
        return summary
