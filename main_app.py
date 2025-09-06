#!/usr/bin/env python3
"""
KMRL IntelliFleet - AI-Driven Train Induction System
Phase 2 Complete Implementation
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append('src')

from src.data_loader import DataLoader
from src.constraint_engine import CustomConstraintEngine
from src.multi_objective_optimizer import MultiObjectiveOptimizer
from src.dashboard import InductionDashboard

def main():
    """Main application entry point"""
    print("🚀 Starting KMRL IntelliFleet System...")
    
    # Phase 1: Data Integration
    print("\n=== Phase 1: Data Integration ===")
    loader = DataLoader()
    data = loader.get_integrated_data()
    
    if loader.data_quality_score < 50:
        print("❌ Data quality too low. Please check data sources.")
        return
    
    # Phase 2: Constraint Processing
    print("\n=== Phase 2: Constraint Processing ===")
    constraint_engine = CustomConstraintEngine(data)
    constraint_result = constraint_engine.run_constraint_optimization()
    
    # Phase 3: Multi-Objective Optimization
    print("\n=== Phase 3: Multi-Objective Optimization ===")
    optimizer = MultiObjectiveOptimizer(constraint_result, data)
    optimized_result = optimizer.optimize_induction_ranking()
    optimizer.optimized_result = optimized_result
    
    # Phase 4: Display Dashboard
    print("\n=== Phase 4: Dashboard Display ===")
    dashboard = InductionDashboard(loader, constraint_engine, optimizer)
    summary = dashboard.display_complete_dashboard()
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'induction_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    
    print(f"\n💾 Report saved to: induction_report_{timestamp}.txt")
    
    return {
        'data_loader': loader,
        'constraint_engine': constraint_engine,
        'optimizer': optimizer,
        'dashboard': dashboard
    }

if __name__ == "__main__":
    components = main()
    print("\n✅ KMRL IntelliFleet System Ready!")
