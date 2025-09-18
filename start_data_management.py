#!/usr/bin/env python3
"""
Start Data Management System
Starts both the API server and Data Management Dashboard
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime

def start_api_server():
    """Start the API server in a separate thread"""
    print("🚀 Starting API Server...")
    try:
        from src.api_gateway import APIGateway
        from src.digital_twin_engine import DigitalTwinEngine
        from src.enhanced_optimizer import EnhancedMultiObjectiveOptimizer
        
        # Create minimal required components
        digital_twin = DigitalTwinEngine()
        ai_optimizer = EnhancedMultiObjectiveOptimizer()
        
        api = APIGateway(digital_twin, ai_optimizer)
        api.run_server()
    except Exception as e:
        print(f"❌ API Server failed: {e}")
        print("ℹ️ Dashboard will work in offline mode")

def start_data_dashboard():
    """Start the Data Management Dashboard in a separate thread"""
    print("🚀 Starting Data Management Dashboard...")
    try:
        from src.data_management_dashboard import DataManagementDashboard
        dashboard = DataManagementDashboard()
        dashboard.run()
    except Exception as e:
        print(f"❌ Data Dashboard failed: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 KMRL IntelliFleet Data Management System")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    print("⏳ Waiting for API server to start...")
    time.sleep(3)
    
    # Start Data Management Dashboard (this will block)
    start_data_dashboard()

if __name__ == "__main__":
    main()
