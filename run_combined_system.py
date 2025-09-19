#!/usr/bin/env python3
"""
Simple launcher for KMRL IntelliFleet Combined System
Run this file to start the complete system with both Animated and Classical UI
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the combined KMRL IntelliFleet system"""
    
    print("🚀 KMRL IntelliFleet Combined System Launcher")
    print("=" * 60)
    print("🎯 Starting system with:")
    print("  🎬 Animated Simulation Tab - Live train tracking")
    print("  📊 Analytics Dashboard Tab - AI insights & metrics")
    print("  🔄 Real-time updates every 1.5 seconds")
    print("  🗺️ Interactive KMRL route map with GPS coordinates")
    print("=" * 60)
    
    try:
        # Import and start the combined system
        from main.combined_enterprise_main import KMRLCombinedIntelliFleetSystem
        
        print("✅ Initializing AI optimization components...")
        print("✅ Setting up digital twin engine...")
        print("✅ Loading combined dashboard with navigation tabs...")
        
        # Create and start the system
        system = KMRLCombinedIntelliFleetSystem()
        
        print("\n🌐 Dashboard will be available at: http://127.0.0.1:8050")
        print("🔧 API Gateway will be available at: http://127.0.0.1:8002")
        print("\n🎮 Controls:")
        print("  - Use navigation tabs to switch between Animated and Analytics views")
        print("  - Animated tab: Play/Pause controls, speed adjustment")
        print("  - Analytics tab: Real-time AI insights and performance metrics")
        print("\n🛑 Press Ctrl+C to stop the system")
        print("=" * 60)
        
        # Start the system (this will block until shutdown)
        system.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested by user")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all required files are in the correct location:")
        print("   - combined_enterprise_main.py")
        print("   - combined_dashboard.py") 
        print("   - animated_web_dashboard.py")
        print("   - enhanced_web_dashboard.py")
        print("   - src/ folder with all system components")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        print("💡 Check that all dependencies are installed and src/ folder exists")
        sys.exit(1)

if __name__ == "__main__":
    main()