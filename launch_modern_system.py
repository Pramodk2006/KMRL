#!/usr/bin/env python3
"""
🚀 Modern KMRL IntelliFleet Launcher
Quick launcher for the enhanced live simulation with modern UI
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print the modern launcher banner"""
    print("=" * 80)
    print("🚄" + " " * 25 + "KMRL INTELLIFLEET" + " " * 26 + "🚄")
    print("🎨" + " " * 23 + "MODERN UI LAUNCHER" + " " * 25 + "🎨")
    print("=" * 80)
    print("✨ Enhanced Live Simulation with Professional Design")
    print("📱 Responsive Layout • Material Design 3.0 • Real-time Updates")
    print("🎬 3-Column Layout: Controls • Interactive Map • Activity Feed")
    print("📊 Analytics Dashboard with AI Insights & Performance Metrics")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    required_packages = [
        'dash', 'plotly', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def launch_system():
    """Launch the modern KMRL IntelliFleet system"""
    print(f"\n🚀 Launching Modern KMRL IntelliFleet System...")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Run the modern enterprise main
        subprocess.run([sys.executable, 'modern_enterprise_main.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching system: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all files are in the correct directory")
        print("   2. Check that src/ directory exists with required modules")
        print("   3. Verify Python version (3.8+ recommended)")
        print("   4. Install missing dependencies if any")
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def show_features():
    """Display system features"""
    print("\n🎯 KEY FEATURES:")
    print("   🎬 Enhanced Live Simulation:")
    print("      • 3-column responsive layout (Controls | Map | Activity)")
    print("      • Interactive KMRL route map with GPS coordinates")
    print("      • Real-time train tracking with status indicators")
    print("      • Enhanced animation controls and speed adjustment")
    print("      • Live activity feed with timestamp events")
    print("      • System health monitoring panel")
    print("")
    print("   📊 Analytics Dashboard:")
    print("      • AI-powered train induction optimization")
    print("      • Performance metrics and cost savings analysis")
    print("      • Constraint violation monitoring")
    print("      • Interactive charts and visualizations")
    print("      • Bay layout and occupancy tracking")
    print("")
    print("   🎨 Modern Design System:")
    print("      • Material Design 3.0 principles")
    print("      • Professional color scheme and typography")
    print("      • Smooth animations and micro-interactions")
    print("      • Fully responsive (mobile, tablet, desktop)")
    print("      • Enhanced accessibility and user experience")
    print("")
    print("   ⚡ Technical Features:")
    print("      • 1.5s real-time update intervals")
    print("      • WebSocket integration for IoT data")
    print("      • RESTful API gateway")
    print("      • Digital twin simulation engine")
    print("      • AI optimization with constraint solving")

def main():
    """Main launcher function"""
    print_banner()
    
    print("\n🏁 LAUNCH OPTIONS:")
    print("   1. 🚀 Launch Modern System (Recommended)")
    print("   2. 📋 Show System Features")
    print("   3. 🔍 Check Dependencies")
    print("   4. ❌ Exit")
    
    while True:
        try:
            choice = input("\n🎯 Enter your choice (1-4): ").strip()
            
            if choice == '1':
                if check_dependencies():
                    launch_system()
                break
            elif choice == '2':
                show_features()
                input("\n⏎ Press Enter to continue...")
                continue
            elif choice == '3':
                check_dependencies()
                input("\n⏎ Press Enter to continue...")
                continue
            elif choice == '4':
                print("👋 Goodbye! Thank you for using KMRL IntelliFleet")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using KMRL IntelliFleet")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()