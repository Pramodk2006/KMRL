#!/usr/bin/env python3
"""
IBM Maximo Setup Script for KMRL IntelliFleet
Automates the setup process for IBM Maximo integration
"""

import os
import sys
import requests
import json
from datetime import datetime

def setup_environment_variables():
    """Set up environment variables for IBM Maximo"""
    print("🔧 Setting up IBM Maximo environment variables...")
    
    # Get user input
    base_url = input("Enter IBM Maximo Base URL (e.g., https://your-instance.maas.ibm.com): ").strip()
    tenant_id = input("Enter Tenant ID: ").strip()
    client_id = input("Enter Client ID: ").strip()
    client_secret = input("Enter Client Secret: ").strip()
    
    # Create .env file
    env_content = f"""# IBM Maximo Configuration
KMRL_MAXIMO_BASE_URL={base_url}
KMRL_MAXIMO_TENANT_ID={tenant_id}
KMRL_MAXIMO_CLIENT_ID={client_id}
KMRL_MAXIMO_CLIENT_SECRET={client_secret}

# Optional: Username/Password (if using basic auth)
# KMRL_MAXIMO_USERNAME=your-username
# KMRL_MAXIMO_PASSWORD=your-password
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment variables saved to .env file")
    
    # Set environment variables for current session
    os.environ['KMRL_MAXIMO_BASE_URL'] = base_url
    os.environ['KMRL_MAXIMO_TENANT_ID'] = tenant_id
    os.environ['KMRL_MAXIMO_CLIENT_ID'] = client_id
    os.environ['KMRL_MAXIMO_CLIENT_SECRET'] = client_secret
    
    return {
        'base_url': base_url,
        'tenant_id': tenant_id,
        'client_id': client_id,
        'client_secret': client_secret
    }

def test_connection(config):
    """Test connection to IBM Maximo"""
    print("🔍 Testing IBM Maximo connection...")
    
    try:
        # Test OAuth authentication
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'scope': 'maximo.api'
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        oauth_url = f"{config['base_url']}/maximo/api/oslc/token"
        response = requests.post(oauth_url, data=auth_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            print("✅ Authentication successful!")
            
            # Test API access
            api_headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            workorder_url = f"{config['base_url']}/maximo/api/oslc/os/mxapiwodetail"
            api_response = requests.get(workorder_url, headers=api_headers, timeout=30)
            
            if api_response.status_code == 200:
                print("✅ API access successful!")
                return True
            else:
                print(f"❌ API access failed: {api_response.status_code}")
                return False
        else:
            print(f"❌ Authentication failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def create_sample_data(config):
    """Create sample data in IBM Maximo"""
    print("📊 Creating sample data in IBM Maximo...")
    
    try:
        # Get access token
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'scope': 'maximo.api'
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        oauth_url = f"{config['base_url']}/maximo/api/oslc/token"
        response = requests.post(oauth_url, data=auth_data, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print("❌ Failed to get access token")
            return False
        
        token_data = response.json()
        access_token = token_data.get('access_token')
        
        # API headers
        api_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'maxversion': '1.0'
        }
        
        # Create sample assets (trains)
        asset_url = f"{config['base_url']}/maximo/api/oslc/os/mxapiasset"
        
        sample_assets = [
            {
                'assetnum': 'T001',
                'description': 'Kochi Metro Train Set 001',
                'assettype': 'TRAIN',
                'status': 'ACTIVE',
                'location': 'DEPOT_A'
            },
            {
                'assetnum': 'T002',
                'description': 'Kochi Metro Train Set 002',
                'assettype': 'TRAIN',
                'status': 'ACTIVE',
                'location': 'DEPOT_A'
            },
            {
                'assetnum': 'T003',
                'description': 'Kochi Metro Train Set 003',
                'assettype': 'TRAIN',
                'status': 'ACTIVE',
                'location': 'DEPOT_A'
            }
        ]
        
        for asset in sample_assets:
            payload = {'member': [asset]}
            response = requests.post(asset_url, headers=api_headers, json=payload, timeout=30)
            
            if response.status_code in [200, 201]:
                print(f"✅ Created asset: {asset['assetnum']}")
            else:
                print(f"⚠️ Asset {asset['assetnum']} may already exist or failed: {response.status_code}")
        
        # Create sample work orders
        workorder_url = f"{config['base_url']}/maximo/api/oslc/os/mxapiwodetail"
        
        sample_workorders = [
            {
                'wonum': 'JC001',
                'description': 'Regular maintenance check for T001',
                'assetnum': 'T001',
                'worktype': 'PM',
                'priority': '3',
                'status': 'WAPPR'
            },
            {
                'wonum': 'JC002',
                'description': 'Brake system inspection for T002',
                'assetnum': 'T002',
                'worktype': 'CM',
                'priority': '2',
                'status': 'INPRG'
            },
            {
                'wonum': 'JC003',
                'description': 'HVAC system repair for T003',
                'assetnum': 'T003',
                'worktype': 'CM',
                'priority': '2',
                'status': 'WAPPR'
            }
        ]
        
        for wo in sample_workorders:
            payload = {'member': [wo]}
            response = requests.post(workorder_url, headers=api_headers, json=payload, timeout=30)
            
            if response.status_code in [200, 201]:
                print(f"✅ Created work order: {wo['wonum']}")
            else:
                print(f"⚠️ Work order {wo['wonum']} may already exist or failed: {response.status_code}")
        
        print("✅ Sample data creation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def test_integration():
    """Test the KMRL IntelliFleet integration"""
    print("🧪 Testing KMRL IntelliFleet integration...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from production_maximo_integration import get_production_maximo
        
        maximo = get_production_maximo()
        
        # Test connection status
        status = maximo.get_connection_status()
        print(f"📊 Connection Status: {status}")
        
        # Test fetching work orders
        work_orders = maximo.fetch_work_orders()
        print(f"📋 Work Orders: {len(work_orders)} found")
        
        # Test fetching assets
        assets = maximo.fetch_assets()
        print(f"🚂 Assets: {len(assets)} found")
        
        print("✅ Integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 IBM Maximo Setup for KMRL IntelliFleet")
    print("=" * 50)
    
    # Step 1: Set up environment variables
    config = setup_environment_variables()
    
    # Step 2: Test connection
    if not test_connection(config):
        print("❌ Setup failed: Could not connect to IBM Maximo")
        return False
    
    # Step 3: Create sample data
    create_sample = input("Create sample data in Maximo? (y/n): ").strip().lower()
    if create_sample == 'y':
        create_sample_data(config)
    
    # Step 4: Test integration
    if not test_integration():
        print("❌ Setup failed: Integration test failed")
        return False
    
    print("\n🎉 IBM Maximo setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the KMRL IntelliFleet system: python run_system_orchestrator.py")
    print("2. Access the web dashboard: http://127.0.0.1:8050")
    print("3. Access the data management dashboard: http://127.0.0.1:8051")
    print("4. Check Maximo integration status: http://127.0.0.1:8000/maximo/status")
    
    return True

if __name__ == "__main__":
    main()

