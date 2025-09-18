# IBM Maximo Setup Guide for KMRL IntelliFleet

## Overview

This guide will help you set up a real IBM Maximo account and integrate it with the KMRL IntelliFleet system for production-grade maintenance data management.

## 🏢 **Step 1: Create IBM Maximo Account**

### **Option A: IBM Maximo Application Suite (Recommended)**

#### **1.1 Sign Up for IBM Maximo**
1. **Visit**: https://www.ibm.com/products/maximo
2. **Click**: "Try Maximo Application Suite"
3. **Choose**: "Maximo Application Suite on IBM Cloud"
4. **Select Plan**: 
   - **Trial**: Free 30-day trial (recommended for testing)
   - **Standard**: Production-ready with full features

#### **1.2 IBM Cloud Setup**
1. **Go to**: https://cloud.ibm.com/catalog/services/maximo-application-suite
2. **Create Service Instance**:
   - Service name: `kmrl-maximo-instance`
   - Resource group: Default
   - Region: Choose closest to your location
3. **Configure Service**:
   - Plan: Standard or Trial
   - Enable all applications (Maximo, Monitor, Predict, etc.)

#### **1.3 Get Access Credentials**
After service creation, you'll get:
- **Base URL**: `https://your-instance.maas.ibm.com`
- **Tenant ID**: Your tenant identifier
- **Admin credentials**: Username and password

### **Option B: IBM Maximo SaaS (Alternative)**

#### **1.1 Direct SaaS Signup**
1. **Visit**: https://www.ibm.com/products/maximo/application-suite
2. **Request Demo/Trial**
3. **Work with IBM Sales** for enterprise setup
4. **Get production credentials**

## 🔧 **Step 2: Configure Maximo for Train Maintenance**

### **2.1 Set Up Asset Hierarchy**

#### **Create Asset Types**:
1. **Login** to Maximo Application Suite
2. **Go to**: Asset Management → Asset Types
3. **Create Types**:
   - `TRAIN` - Main train asset type
   - `CAR` - Individual car type
   - `COMPONENT` - Train components

#### **Create Location Hierarchy**:
1. **Go to**: Asset Management → Locations
2. **Create Locations**:
   - `KOCHI_DEPOT` - Main depot
   - `DEPOT_A` - Service depot A
   - `DEPOT_B` - Service depot B
   - `PLATFORM_1`, `PLATFORM_2`, etc.

### **2.2 Set Up Work Order Types**

#### **Create Work Types**:
1. **Go to**: Work Management → Work Types
2. **Create Types**:
   - `PM` - Preventive Maintenance
   - `CM` - Corrective Maintenance
   - `EM` - Emergency Maintenance
   - `INSP` - Inspection

#### **Create Priority Levels**:
1. **Go to**: Work Management → Priorities
2. **Set Priorities**:
   - `1` - Critical (Emergency)
   - `2` - High (Safety issues)
   - `3` - Medium (Planned maintenance)
   - `4` - Low (Routine tasks)

### **2.3 Create Sample Train Assets**

#### **Add Train Assets**:
1. **Go to**: Asset Management → Assets
2. **Create Assets** for each train:
   ```
   Asset Number: T001
   Description: Kochi Metro Train Set 001
   Asset Type: TRAIN
   Location: DEPOT_A
   Status: ACTIVE
   
   Asset Number: T002
   Description: Kochi Metro Train Set 002
   Asset Type: TRAIN
   Location: DEPOT_A
   Status: ACTIVE
   
   ... (continue for all 25 trains)
   ```

### **2.4 Create Sample Work Orders**

#### **Add Work Orders**:
1. **Go to**: Work Management → Work Orders
2. **Create Work Orders**:
   ```
   Work Order: JC001
   Description: Regular maintenance check for T001
   Asset: T001
   Work Type: PM
   Priority: 3
   Status: WAPPR
   Assigned To: John Smith
   
   Work Order: JC002
   Description: Brake system inspection for T002
   Asset: T002
   Work Type: CM
   Priority: 2
   Status: INPRG
   Assigned To: Alice Johnson
   ```

## 🔑 **Step 3: Configure API Access**

### **3.1 Enable API Access**

#### **In Maximo Application Suite**:
1. **Go to**: Administration → System Configuration
2. **Enable**: REST API access
3. **Configure**: OAuth 2.0 settings
4. **Create**: API client credentials

#### **Get API Credentials**:
1. **Go to**: Administration → API Management
2. **Create Client**:
   - Client ID: `kmrl-intellifleet-client`
   - Client Secret: (generated)
   - Scopes: `maximo.api`
   - Grant Types: `client_credentials`

### **3.2 Test API Connection**

#### **Test with cURL**:
```bash
# Get access token
curl -X POST "https://your-instance.maas.ibm.com/maximo/api/oslc/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET&scope=maximo.api"

# Test work order API
curl -X GET "https://your-instance.maas.ibm.com/maximo/api/oslc/os/mxapiwodetail" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Accept: application/json"
```

## ⚙️ **Step 4: Configure KMRL IntelliFleet**

### **4.1 Set Environment Variables**

#### **Create `.env` file**:
```bash
# IBM Maximo Configuration
KMRL_MAXIMO_BASE_URL=https://your-instance.maas.ibm.com
KMRL_MAXIMO_TENANT_ID=your-tenant-id
KMRL_MAXIMO_CLIENT_ID=kmrl-intellifleet-client
KMRL_MAXIMO_CLIENT_SECRET=your-client-secret

# Optional: Username/Password (if using basic auth)
KMRL_MAXIMO_USERNAME=your-username
KMRL_MAXIMO_PASSWORD=your-password
```

#### **Or set system environment variables**:
```bash
export KMRL_MAXIMO_BASE_URL="https://your-instance.maas.ibm.com"
export KMRL_MAXIMO_TENANT_ID="your-tenant-id"
export KMRL_MAXIMO_CLIENT_ID="kmrl-intellifleet-client"
export KMRL_MAXIMO_CLIENT_SECRET="your-client-secret"
```

### **4.2 Update System Configuration**

#### **Modify `config/settings.py`**:
```python
# Maximo Integration Settings
MAXIMO = {
    'enabled': True,
    'base_url': os.environ.get('KMRL_MAXIMO_BASE_URL', ''),
    'tenant_id': os.environ.get('KMRL_MAXIMO_TENANT_ID', ''),
    'client_id': os.environ.get('KMRL_MAXIMO_CLIENT_ID', ''),
    'client_secret': os.environ.get('KMRL_MAXIMO_CLIENT_SECRET', ''),
    'sync_interval': 300,  # 5 minutes
    'batch_size': 100
}
```

### **4.3 Test Integration**

#### **Run Integration Test**:
```bash
python -c "
from src.production_maximo_integration import get_production_maximo
maximo = get_production_maximo()
status = maximo.get_connection_status()
print('Connection Status:', status)
work_orders = maximo.fetch_work_orders()
print('Work Orders:', len(work_orders))
"
```

## 📊 **Step 5: Data Mapping**

### **5.1 Train Data Mapping**

#### **Maximo Asset → KMRL Train**:
```python
# Maximo Asset Fields → KMRL Train Fields
{
    'assetnum': 'train_id',           # T001, T002, etc.
    'description': 'description',     # Train description
    'status': 'status',              # ACTIVE, INACTIVE
    'location': 'depot_id',          # DEPOT_A, DEPOT_B
    'installdate': 'install_date',   # Installation date
    'lastpmdate': 'last_maintenance_date',
    'nextpmdate': 'next_maintenance_date',
    'condition': 'condition',        # GOOD, FAIR, POOR
    'priority': 'priority'           # 1-4 priority
}
```

### **5.2 Work Order Mapping**

#### **Maximo Work Order → KMRL Job Card**:
```python
# Maximo Work Order Fields → KMRL Job Card Fields
{
    'wonum': 'job_card_id',          # JC001, JC002, etc.
    'description': 'description',    # Work description
    'status': 'status',             # WAPPR→open, INPRG→in_progress, COMP→closed
    'assetnum': 'train_id',         # T001, T002, etc.
    'worktype': 'work_type',        # PM→preventive, CM→corrective, EM→emergency
    'priority': 'priority',         # 1→critical, 2→high, 3→medium, 4→low
    'assignedto': 'assigned_technician',
    'schedstart': 'start_date',
    'schedfinish': 'end_date',
    'actualstart': 'actual_start',
    'actualfinish': 'actual_finish'
}
```

## 🔄 **Step 6: Automated Data Sync**

### **6.1 Set Up Scheduled Sync**

#### **Create sync script** (`scripts/sync_maximo.py`):
```python
#!/usr/bin/env python3
"""
Scheduled Maximo data synchronization
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from production_maximo_integration import get_production_maximo

def sync_maximo_data():
    """Sync data from Maximo"""
    try:
        maximo = get_production_maximo()
        
        # Fetch and update work orders
        work_orders = maximo.fetch_work_orders()
        print(f"✅ Synced {len(work_orders)} work orders")
        
        # Fetch and update assets
        assets = maximo.fetch_assets()
        print(f"✅ Synced {len(assets)} assets")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return False

if __name__ == "__main__":
    sync_maximo_data()
```

#### **Set up cron job** (Linux/Mac):
```bash
# Sync every 5 minutes
*/5 * * * * /path/to/venv/bin/python /path/to/scripts/sync_maximo.py

# Or sync every hour
0 * * * * /path/to/venv/bin/python /path/to/scripts/sync_maximo.py
```

#### **Set up Windows Task Scheduler**:
1. **Open**: Task Scheduler
2. **Create Task**:
   - Name: `KMRL Maximo Sync`
   - Trigger: Every 5 minutes
   - Action: Start program
   - Program: `python.exe`
   - Arguments: `C:\path\to\scripts\sync_maximo.py`

### **6.2 Real-time Integration**

#### **Webhook Integration** (Advanced):
1. **Configure Maximo webhooks** for real-time updates
2. **Set up webhook endpoint** in KMRL system
3. **Process real-time updates** as they occur

## 📈 **Step 7: Monitoring & Maintenance**

### **7.1 Monitor Integration Health**

#### **Check Connection Status**:
```bash
curl http://127.0.0.1:8000/maximo/status
```

#### **View System Logs**:
```bash
tail -f logs/kmrl_intellifleet.log | grep -i maximo
```

### **7.2 Data Quality Monitoring**

#### **Set up Alerts**:
- API connection failures
- Data sync errors
- Authentication issues
- Data validation failures

### **7.3 Regular Maintenance**

#### **Weekly Tasks**:
- Review sync logs
- Check data quality metrics
- Update API credentials if needed
- Test backup/restore procedures

#### **Monthly Tasks**:
- Review Maximo configuration
- Update asset information
- Clean up old work orders
- Performance optimization

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Authentication Failures**
```bash
# Check credentials
echo $KMRL_MAXIMO_CLIENT_ID
echo $KMRL_MAXIMO_CLIENT_SECRET

# Test authentication
curl -X POST "https://your-instance.maas.ibm.com/maximo/api/oslc/token" \
  -d "grant_type=client_credentials&client_id=$KMRL_MAXIMO_CLIENT_ID&client_secret=$KMRL_MAXIMO_CLIENT_SECRET"
```

#### **2. API Connection Issues**
```bash
# Test network connectivity
ping your-instance.maas.ibm.com

# Check SSL certificate
openssl s_client -connect your-instance.maas.ibm.com:443
```

#### **3. Data Sync Problems**
```bash
# Check database connectivity
python -c "from src.db import get_connection; print('DB OK')"

# Test Maximo integration
python -c "from src.production_maximo_integration import get_production_maximo; print('Maximo OK')"
```

### **Support Resources**

#### **IBM Maximo Documentation**:
- https://www.ibm.com/docs/en/maximo-application-suite
- https://developer.ibm.com/learningpaths/get-started-maximo-application-suite/

#### **API Reference**:
- https://www.ibm.com/docs/en/maximo-application-suite/8.11.0?topic=api-rest-api-reference

#### **Community Support**:
- IBM Maximo Community: https://community.ibm.com/community/user/aiops/communities/community-home
- Stack Overflow: Tag `ibm-maximo`

## 💰 **Cost Considerations**

### **IBM Maximo Pricing**

#### **Trial Version**:
- **Free**: 30-day trial
- **Limitations**: Limited users, basic features

#### **Production Plans**:
- **Standard**: $50-100 per user per month
- **Enterprise**: Custom pricing
- **Cloud**: Pay-as-you-use model

#### **Cost Optimization**:
- Start with trial version
- Use minimal user licenses
- Optimize API calls
- Implement efficient data sync

## ✅ **Success Checklist**

- [ ] IBM Maximo account created
- [ ] Service instance configured
- [ ] API credentials obtained
- [ ] Train assets created in Maximo
- [ ] Work orders set up
- [ ] Environment variables configured
- [ ] Integration tested
- [ ] Data sync working
- [ ] Monitoring set up
- [ ] Documentation updated

## 🎯 **Next Steps**

1. **Start with trial version** to test integration
2. **Create sample data** in Maximo
3. **Test API connectivity** and data sync
4. **Set up monitoring** and alerts
5. **Plan production deployment** with proper licensing
6. **Train users** on Maximo interface
7. **Implement backup** and disaster recovery

This setup will give you a production-ready IBM Maximo integration that works exactly like KMRL would use in their real operations!
