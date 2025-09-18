# KMRL IntelliFleet Data Integration Guide

## Overview

The KMRL IntelliFleet system integrates data from multiple sources to provide comprehensive train induction optimization. This guide explains how to set up and manage all data sources.

## Data Sources

### 1. 🚂 Trains Data
**Purpose**: Core fleet information including fitness certificates, mileage, and operational status.

**Required Fields**:
- `train_id` (string): Unique train identifier (e.g., "T001")
- `mileage_km` (integer): Current mileage in kilometers
- `branding_hours_left` (integer): Remaining branding exposure hours
- `fitness_valid_until` (date): Fitness certificate expiry date (YYYY-MM-DD)
- `cleaning_slot_id` (string): Assigned cleaning slot (e.g., "CS001")
- `bay_geometry_score` (float): Bay compatibility score (0.0-1.0)

**Optional Fields**:
- `depot_id` (string): Depot identifier (defaults to "DepotA")
- `last_maintenance_date` (date): Last maintenance date

**Sample Data**:
```csv
train_id,mileage_km,branding_hours_left,fitness_valid_until,cleaning_slot_id,bay_geometry_score,depot_id,last_maintenance_date
T001,25000,120,2025-12-31,CS001,0.85,DepotA,2025-08-15
T002,28000,96,2025-12-30,CS002,0.90,DepotA,2025-08-20
```

### 2. 🔧 Job Cards (Maximo Integration)
**Purpose**: Maintenance work orders and status from IBM Maximo or manual entry.

**Required Fields**:
- `train_id` (string): Train identifier
- `job_card_id` (string): Unique job card identifier
- `status` (string): Work order status (open, in_progress, closed, cancelled)
- `work_type` (string): Type of work (preventive, corrective, emergency, inspection)
- `priority` (string): Priority level (low, medium, high, critical)

**Optional Fields**:
- `description` (string): Work description
- `assigned_technician` (string): Technician name
- `estimated_duration` (integer): Estimated hours
- `start_date` (date): Work start date
- `end_date` (date): Work completion date

**Sample Data**:
```csv
train_id,job_card_id,status,work_type,priority,description,assigned_technician,estimated_duration
T001,JC001,closed,preventive,medium,Regular maintenance check,John Smith,2
T002,JC002,open,corrective,high,Brake system inspection,Alice Johnson,4
```

### 3. 🧹 Cleaning Slots
**Purpose**: Available cleaning bays and scheduling information.

**Required Fields**:
- `slot_id` (string): Unique slot identifier
- `available_bays` (integer): Number of available bays
- `start_time` (time): Slot start time (HH:MM)
- `end_time` (time): Slot end time (HH:MM)

**Optional Fields**:
- `assigned_crew` (string): Crew identifier
- `cleaning_type` (string): Type of cleaning (deep_clean, regular_clean)

**Sample Data**:
```csv
slot_id,available_bays,start_time,end_time,assigned_crew,cleaning_type
CS001,2,21:00,23:00,Crew A,deep_clean
CS002,2,22:00,00:00,Crew B,regular_clean
```

### 4. 🏗️ Bay Configuration
**Purpose**: Service bay types, capacities, and geometry information.

**Required Fields**:
- `bay_id` (string): Unique bay identifier
- `bay_type` (string): Bay type (service, maintenance, storage)
- `max_capacity` (integer): Maximum train capacity
- `geometry_score` (float): Geometry compatibility score (0.0-1.0)

**Optional Fields**:
- `depot_id` (string): Depot identifier
- `power_available` (boolean): Power availability
- `status` (string): Bay status (available, occupied, maintenance)

**Sample Data**:
```csv
bay_id,bay_type,max_capacity,geometry_score,depot_id,power_available,status
SB001,service,2,0.95,DepotA,true,available
MB001,maintenance,1,0.80,DepotA,true,available
```

### 5. 📝 Branding Contracts
**Purpose**: Branding contract commitments and exposure requirements.

**Required Fields**:
- `contract_id` (string): Unique contract identifier
- `brand` (string): Brand name
- `train_id` (string): Assigned train
- `hours_committed` (integer): Committed exposure hours
- `start_date` (date): Contract start date
- `end_date` (date): Contract end date

**Optional Fields**:
- `priority` (string): Contract priority (low, medium, high)
- `notes` (string): Additional notes

**Sample Data**:
```csv
contract_id,brand,train_id,hours_committed,start_date,end_date,priority,notes
BC001,MetroCorp,T001,200,2025-01-01,2025-12-31,high,Main sponsor
BC002,CityBank,T002,150,2025-06-01,2025-12-31,medium,Financial services
```

### 6. 📊 Historical Outcomes
**Purpose**: Historical operational data for ML training and analysis.

**Required Fields**:
- `date` (date): Operation date
- `train_id` (string): Train identifier
- `inducted` (integer): Whether train was inducted (0 or 1)
- `failures` (integer): Number of failures

**Optional Fields**:
- `notes` (string): Additional notes
- `energy_consumed_kwh` (float): Energy consumption
- `branding_sla_met` (float): SLA compliance score

**Sample Data**:
```csv
date,train_id,inducted,failures,notes,energy_consumed_kwh,branding_sla_met
2025-09-15,T001,1,0,Successful induction,180.5,1.0
2025-09-15,T002,1,1,Minor brake issue,175.2,0.8
```

## Data Integration Methods

### 1. 📤 CSV Upload via Web Interface
**Access**: http://127.0.0.1:8051 (Data Management Dashboard)

**Features**:
- Drag & drop CSV upload
- Real-time validation
- Format guides and templates
- Error reporting
- Preview before upload

**Steps**:
1. Open Data Management Dashboard
2. Select data type from dropdown
3. Upload CSV file
4. Review validation results
5. Confirm upload

### 2. 🔌 API Integration
**Base URL**: http://127.0.0.1:8000

**Endpoints**:
- `POST /ingest/{table}` - Upload CSV/JSON data
- `GET /ingest/templates/{table}` - Get column requirements
- `GET /ingest/spec/{table}` - Get detailed specifications
- `POST /maximo/upload` - Upload Maximo data
- `GET /maximo/templates` - Get Maximo templates

**Example API Call**:
```python
import requests

# Upload trains data
with open('trains.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://127.0.0.1:8000/ingest/trains', files=files)
    print(response.json())
```

### 3. 🗄️ Database Direct Access
**Database**: SQLite (kmrl_intellifleet.db)

**Tables**:
- `trains` - Train fleet data
- `job_cards` - Maintenance work orders
- `cleaning_slots` - Cleaning bay schedules
- `bay_config` - Bay configuration
- `branding_contracts` - Branding agreements
- `outcomes` - Historical outcomes
- `integration_heartbeat` - Data freshness tracking

## Real-time Data Simulation

### IoT Sensors
**Simulation**: Automatic generation of realistic sensor data
**Data Types**:
- Temperature, humidity, vibration
- GPS location tracking
- Energy consumption
- Door status, brake pressure
- HVAC performance

**Configuration**: Automatically runs when system starts
**Access**: WebSocket at ws://127.0.0.1:8765

### Computer Vision
**Simulation**: Automated defect detection simulation
**Features**:
- Visual inspection reports
- Defect classification
- Confidence scores
- Inspection timestamps

**Configuration**: Automatically runs when system starts

## Maximo Integration Options

### Option 1: API Integration (Recommended)
**Requirements**:
- IBM Maximo API access
- API key or OAuth credentials
- Network connectivity

**Configuration**:
```python
# Set environment variables
export KMRL_MAXIMO_API_ENDPOINT="https://your-maximo-instance/api"
export KMRL_MAXIMO_API_KEY="your-api-key"
```

### Option 2: File Export Integration
**Requirements**:
- Maximo export capability
- Scheduled file exports
- File system access

**Configuration**:
```python
# Set export file path
export KMRL_MAXIMO_EXPORT="/path/to/maximo_export.csv"
```

### Option 3: Manual Data Entry
**Requirements**:
- CSV templates
- Data validation
- Manual upload process

**Process**:
1. Download CSV templates
2. Fill in data manually
3. Upload via web interface
4. Validate and confirm

## Data Quality & Validation

### Automatic Validation
- **Required Fields**: All required columns must be present
- **Data Types**: Automatic type conversion and validation
- **Value Ranges**: Numeric values within expected ranges
- **Date Formats**: ISO date format validation
- **Enumeration**: Status values from predefined lists

### Quality Metrics
- **Completeness**: Percentage of non-null values
- **Accuracy**: Data within expected ranges
- **Consistency**: Cross-field validation
- **Timeliness**: Data freshness indicators

### Error Handling
- **Row-level Errors**: Detailed error reporting per row
- **Column-level Errors**: Field-specific validation messages
- **Bulk Upload**: Partial success handling
- **Rollback**: Transaction safety

## Best Practices

### 1. Data Preparation
- Use provided CSV templates
- Validate data before upload
- Include all required fields
- Use consistent date formats (YYYY-MM-DD)
- Ensure unique identifiers

### 2. Regular Updates
- Update train data daily
- Refresh job cards after maintenance
- Update cleaning slots weekly
- Monitor data freshness indicators

### 3. Error Monitoring
- Check validation results
- Monitor system alerts
- Review data quality metrics
- Address errors promptly

### 4. Backup & Recovery
- Regular database backups
- Export critical data
- Test recovery procedures
- Maintain data archives

## Troubleshooting

### Common Issues

**1. Upload Failures**
- Check file format (CSV required)
- Verify required columns
- Validate data types
- Check file size limits

**2. Validation Errors**
- Review error messages
- Check data formats
- Verify value ranges
- Update missing fields

**3. API Connection Issues**
- Verify endpoint URLs
- Check authentication
- Test network connectivity
- Review API documentation

**4. Data Freshness Issues**
- Check heartbeat status
- Verify data sources
- Restart simulations
- Update manual data

### Support
- Check system logs
- Review API documentation
- Contact system administrator
- Submit support tickets

## Security Considerations

### Access Control
- Role-based permissions
- API authentication
- Secure file uploads
- Audit logging

### Data Protection
- Encrypted connections
- Secure storage
- Access logging
- Data retention policies

### Compliance
- Data privacy regulations
- Audit trail maintenance
- Backup procedures
- Recovery testing
