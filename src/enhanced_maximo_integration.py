"""
Enhanced Maximo Integration for KMRL IntelliFleet
Handles both automated API integration and manual data entry scenarios
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from .db import init_db, upsert, fetch_df, upsert_heartbeat

class EnhancedMaximoIntegration:
    """Enhanced Maximo integration with manual fallback capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_endpoint = self.config.get('api_endpoint', '')
        self.api_key = self.config.get('api_key', '')
        self.username = self.config.get('username', '')
        self.password = self.config.get('password', '')
        self.manual_mode = not bool(self.api_endpoint and self.api_key)
        
        # Initialize database
        init_db()
        
        # Set up manual data templates
        self._setup_manual_templates()
    
    def _setup_manual_templates(self):
        """Set up templates for manual data entry"""
        self.manual_templates = {
            'job_cards': {
                'required_fields': ['train_id', 'job_card_id', 'status', 'work_type', 'priority'],
                'optional_fields': ['description', 'assigned_technician', 'estimated_duration', 'start_date', 'end_date'],
                'status_options': ['open', 'in_progress', 'closed', 'cancelled'],
                'work_types': ['preventive', 'corrective', 'emergency', 'inspection'],
                'priorities': ['low', 'medium', 'high', 'critical']
            },
            'fitness_certificates': {
                'required_fields': ['train_id', 'certificate_type', 'valid_until', 'issued_by'],
                'optional_fields': ['certificate_number', 'issued_date', 'notes'],
                'certificate_types': ['rolling_stock', 'signaling', 'telecom', 'safety'],
                'validity_periods': [30, 60, 90, 180, 365]  # days
            }
        }
    
    def check_connection(self) -> Dict[str, Any]:
        """Check Maximo API connection status"""
        if self.manual_mode:
            return {
                'status': 'manual_mode',
                'message': 'Operating in manual data entry mode',
                'last_check': datetime.now().isoformat(),
                'api_available': False
            }
        
        try:
            # Test API connection
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{self.api_endpoint}/api/workorder",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    'status': 'connected',
                    'message': 'Maximo API connection successful',
                    'last_check': datetime.now().isoformat(),
                    'api_available': True,
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status': 'error',
                    'message': f'API returned status {response.status_code}',
                    'last_check': datetime.now().isoformat(),
                    'api_available': False
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection failed: {str(e)}',
                'last_check': datetime.now().isoformat(),
                'api_available': False
            }
    
    def fetch_job_cards(self, train_ids: List[str] = None, status: str = None) -> pd.DataFrame:
        """Fetch job cards from Maximo API or manual data"""
        if not self.manual_mode:
            return self._fetch_from_api(train_ids, status)
        else:
            return self._fetch_from_manual_data(train_ids, status)
    
    def _fetch_from_api(self, train_ids: List[str] = None, status: str = None) -> pd.DataFrame:
        """Fetch job cards from Maximo API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Build query parameters
            params = {}
            if train_ids:
                params['train_id'] = ','.join(train_ids)
            if status:
                params['status'] = status
            
            response = requests.get(
                f"{self.api_endpoint}/api/workorder",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data.get('workorders', []))
                
                # Update heartbeat
                upsert_heartbeat('maximo', 'ok', f'Fetched {len(df)} job cards')
                
                return df
            else:
                raise Exception(f"API returned status {response.status_code}")
                
        except Exception as e:
            # Fallback to manual data on API failure
            logger.warning(f"Maximo API failed: {e}. Falling back to manual data.")
            upsert_heartbeat('maximo', 'error', f'API failed: {str(e)}')
            return self._fetch_from_manual_data(train_ids, status)
    
    def _fetch_from_manual_data(self, train_ids: List[str] = None, status: str = None) -> pd.DataFrame:
        """Fetch job cards from manual data (database)"""
        try:
            df = fetch_df('job_cards')
            
            # Apply filters
            if train_ids:
                df = df[df['train_id'].isin(train_ids)]
            if status:
                df = df[df['status'] == status]
            
            # Update heartbeat
            upsert_heartbeat('maximo', 'manual', f'Using manual data: {len(df)} records')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch manual data: {e}")
            upsert_heartbeat('maximo', 'error', f'Manual data failed: {str(e)}')
            return pd.DataFrame()
    
    def upload_manual_data(self, data: List[Dict[str, Any]], data_type: str = 'job_cards') -> Dict[str, Any]:
        """Upload manual data to the system"""
        try:
            if data_type == 'job_cards':
                # Validate job card data
                validation_result = self._validate_job_cards(data)
                if not validation_result['valid']:
                    return {
                        'success': False,
                        'errors': validation_result['errors'],
                        'rows_processed': 0
                    }
                
                # Add metadata
                for record in data:
                    record['last_updated'] = datetime.now().isoformat()
                    record['source'] = 'manual_entry'
                
                # Upsert to database
                upsert('job_cards', data, ['train_id', 'job_card_id'])
                
                # Update heartbeat
                upsert_heartbeat('maximo', 'manual', f'Uploaded {len(data)} job cards')
                
                return {
                    'success': True,
                    'rows_processed': len(data),
                    'message': f'Successfully uploaded {len(data)} job cards'
                }
            
            elif data_type == 'fitness_certificates':
                # Handle fitness certificates
                return self._upload_fitness_certificates(data)
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported data type: {data_type}'
                }
                
        except Exception as e:
            logger.error(f"Manual data upload failed: {e}")
            upsert_heartbeat('maximo', 'error', f'Upload failed: {str(e)}')
            return {
                'success': False,
                'error': str(e),
                'rows_processed': 0
            }
    
    def _validate_job_cards(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate job card data"""
        errors = []
        template = self.manual_templates['job_cards']
        
        for i, record in enumerate(data):
            # Check required fields
            for field in template['required_fields']:
                if field not in record or not record[field]:
                    errors.append(f"Row {i+1}: Missing required field '{field}'")
            
            # Validate status
            if 'status' in record and record['status'] not in template['status_options']:
                errors.append(f"Row {i+1}: Invalid status '{record['status']}'. Must be one of {template['status_options']}")
            
            # Validate work type
            if 'work_type' in record and record['work_type'] not in template['work_types']:
                errors.append(f"Row {i+1}: Invalid work_type '{record['work_type']}'. Must be one of {template['work_types']}")
            
            # Validate priority
            if 'priority' in record and record['priority'] not in template['priorities']:
                errors.append(f"Row {i+1}: Invalid priority '{record['priority']}'. Must be one of {template['priorities']}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _upload_fitness_certificates(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload fitness certificates"""
        try:
            # Validate fitness certificate data
            template = self.manual_templates['fitness_certificates']
            errors = []
            
            for i, record in enumerate(data):
                # Check required fields
                for field in template['required_fields']:
                    if field not in record or not record[field]:
                        errors.append(f"Row {i+1}: Missing required field '{field}'")
                
                # Validate certificate type
                if 'certificate_type' in record and record['certificate_type'] not in template['certificate_types']:
                    errors.append(f"Row {i+1}: Invalid certificate_type '{record['certificate_type']}'")
            
            if errors:
                return {
                    'success': False,
                    'errors': errors,
                    'rows_processed': 0
                }
            
            # Add metadata
            for record in data:
                record['last_updated'] = datetime.now().isoformat()
                record['source'] = 'manual_entry'
            
            # Update trains table with fitness certificate info
            for record in data:
                train_id = record['train_id']
                certificate_type = record['certificate_type']
                valid_until = record['valid_until']
                
                # Update the trains table
                train_data = [{
                    'train_id': train_id,
                    f'{certificate_type}_valid_until': valid_until,
                    'last_updated': datetime.now().isoformat()
                }]
                upsert('trains', train_data, ['train_id'])
            
            upsert_heartbeat('maximo', 'manual', f'Uploaded {len(data)} fitness certificates')
            
            return {
                'success': True,
                'rows_processed': len(data),
                'message': f'Successfully uploaded {len(data)} fitness certificates'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'rows_processed': 0
            }
    
    def get_manual_data_templates(self) -> Dict[str, Any]:
        """Get templates for manual data entry"""
        return {
            'templates': self.manual_templates,
            'sample_data': {
                'job_cards': [
                    {
                        'train_id': 'T001',
                        'job_card_id': 'JC001',
                        'status': 'open',
                        'work_type': 'preventive',
                        'priority': 'medium',
                        'description': 'Regular maintenance check',
                        'assigned_technician': 'John Smith',
                        'estimated_duration': 2
                    }
                ],
                'fitness_certificates': [
                    {
                        'train_id': 'T001',
                        'certificate_type': 'rolling_stock',
                        'valid_until': '2025-12-31',
                        'issued_by': 'Rolling Stock Department',
                        'certificate_number': 'RS-2025-001',
                        'issued_date': '2025-01-01'
                    }
                ]
            }
        }
    
    def refresh_data(self) -> Dict[str, Any]:
        """Refresh all Maximo data"""
        try:
            # Fetch latest job cards
            job_cards_df = self.fetch_job_cards()
            
            # Update database
            if not job_cards_df.empty:
                job_cards_data = job_cards_df.to_dict('records')
                upsert('job_cards', job_cards_data, ['train_id', 'job_card_id'])
            
            # Update heartbeat
            upsert_heartbeat('maximo', 'ok', f'Refreshed {len(job_cards_df)} records')
            
            return {
                'success': True,
                'job_cards_updated': len(job_cards_df),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            upsert_heartbeat('maximo', 'error', f'Refresh failed: {str(e)}')
            return {
                'success': False,
                'error': str(e)
            }

# Global instance for easy access
maximo_integration = EnhancedMaximoIntegration()

def get_maximo_integration() -> EnhancedMaximoIntegration:
    """Get the global Maximo integration instance"""
    return maximo_integration

