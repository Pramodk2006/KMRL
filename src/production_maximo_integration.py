"""
Production IBM Maximo Integration for KMRL IntelliFleet
Real-world integration with IBM Maximo Application Suite
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from .db import init_db, upsert, fetch_df, upsert_heartbeat

logger = logging.getLogger(__name__)

class ProductionMaximoIntegration:
    """Production-ready IBM Maximo integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # IBM Maximo Application Suite Configuration
        self.base_url = os.environ.get('KMRL_MAXIMO_BASE_URL', '')
        self.api_key = os.environ.get('KMRL_MAXIMO_API_KEY', '')
        self.username = os.environ.get('KMRL_MAXIMO_USERNAME', '')
        self.password = os.environ.get('KMRL_MAXIMO_PASSWORD', '')
        self.tenant_id = os.environ.get('KMRL_MAXIMO_TENANT_ID', '')
        
        # OAuth Configuration
        self.client_id = os.environ.get('KMRL_MAXIMO_CLIENT_ID', '')
        self.client_secret = os.environ.get('KMRL_MAXIMO_CLIENT_SECRET', '')
        
        # API Endpoints
        self.oauth_url = f"{self.base_url}/maximo/api/oslc/token"
        self.workorder_url = f"{self.base_url}/maximo/api/oslc/os/mxapiwodetail"
        self.asset_url = f"{self.base_url}/maximo/api/oslc/os/mxapiasset"
        self.person_url = f"{self.base_url}/maximo/api/oslc/os/mxapiperson"
        
        # Authentication
        self.access_token = None
        self.token_expires = None
        
        # Initialize database
        init_db()
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test Maximo connection and authentication"""
        try:
            if self._authenticate():
                logger.info("✅ IBM Maximo connection successful")
                upsert_heartbeat('maximo', 'connected', 'Production Maximo connected')
            else:
                logger.warning("⚠️ IBM Maximo connection failed - using fallback mode")
                upsert_heartbeat('maximo', 'error', 'Production Maximo connection failed')
        except Exception as e:
            logger.error(f"❌ Maximo connection error: {e}")
            upsert_heartbeat('maximo', 'error', f'Connection error: {str(e)}')
    
    def _authenticate(self) -> bool:
        """Authenticate with IBM Maximo using OAuth 2.0"""
        try:
            if not self.base_url or not self.client_id or not self.client_secret:
                logger.warning("Maximo credentials not configured")
                return False
            
            # OAuth 2.0 Client Credentials flow
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'maximo.api'
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                self.oauth_url,
                data=auth_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
                
                logger.info("✅ Maximo authentication successful")
                return True
            else:
                logger.error(f"❌ Maximo authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Maximo authentication error: {e}")
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authenticated headers for API requests"""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            if not self._authenticate():
                raise Exception("Failed to authenticate with Maximo")
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'maxversion': '1.0'
        }
    
    def fetch_work_orders(self, train_ids: List[str] = None, status: str = None) -> pd.DataFrame:
        """Fetch work orders from IBM Maximo"""
        try:
            headers = self._get_auth_headers()
            
            # Build OSLC query
            query_params = []
            
            # Filter by train IDs (assuming train_id is stored in assetnum or location)
            if train_ids:
                train_filter = " or ".join([f"assetnum='{tid}'" for tid in train_ids])
                query_params.append(f"({train_filter})")
            
            # Filter by status
            if status:
                status_map = {
                    'open': 'WAPPR',
                    'in_progress': 'INPRG',
                    'closed': 'COMP',
                    'cancelled': 'CAN'
                }
                maximo_status = status_map.get(status, status)
                query_params.append(f"status='{maximo_status}'")
            
            # Add default filters for active work orders
            query_params.append("status in ('WAPPR','INPRG','COMP')")
            
            # Build final query
            oslc_query = " and ".join(query_params) if query_params else ""
            
            # OSLC query parameters
            params = {
                'oslc.select': 'wonum,description,status,assetnum,location,worktype,priority,assignedto,actualstart,actualfinish,schedstart,schedfinish',
                'oslc.where': oslc_query,
                'oslc.pageSize': '1000'
            }
            
            response = requests.get(
                self.workorder_url,
                headers=headers,
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                work_orders = data.get('member', [])
                
                # Transform to our format
                transformed_data = []
                for wo in work_orders:
                    transformed_data.append({
                        'train_id': wo.get('assetnum', ''),
                        'job_card_id': wo.get('wonum', ''),
                        'status': self._map_maximo_status(wo.get('status', '')),
                        'work_type': self._map_work_type(wo.get('worktype', '')),
                        'priority': self._map_priority(wo.get('priority', '')),
                        'description': wo.get('description', ''),
                        'assigned_technician': wo.get('assignedto', ''),
                        'start_date': wo.get('schedstart', ''),
                        'end_date': wo.get('schedfinish', ''),
                        'actual_start': wo.get('actualstart', ''),
                        'actual_finish': wo.get('actualfinish', ''),
                        'location': wo.get('location', ''),
                        'last_updated': datetime.now().isoformat(),
                        'source': 'maximo_production'
                    })
                
                df = pd.DataFrame(transformed_data)
                
                # Update database
                if not df.empty:
                    upsert('job_cards', df.to_dict('records'), ['train_id', 'job_card_id'])
                
                # Update heartbeat
                upsert_heartbeat('maximo', 'ok', f'Fetched {len(df)} work orders from production Maximo')
                
                logger.info(f"✅ Fetched {len(df)} work orders from IBM Maximo")
                return df
                
            else:
                logger.error(f"❌ Maximo API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error fetching work orders: {e}")
            upsert_heartbeat('maximo', 'error', f'Fetch error: {str(e)}')
            return pd.DataFrame()
    
    def fetch_assets(self, train_ids: List[str] = None) -> pd.DataFrame:
        """Fetch asset information from IBM Maximo"""
        try:
            headers = self._get_auth_headers()
            
            # Build query for train assets
            query_params = []
            if train_ids:
                asset_filter = " or ".join([f"assetnum='{tid}'" for tid in train_ids])
                query_params.append(f"({asset_filter})")
            
            # Filter for train assets
            query_params.append("assettype='TRAIN'")
            
            oslc_query = " and ".join(query_params) if query_params else "assettype='TRAIN'"
            
            params = {
                'oslc.select': 'assetnum,description,status,location,parent,installdate,lastpmdate,nextpmdate,condition,priority',
                'oslc.where': oslc_query,
                'oslc.pageSize': '1000'
            }
            
            response = requests.get(
                self.asset_url,
                headers=headers,
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                assets = data.get('member', [])
                
                # Transform to our format
                transformed_data = []
                for asset in assets:
                    transformed_data.append({
                        'train_id': asset.get('assetnum', ''),
                        'description': asset.get('description', ''),
                        'status': asset.get('status', ''),
                        'location': asset.get('location', ''),
                        'install_date': asset.get('installdate', ''),
                        'last_pm_date': asset.get('lastpmdate', ''),
                        'next_pm_date': asset.get('nextpmdate', ''),
                        'condition': asset.get('condition', ''),
                        'priority': asset.get('priority', ''),
                        'last_updated': datetime.now().isoformat(),
                        'source': 'maximo_production'
                    })
                
                df = pd.DataFrame(transformed_data)
                logger.info(f"✅ Fetched {len(df)} assets from IBM Maximo")
                return df
                
            else:
                logger.error(f"❌ Maximo assets API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error fetching assets: {e}")
            return pd.DataFrame()
    
    def create_work_order(self, work_order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new work order in IBM Maximo"""
        try:
            headers = self._get_auth_headers()
            
            # Transform our format to Maximo format
            maximo_wo = {
                'wonum': work_order_data.get('job_card_id', ''),
                'description': work_order_data.get('description', ''),
                'assetnum': work_order_data.get('train_id', ''),
                'worktype': self._map_work_type_reverse(work_order_data.get('work_type', '')),
                'priority': self._map_priority_reverse(work_order_data.get('priority', '')),
                'assignedto': work_order_data.get('assigned_technician', ''),
                'schedstart': work_order_data.get('start_date', ''),
                'schedfinish': work_order_data.get('end_date', '')
            }
            
            payload = {
                'member': [maximo_wo]
            }
            
            response = requests.post(
                self.workorder_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"✅ Created work order {work_order_data.get('job_card_id')} in Maximo")
                return {
                    'success': True,
                    'work_order_id': work_order_data.get('job_card_id'),
                    'maximo_response': result
                }
            else:
                logger.error(f"❌ Failed to create work order: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Maximo API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"❌ Error creating work order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_work_order_status(self, work_order_id: str, status: str, notes: str = None) -> Dict[str, Any]:
        """Update work order status in IBM Maximo"""
        try:
            headers = self._get_auth_headers()
            
            # Get current work order
            params = {
                'oslc.where': f"wonum='{work_order_id}'",
                'oslc.select': 'wonum,status,description'
            }
            
            response = requests.get(
                self.workorder_url,
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                work_orders = data.get('member', [])
                
                if work_orders:
                    wo = work_orders[0]
                    wo['status'] = self._map_status_reverse(status)
                    
                    if notes:
                        wo['description'] = f"{wo.get('description', '')}\nUpdate: {notes}"
                    
                    # Update work order
                    update_payload = {'member': [wo]}
                    
                    update_response = requests.put(
                        f"{self.workorder_url}/{wo.get('wonum')}",
                        headers=headers,
                        json=update_payload,
                        timeout=30
                    )
                    
                    if update_response.status_code in [200, 204]:
                        logger.info(f"✅ Updated work order {work_order_id} status to {status}")
                        return {'success': True, 'message': 'Work order updated successfully'}
                    else:
                        return {'success': False, 'error': f"Update failed: {update_response.status_code}"}
                else:
                    return {'success': False, 'error': 'Work order not found'}
            else:
                return {'success': False, 'error': f"Failed to fetch work order: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Error updating work order: {e}")
            return {'success': False, 'error': str(e)}
    
    def _map_maximo_status(self, maximo_status: str) -> str:
        """Map Maximo status to our internal status"""
        status_map = {
            'WAPPR': 'open',
            'INPRG': 'in_progress',
            'COMP': 'closed',
            'CAN': 'cancelled'
        }
        return status_map.get(maximo_status, maximo_status.lower())
    
    def _map_work_type(self, maximo_worktype: str) -> str:
        """Map Maximo work type to our internal work type"""
        worktype_map = {
            'PM': 'preventive',
            'CM': 'corrective',
            'EM': 'emergency',
            'INSP': 'inspection'
        }
        return worktype_map.get(maximo_worktype, maximo_worktype.lower())
    
    def _map_priority(self, maximo_priority: str) -> str:
        """Map Maximo priority to our internal priority"""
        priority_map = {
            '1': 'critical',
            '2': 'high',
            '3': 'medium',
            '4': 'low'
        }
        return priority_map.get(maximo_priority, 'medium')
    
    def _map_work_type_reverse(self, work_type: str) -> str:
        """Map our work type to Maximo work type"""
        reverse_map = {
            'preventive': 'PM',
            'corrective': 'CM',
            'emergency': 'EM',
            'inspection': 'INSP'
        }
        return reverse_map.get(work_type, 'CM')
    
    def _map_priority_reverse(self, priority: str) -> str:
        """Map our priority to Maximo priority"""
        reverse_map = {
            'critical': '1',
            'high': '2',
            'medium': '3',
            'low': '4'
        }
        return reverse_map.get(priority, '3')
    
    def _map_status_reverse(self, status: str) -> str:
        """Map our status to Maximo status"""
        reverse_map = {
            'open': 'WAPPR',
            'in_progress': 'INPRG',
            'closed': 'COMP',
            'cancelled': 'CAN'
        }
        return reverse_map.get(status, 'WAPPR')
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'connected': bool(self.access_token and (not self.token_expires or datetime.now() < self.token_expires)),
            'base_url': self.base_url,
            'tenant_id': self.tenant_id,
            'token_expires': self.token_expires.isoformat() if self.token_expires else None,
            'last_check': datetime.now().isoformat()
        }
    
    def refresh_data(self) -> Dict[str, Any]:
        """Refresh all data from Maximo"""
        try:
            # Fetch work orders
            work_orders_df = self.fetch_work_orders()
            
            # Fetch assets
            assets_df = self.fetch_assets()
            
            return {
                'success': True,
                'work_orders_updated': len(work_orders_df),
                'assets_updated': len(assets_df),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Data refresh failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Global instance
production_maximo = ProductionMaximoIntegration()

def get_production_maximo() -> ProductionMaximoIntegration:
    """Get the global production Maximo integration instance"""
    return production_maximo

