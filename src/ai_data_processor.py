"""
src/ai_data_processor.py - FIXED VERSION

Fixes the critical bay assignment field mapping issue and ensures proper
data flow from constraint engine to web dashboards.
"""

class AIDataProcessor:
    """Processes AI optimization results for web dashboard visualization"""
    
    # Cost Calculation Constants
    BASE_SAVINGS_PER_TRAIN = 23000
    ANNUAL_OPERATIONAL_FACTOR = 0.6
    MAINTENANCE_SAVINGS_PER_RISK_POINT = 500

    def __init__(self, optimizer, constraint_engine, data_loader):
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        self.data_loader = data_loader

    def get_train_status_summary(self):
        """Get comprehensive train status summary"""
        summary = {
            'total_trains': 0,
            'inducted_trains': 0,
            'ready_trains': 0,
            'maintenance_trains': 0,
            'standby_trains': 0,
            'ineligible_trains': 0
        }

        if not hasattr(self.optimizer, 'optimized_result') or not self.optimizer.optimized_result:
            return summary

        results = self.optimizer.optimized_result
        
        # Process inducted trains
        inducted_trains = results.get('inducted_trains', [])
        summary['inducted_trains'] = len(inducted_trains)
        
        # Count by status for inducted trains
        for train in inducted_trains:
            priority_score = train.get('priority_score', 0)
            if priority_score >= 80:
                summary['ready_trains'] += 1
            elif priority_score >= 60:
                summary['ready_trains'] += 1
            else:
                summary['maintenance_trains'] += 1
        
        # Add standby trains
        standby_trains = results.get('standby_trains', [])
        summary['standby_trains'] = len(standby_trains)
        
        # Add ineligible trains from constraint engine
        if hasattr(self.constraint_engine, 'constraint_result'):
            constraint_result = self.constraint_engine.constraint_result
            ineligible = constraint_result.get('ineligible_trains', [])
            summary['ineligible_trains'] = len(ineligible)
        
        # Calculate total trains
        summary['total_trains'] = (summary['inducted_trains'] + 
                                 summary['standby_trains'] + 
                                 summary['ineligible_trains'])
        
        return summary

    def get_detailed_train_list(self):
        """FIXED: Get detailed list with proper bay assignment mapping"""
        train_details = []

        if not hasattr(self.optimizer, 'optimized_result') or not self.optimizer.optimized_result:
            return train_details

        results = self.optimizer.optimized_result
        
        # Process inducted trains with FIXED field mapping
        inducted_trains = results.get('inducted_trains', [])
        for i, train in enumerate(inducted_trains, 1):
            # FIXED: Map assigned_bay to bay_assignment for UI consistency
            bay_assignment = train.get('assigned_bay', train.get('bay_assignment', 'N/A'))
            
            train_details.append({
                'rank': i,
                'train_id': train.get('train_id', 'N/A'),
                'status': self._get_train_status(train),
                'bay_assignment': bay_assignment,  # FIXED: Consistent field name
                'priority_score': train.get('priority_score', train.get('composite_score', 0.0)),
                'branding_hours': train.get('branding_hours_left', 0.0),
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': True
            })

        # Process standby trains
        standby_trains = results.get('standby_trains', [])
        for train in standby_trains:
            train_details.append({
                'rank': '-',
                'train_id': train.get('train_id', 'N/A'),
                'status': 'Standby',
                'bay_assignment': 'N/A',
                'priority_score': train.get('priority_score', train.get('composite_score', 0.0)),
                'branding_hours': train.get('branding_hours_left', 0.0),
                'mileage_km': train.get('mileage_km', 0),
                'fitness_valid': train.get('fitness_valid_until', 'Unknown'),
                'inducted': False
            })

        # FIXED: Add ineligible trains with better error handling
        if hasattr(self.constraint_engine, 'constraint_result'):
            constraint_result = self.constraint_engine.constraint_result
            ineligible_trains = constraint_result.get('ineligible_trains', [])
            
            for train in ineligible_trains:
                if isinstance(train, dict):
                    train_id = train.get('train_id', 'Unknown')
                    conflicts = train.get('conflicts', [])
                    reason = conflicts[0] if conflicts else 'Constraint violation'
                else:
                    train_id = str(train)
                    reason = 'Constraint violation'
                
                train_details.append({
                    'rank': '-',
                    'train_id': train_id,
                    'status': f'Ineligible ({reason})',
                    'bay_assignment': 'N/A',
                    'priority_score': 0.0,
                    'branding_hours': 0.0,
                    'mileage_km': 0,
                    'fitness_valid': 'Expired/Invalid',
                    'inducted': False
                })

        return train_details

    def _get_train_status(self, train):
        """Determine train status based on optimization results"""
        score = train.get('priority_score', train.get('composite_score', 0))
        if score >= 80:
            return 'Ready'
        elif score >= 60:
            return 'Ready (Caution)'
        else:
            return 'Maintenance Required'

    def get_performance_metrics(self):
        """Get performance metrics with authentic calculations"""
        if not hasattr(self.optimizer, 'optimized_result') or not self.optimizer.optimized_result:
            metrics = self._get_default_metrics()
            metrics['optimization_score'] = metrics.get('system_performance', 0)
            return metrics

        results = self.optimizer.optimized_result
        inducted_trains = results.get('inducted_trains', [])

        if not inducted_trains:
            return self._get_default_metrics()

        # Calculate authentic metrics
        avg_score = sum(t.get('priority_score', t.get('composite_score', 0)) for t in inducted_trains) / len(inducted_trains)
        total_branding = sum(t.get('branding_hours_left', 0) for t in inducted_trains)

        # Performance multiplier based on actual scores
        performance_multiplier = avg_score / 100
        
        # Branding compliance factor
        branding_factor = 1.0
        if total_branding > 0:
            avg_branding_hours = total_branding / len(inducted_trains)
            if avg_branding_hours >= 8:
                branding_factor = 1.3
            elif avg_branding_hours >= 4:
                branding_factor = 1.15

        # Cost calculations
        trains_processed = len(inducted_trains)
        tonight_savings = (self.BASE_SAVINGS_PER_TRAIN * trains_processed * 
                          performance_multiplier * branding_factor)
        
        annual_days = 365 * self.ANNUAL_OPERATIONAL_FACTOR
        annual_savings = tonight_savings * annual_days
        
        maintenance_risk_reduction = max(0, 100 - avg_score)
        maintenance_savings = (trains_processed * maintenance_risk_reduction * 
                             self.MAINTENANCE_SAVINGS_PER_RISK_POINT)

        result = {
            'system_performance': avg_score,
            'service_readiness': min(100, avg_score * 1.1),
            'maintenance_risk': max(0, 100 - avg_score),
            'branding_compliance': min(100, (total_branding / len(inducted_trains)) * 10) if len(inducted_trains) > 0 else 0,
            'cost_savings': int(tonight_savings + maintenance_savings),
            'annual_savings': int(annual_savings),
            'trains_processed': trains_processed,
            'performance_multiplier': performance_multiplier,
            'branding_factor': branding_factor,
            'calculation_basis': f'Based on {trains_processed} trains with avg score {avg_score:.1f}'
        }
        # Add alias expected by tests
        result['optimization_score'] = result['system_performance']
        return result

    def _get_default_metrics(self):
        """Return default metrics when no optimization data is available"""
        return {
            'system_performance': 0,
            'service_readiness': 0,
            'maintenance_risk': 100,
            'branding_compliance': 0,
            'cost_savings': 0,
            'annual_savings': 0,
            'trains_processed': 0,
            'calculation_basis': 'No optimization data available'
        }

    def get_constraint_violations(self):
        """Get constraint violations for display"""
        violations = []

        # Get conflicts from constraint engine
        if hasattr(self.constraint_engine, 'constraint_result'):
            constraint_result = self.constraint_engine.constraint_result
            conflicts = constraint_result.get('conflicts', [])
            
            # Group conflicts by train
            current_train_id = None
            current_violations = []
            
            for conflict in conflicts:
                if isinstance(conflict, str) and ':' in conflict:
                    train_id, violation = conflict.split(':', 1)
                    train_id = train_id.strip()
                    violation = violation.strip()
                    
                    if train_id != current_train_id:
                        if current_train_id and current_violations:
                            violations.append({
                                'train_id': current_train_id,
                                'violations': current_violations
                            })
                        current_train_id = train_id
                        current_violations = [violation]
                    else:
                        current_violations.append(violation)
                else:
                    violations.append({
                        'train_id': 'Unknown',
                        'violations': [str(conflict)]
                    })
            
            # Add the last train's violations
            if current_train_id and current_violations:
                violations.append({
                    'train_id': current_train_id,
                    'violations': current_violations
                })

        return violations