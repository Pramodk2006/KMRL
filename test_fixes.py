#!/usr/bin/env python3
"""
Test script to verify all the fixes applied to the KMRL IntelliFleet system
"""

import sys
import os
sys.path.append('src')

def test_digital_twin_update_method():
    """Test 1: Verify DigitalTwinEngine has update_train_state method"""
    print("🔧 Test 1: Testing DigitalTwinEngine.update_train_state method...")
    
    from src.digital_twin_engine import DigitalTwinEngine
    
    # Create a test digital twin
    initial_data = {
        'trains': {
            'T001': {'location': 'depot', 'status': 'idle', 'mileage_km': 15000}
        },
        'bay_config': {
            'Bay1': {'bay_type': 'service', 'max_capacity': 2, 'geometry_score': 9}
        }
    }
    
    dt = DigitalTwinEngine(initial_data)
    
    # Test the update method
    success = dt.update_train_state('T001', {'status': 'service', 'assigned_bay': 'Bay1'})
    
    if success and hasattr(dt, 'update_train_state'):
        print("   ✅ PASS: update_train_state method exists and works")
        return True
    else:
        print("   ❌ FAIL: update_train_state method missing or not working")
        return False

def test_ai_data_processor_ready_trains():
    """Test 2: Verify AIDataProcessor handles ready_trains field"""
    print("🤖 Test 2: Testing AIDataProcessor ready_trains field...")
    
    try:
        # Import the system components
        from src.data_loader import DataLoader
        from src.constraint_engine import CustomConstraintEngine
        from src.multi_objective_optimizer import MultiObjectiveOptimizer
        
        # Create test components
        data_loader = DataLoader()
        ai_data = data_loader.get_integrated_data()
        constraint_engine = CustomConstraintEngine(ai_data)
        constraint_result = constraint_engine.run_constraint_optimization()
        optimizer = MultiObjectiveOptimizer(constraint_result, ai_data)
        optimized_result = optimizer.optimize_induction_ranking()
        optimizer.optimized_result = optimized_result
        
        # Import and test AIDataProcessor from enterprise_main.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("enterprise_main", "enterprise_main.py")
        enterprise_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enterprise_main)
        
        ai_processor = enterprise_main.AIDataProcessor(optimizer, constraint_engine, data_loader)
        summary = ai_processor.get_train_status_summary()
        
        if 'ready_trains' in summary and 'ineligible_trains' in summary:
            print(f"   ✅ PASS: ready_trains and ineligible_trains fields exist")
            print(f"      - ready_trains: {summary['ready_trains']}")
            print(f"      - ineligible_trains: {summary['ineligible_trains']}")
            return True
        else:
            print("   ❌ FAIL: Missing ready_trains or ineligible_trains fields")
            return False
    except Exception as e:
        print(f"   ❌ FAIL: Error testing AIDataProcessor: {e}")
        return False

def test_inducted_trains_scores():
    """Test 3: Verify inducted trains have proper scores"""
    print("📊 Test 3: Testing inducted trains score display...")
    
    try:
        # Import the system components
        from src.data_loader import DataLoader
        from src.constraint_engine import CustomConstraintEngine
        from src.multi_objective_optimizer import MultiObjectiveOptimizer
        
        # Create test components
        data_loader = DataLoader()
        ai_data = data_loader.get_integrated_data()
        constraint_engine = CustomConstraintEngine(ai_data)
        constraint_result = constraint_engine.run_constraint_optimization()
        optimizer = MultiObjectiveOptimizer(constraint_result, ai_data)
        optimized_result = optimizer.optimize_induction_ranking()
        optimizer.optimized_result = optimized_result
        
        # Import and test AIDataProcessor from enterprise_main.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("enterprise_main", "enterprise_main.py")
        enterprise_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enterprise_main)
        
        ai_processor = enterprise_main.AIDataProcessor(optimizer, constraint_engine, data_loader)
        train_details = ai_processor.get_detailed_train_list()
        
        # Check if inducted trains have scores
        inducted_trains = [t for t in train_details if t.get('inducted', False)]
        
        if inducted_trains:
            has_scores = all(t.get('priority_score', 0) > 0 for t in inducted_trains)
            if has_scores:
                print(f"   ✅ PASS: {len(inducted_trains)} inducted trains have scores")
                for train in inducted_trains[:3]:  # Show first 3
                    print(f"      - {train['train_id']}: Score = {train['priority_score']:.1f}")
                return True
            else:
                print(f"   ❌ FAIL: Inducted trains missing scores")
                return False
        else:
            print("   ⚠️  WARNING: No inducted trains found to test scores")
            return True
    except Exception as e:
        print(f"   ❌ FAIL: Error testing inducted train scores: {e}")
        return False

def test_bay_assignments():
    """Test 4: Verify bay assignments use correct bay IDs"""
    print("🏗️  Test 4: Testing bay assignments...")
    
    try:
        from src.multi_objective_optimizer import MultiObjectiveOptimizer
        from src.data_loader import DataLoader
        from src.constraint_engine import CustomConstraintEngine
        
        # Create test components
        data_loader = DataLoader()
        ai_data = data_loader.get_integrated_data()
        constraint_engine = CustomConstraintEngine(ai_data)
        constraint_result = constraint_engine.run_constraint_optimization()
        optimizer = MultiObjectiveOptimizer(constraint_result, ai_data)
        optimized_result = optimizer.optimize_induction_ranking()
        
        inducted_trains = optimized_result.get('inducted_trains', [])
        
        if inducted_trains:
            # Check if bay assignments use correct format (Bay1, Bay2, etc.)
            bay_assignments = [t.get('assigned_bay', 'N/A') for t in inducted_trains]
            correct_format = all(bay.startswith('Bay') or bay == 'N/A' for bay in bay_assignments)
            
            if correct_format:
                print(f"   ✅ PASS: Bay assignments use correct format")
                print(f"      - Assignments: {bay_assignments}")
                return True
            else:
                print(f"   ❌ FAIL: Incorrect bay assignment format")
                print(f"      - Assignments: {bay_assignments}")
                return False
        else:
            print("   ⚠️  WARNING: No inducted trains found to test bay assignments")
            return True
    except Exception as e:
        print(f"   ❌ FAIL: Error testing bay assignments: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 KMRL IntelliFleet System - Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_digital_twin_update_method,
        test_ai_data_processor_ready_trains,
        test_inducted_trains_scores,
        test_bay_assignments
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ FAIL: Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📋 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System fixes are working correctly.")
        print("\n🚀 Ready to run: python enterprise_main.py")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
