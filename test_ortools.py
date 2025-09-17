try:
    from ortools.sat.python import cp_model
    print('OR-Tools available')
except ImportError:
    print('OR-Tools not available')
