from typing import List, Dict, Optional, Tuple
import math

def solve_bay_assignment_cpsat(
    eligible_trains: List[Dict],
    service_bays_df,
    yard_graph: Optional[Dict[str, Dict[str, float]]] = None,
    train_start_nodes: Optional[Dict[str, str]] = None,
    bay_nodes: Optional[Dict[str, str]] = None,
    shunting_weight: float = 1.0,
    geometry_weight: float = 1.0,
) -> List[Dict]:
    """Assign trains to bays using CP-SAT if available. Falls back to greedy if OR-Tools missing.

    eligible_trains: list of dicts with keys: train_id, bay_geometry_score
    service_bays_df: DataFrame with bay_id, max_capacity, geometry_score
    returns: list of train dicts with assigned_bay and bay_geometry_match
    """
    try:
        from ortools.sat.python import cp_model
        # Temporarily disable CP-SAT to use fallback
        raise ImportError("Temporarily disabled CP-SAT")
    except Exception:
        # Fallback to greedy matching used in constraint engine
        assigned = []
        bays = {}
        for _, bay in service_bays_df.iterrows():
            bays[bay['bay_id']] = {
                'capacity': int(bay['max_capacity']),
                'assigned': 0,
                'geometry_score': float(bay['geometry_score'])
            }
        for train in eligible_trains:
            best_bay = None
            best_diff = float('inf')
            for bay_id, b in bays.items():
                if b['assigned'] < b['capacity']:
                    diff = abs(float(train['bay_geometry_score']) - b['geometry_score'])
                    if diff < best_diff:
                        best_diff = diff
                        best_bay = bay_id
            t = train.copy()
            if best_bay:
                bays[best_bay]['assigned'] += 1
                t['assigned_bay'] = best_bay
                t['bay_geometry_match'] = bays[best_bay]['geometry_score']
            assigned.append(t)
        return assigned

    # Build CP-SAT model
    model = cp_model.CpModel()
    trains = list(eligible_trains)
    bays = list(service_bays_df.to_dict(orient='records'))
    T = range(len(trains))
    B = range(len(bays))

    # Precompute shunting (routing) costs using yard graph if provided
    def dijkstra(start: str, graph: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not graph or start not in graph:
            return {start: 0.0}
        dist = {start: 0.0}
        visited = set()
        while True:
            # pick unvisited node with smallest dist
            cur = None
            cur_d = math.inf
            for n, d in dist.items():
                if n not in visited and d < cur_d:
                    cur, cur_d = n, d
            if cur is None:
                break
            visited.add(cur)
            for nb, w in graph.get(cur, {}).items():
                nd = cur_d + float(w)
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
        return dist

    shunt_costs = {}
    use_shunt = yard_graph is not None and train_start_nodes is not None and bay_nodes is not None and (shunting_weight or 0) > 0
    if use_shunt:
        # For each train, compute distance map
        cache_dist: Dict[str, Dict[str, float]] = {}
        for t in T:
            tid = str(trains[t].get('train_id'))
            start_node = train_start_nodes.get(tid)
            if start_node and start_node not in cache_dist:
                cache_dist[start_node] = dijkstra(start_node, yard_graph)
        for t in T:
            tid = str(trains[t].get('train_id'))
            start_node = train_start_nodes.get(tid)
            for b in B:
                bnode = bay_nodes.get(str(bays[b]['bay_id'])) if bay_nodes else None
                dist_map = cache_dist.get(start_node, {}) if start_node else {}
                cost = float(dist_map.get(bnode, 0.0)) if bnode else 0.0
                shunt_costs[(t, b)] = int(cost * 1000)
    else:
        # Default zero shunting cost when inputs missing
        for t in T:
            for b in B:
                shunt_costs[(t, b)] = 0

    # Decision vars x[t,b] binary
    x = {}
    for t in T:
        for b in B:
            x[(t, b)] = model.NewBoolVar(f"x_{t}_{b}")

    # Each train assigned to at most one bay
    for t in T:
        model.Add(sum(x[(t, b)] for b in B) <= 1)

    # Bay capacity constraints
    for b in B:
        cap = int(bays[b]['max_capacity'])
        model.Add(sum(x[(t, b)] for t in T) <= cap)

    # Assign up to min(total_capacity, num_trains)
    total_capacity = sum(int(b['max_capacity']) for b in bays)
    assign_limit = min(total_capacity, len(trains))
    model.Add(sum(x[(t, b)] for t in T for b in B) == assign_limit)

    # Objective: minimize weighted geometry mismatch + weighted shunting cost
    costs = {}
    for t in T:
        tgeom = float(trains[t]['bay_geometry_score'])
        for b in B:
            bgeom = float(bays[b]['geometry_score'])
            # Scale to integer
            geom_cost = int(abs(bgeom - tgeom) * 1000)
            sh_cost = shunt_costs[(t, b)]
            cost = int(geometry_weight * geom_cost + shunting_weight * sh_cost)
            costs[(t, b)] = cost
    model.Minimize(sum(costs[(t, b)] * x[(t, b)] for t in T for b in B))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    assigned = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for t in T:
            assigned_bay = None
            bmatch = None
            for b in B:
                if solver.Value(x[(t, b)]) == 1:
                    assigned_bay = bays[b]['bay_id']
                    bmatch = float(bays[b]['geometry_score'])
                    break
            tt = trains[t].copy()
            if assigned_bay:
                tt['assigned_bay'] = assigned_bay
                tt['bay_geometry_match'] = bmatch
            assigned.append(tt)
    else:
        # Fallback: return unassigned list
        assigned = [t.copy() for t in trains]
    return assigned


