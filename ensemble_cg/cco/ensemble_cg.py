import numpy as np
import pandas as pd
import time

import gurobipy as grb
from gurobipy import GRB

from .util import compute_node_ucac, compute_cluster_ucac, compute_solution_ucac, compute_pattern_ucac
from .meta_params import default_gurobi_threads 

import logging
import random
import copy

def patterns_initialize(means, variances, capacity, dalpha):
    vec1 = np.floor(((np.sqrt(dalpha**2 * variances + 4.0*capacity[0]*means) 
                      - dalpha*np.sqrt(variances))/(2* means))**2)
    patterns = np.diag(vec1)
    return patterns

def solve_dual(patterns, demands, kwargs): 
    m = grb.Model()
    m.setParam('OutputFlag', False)
    m.setParam('Threads', kwargs.get('num_threads', default_gurobi_threads))
    y = m.addMVar(len(demands), lb=0)
    m.addConstr(patterns.transpose() @ y <= 1)
    m.setObjective(demands @ y, GRB.MAXIMIZE)
    m.optimize()
    return y.X, m.objVal

def solve_sub(means, variances, capacity, dalpha, y, kwargs):
    def is_deterministic(variances):
        return np.max(variances) < 1e-5

    m = grb.Model()
    m.setParam('OutputFlag', True)
    m.setParam('Threads', kwargs.get('num_threads', default_gurobi_threads))
    m.setParam('NonConvex',2)
    m.setParam("TimeLimit", kwargs.get('dual_time_limit', 100.0))
    m.setParam("IntFeasTol",1e-9)
    m.setParam('IntegralityFocus',1)
    if kwargs.get('num_cols', 1) > 1:
        m.setParam(GRB.Param.PoolSolutions, kwargs['num_cols'])
        m.setParam(GRB.Param.PoolGap, 0.10)
        m.setParam(GRB.Param.PoolSearchMode, 2)
    pat = m.addMVar(len(means), lb=0, vtype=GRB.INTEGER)
    r = m.addVar(lb=0)
    p = pat.tolist()
    n = len(means)
    if not is_deterministic(variances):
        m.addConstr(dalpha**2 * grb.quicksum(p[i] * variances[i] for i in range(n)) <= r * r)
        m.addConstr(r <= capacity[0] - grb.quicksum(p[i] * means[i] for i in range(n)))
    else:
        m.addConstr(grb.quicksum(p[i] * means[i] for i in range(n)) <= capacity[0])
    m.setObjective(1 - y @ pat)  
    m.optimize()  

    patterns_set = []
    if kwargs.get('num_cols', 1) > 1:
        nSolutions = m.SolCount
        for i in range(nSolutions):
            m.setParam(GRB.Param.SolutionNumber, i)
            if m.PoolObjVal < -1e-7:
                patterns_set.append(pat.Xn)
    else:
        if m.objVal < -1e-7:
            patterns_set.append(pat.X)

    new_patterns = np.empty(shape=(len(means),0))
    if len(patterns_set) > 0:
        patterns = np.array(patterns_set).T
        patterns = np.round(patterns,0)
        patterns = np.unique(patterns, axis=1)
        new_patterns = np.c_[new_patterns, patterns]
    return m.objVal, new_patterns

def is_global_converged(means, variances, capacity, dalpha, patterns, demands, kwargs):
    def is_deterministic(variances):
        return np.max(variances) < 1e-5

    y, _ = solve_dual(patterns, demands, kwargs)

    m = grb.Model()
    m.setParam('OutputFlag', True)
    m.setParam('Threads', kwargs.get('num_threads', default_gurobi_threads))
    m.setParam('NonConvex',2)
    m.setParam("TimeLimit", kwargs.get('dual_time_limit', 100.0))
    m.setParam("IntFeasTol",1e-9)
    m.setParam('IntegralityFocus',1)
    pat = m.addMVar(len(means), lb=0, vtype=GRB.INTEGER)
    r = m.addVar(lb=0)
    p = pat.tolist()
    n = len(means)
    if not is_deterministic(variances):
        m.addConstr(dalpha**2 * grb.quicksum(p[i] * variances[i] for i in range(n)) <= r * r)
        m.addConstr(r <= capacity[0] - grb.quicksum(p[i] * means[i] for i in range(n)))
    else:
        m.addConstr(grb.quicksum(p[i] * means[i] for i in range(n)) <= capacity[0])
    m.setObjective(1 - y @ pat)  
    m.optimize()  

    return m.objVal >= -1e-7


def patterns_generation(means, variances, capacity, dalpha, demands, initial_patterns, 
                        kwargs):
    max_throttles = kwargs.get('max_throttles', 5)
    max_depth = kwargs.get('max_depth', 20)
    dual_time_limit = kwargs.get('dual_time_limit', 100.0)
    max_dual_throttles = kwargs.get('max_dual_throttles',3)
    obj_threshold = kwargs.get('dual_obj_threshold',0.01)
    
    k = len(means)

    # check initial patterns
    if initial_patterns is None:
        patterns = patterns_initialize(means, variances, capacity, dalpha)
    else:
        patterns = initial_patterns   

    # solve
    l = int(min(max_throttles*1.5, max_depth))
    dl = int(min(max_dual_throttles*1.5, max_depth))
    throttles = [0] * l
    dual_throttles = [0] * dl
    cost = -np.inf
    th = -1e-7
    dual_time = []
    sub_time = [] 
    pi = []
    re_cost = [] 
    dual_obj = []
    old_dlmp = 0

    for d in range(max_depth):
        t0 = time.time()
        y, dlmp = solve_dual(patterns, demands, kwargs)
        t1 = time.time()
        dual_time.append(t1-t0)
        pi.append(np.array(y))
        delta_dlmp = np.abs(dlmp - old_dlmp)
        dual_obj.append(dlmp)
        old_dlmp = dlmp

        t0 = time.time()
        cost, new_patterns = solve_sub(means, variances, capacity, dalpha, y, kwargs)
        t1 = time.time()
        throttles[d % l] = t1-t0 >= dual_time_limit
        dual_throttles[d % dl] = delta_dlmp <= obj_threshold
        sub_time.append(t1-t0)
        re_cost.append(cost)

        if cost < th:
            # filter new patterns
            useful_patterns = np.empty(shape=(k,0))
            for i in range(new_patterns.shape[1]):
                pat = new_patterns[:,i]
                if sum(np.all(patterns >= pat.reshape(k,1),axis = 0)) == 0:
                    useful_patterns = np.c_[useful_patterns, pat]
            if useful_patterns.shape[1] > 0:
                patterns = np.c_[patterns, useful_patterns]

        # escape when reach the threshold of throttles
        if np.sum(throttles) >= max_throttles or cost >= th: break
        if kwargs.get('convergence', True) and np.sum(dual_throttles) >= max_dual_throttles: break
    _, dlmp = solve_dual(patterns, demands, kwargs)
    dual_obj.append(dlmp)

    return np.array(np.round(patterns,0)), dual_time, sub_time, pi, re_cost, dual_obj

def pattern_selection(patterns, demands, kwargs):
    n = len(patterns[:,0])
    original_patterns = copy.deepcopy(patterns)
    selected_patterns = np.empty(shape=(n,0))
    for i in range(kwargs.get('selection_times', 1)):
        #check pattern 
        legal_check = np.sum(patterns,axis=1)
        auxiliary_idx = (np.where(legal_check==0)[0]).tolist()
        deleted_idx = []
        if len(auxiliary_idx):
            auxiliary_patterns = original_patterns[:,auxiliary_idx]
            idx_start = patterns.shape[1]
            patterns = np.c_[patterns,auxiliary_patterns]
            deleted_idx.extend([i for i in range(idx_start,patterns.shape[1])])
        m = grb.Model()
        m.setParam('OutputFlag', False)
        m.setParam('Threads', kwargs.get('num_threads', default_gurobi_threads))
        x = m.addMVar(shape=patterns.shape[1], lb=0, vtype=GRB.CONTINUOUS)
        m.addMConstr(patterns, x, '>=', demands)
        m.setObjective(x.sum())
        m.optimize()
        optimal_solution = np.array(m.getAttr('X'))
        pattern_idx = np.where(optimal_solution > 0)[0].tolist()
        pattern_idx = list(set(pattern_idx) - set(deleted_idx))
        residual_idx = np.where(optimal_solution <= 0)[0].tolist()
        selected_patterns = np.c_[selected_patterns,patterns[:,pattern_idx]]
        patterns = patterns[:,residual_idx]  
        if len(residual_idx)==0:
            break
    return selected_patterns

def solve_cutting_stock(patterns, demands, kwargs):
    csp_time_limit = kwargs.get('csp_time_limit', 500.0)

    m = grb.Model()
    m.setParam('OutputFlag', True)
    m.setParam('Threads', kwargs.get('num_threads', default_gurobi_threads))
    m.setParam('TimeLimit', csp_time_limit) 
    x = m.addMVar(shape=patterns.shape[1], lb=0, vtype=GRB.INTEGER)
    m.addMConstr(patterns, x, '>=', demands)
    m.setObjective(x.sum())
    m.optimize()
    bins = m.objVal
    return bins, np.array(m.getAttr('X'))

def sample_patterns(patterns, n):
    k = patterns.shape[0]
    indices = set()
    for i in range(k):
        pops = set(np.where(patterns[i, :]>0)[0])
        if len(pops.intersection(indices)) == 0:
            indices.add(random.sample(pops, k=1)[0])
    if len(indices) < n:
        pops = set(range(patterns.shape[1])) - indices
        indices.update(random.sample(pops, k=n-len(indices)))
    return patterns[:, list(indices)].copy()

def clean_patterns(new_patterns, patterns):
    k = patterns.shape[0]
    for i in range(new_patterns.shape[1]):
        pat = new_patterns[:,i]
        if sum(np.all(patterns >= pat.reshape(k,1),axis = 0)) == 0:
            patterns = np.c_[patterns, pat]
    return patterns

def patterns_ensemble(means, variances, demands, capacity, dalpha, initial_patterns, 
                      kwargs):
    def is_deterministic(variances):
        return np.max(variances) < 1e-5

    def is_converged(costs):
        return np.max(costs) >= -1e-7


    if initial_patterns is None:
        patterns = patterns_initialize(means, variances, capacity, dalpha)
    else:
        patterns = initial_patterns
       
    k = len(means)
    dual_probelm_time = []
    sub_problem_time = []
    price = []
    reduced_cost = []
    dlpm = []
    old_dlpm = np.inf

    #ensemble
    converged = False
    min_num_ensembles = kwargs.get('min_num_ensembles', 1)
    preference = kwargs.get('ensemble_preference', 'none') #[none, global_converge, local_converge]
    for eid in range(kwargs.get('num_ensembles', 8)):
        # sample base patterns
        base_patterns = sample_patterns(patterns, k)
        # extend via column generation
        new_patterns, dual_time, sub_time, pi, re_cost, dual_obj = patterns_generation(means, variances, 
                capacity, dalpha, demands, base_patterns, kwargs)
        dual_probelm_time.extend(dual_time)
        sub_problem_time.extend(sub_time)
        price.extend(pi)
        reduced_cost.extend(re_cost)
        dlpm.extend(dual_obj)
        # add to pool
        patterns = clean_patterns(new_patterns, patterns)

        if is_converged(re_cost): converged = True
        if eid+1 >= min_num_ensembles:
            if preference=='none': continue
            if preference=='global_converge' and not converged:
                converged = is_global_converged(means, variances, capacity, dalpha, patterns, demands, 
                                                kwargs)
            if converged: break;
 
    #converge
    if kwargs.get('convergence', True) and not converged:
        kwargs_converge = copy.deepcopy(kwargs)
        kwargs_converge['max_depth'] = 1000
        kwargs_converge['max_throttles'] = 1000
        kwargs_converge['dual_time_limit'] = kwargs.get('max_dual_time_limit', 100.0)
        new_patterns, dual_time, sub_time, pi, re_cost, dual_obj = patterns_generation(means, variances, 
	    capacity, dalpha, demands, patterns, kwargs_converge)
        dual_probelm_time.extend(dual_time)
        sub_problem_time.extend(sub_time)
        price.extend(pi)
        reduced_cost.extend(re_cost)
        dlpm.extend(dual_obj)
        patterns = clean_patterns(new_patterns, patterns)

    return patterns, dual_probelm_time, sub_problem_time, price, reduced_cost, dlpm

def solve_sbpp_empty(cluster_layout, means, variances, demands, capacity, dalpha, initial_patterns, 
                     kwargs):
    patterns, dual_probelm_time, sub_problem_time, price, reduced_cost, dlpm = patterns_ensemble(means, variances, 
            demands, capacity, dalpha, initial_patterns, kwargs)

    if kwargs.get('selection_times', 0) > 0:
        selected_patterns = pattern_selection(patterns, demands, kwargs)
    else:
        selected_patterns = patterns
    t0 = time.time()
    num_bins, x = solve_cutting_stock(selected_patterns, demands, kwargs)
    t1 = time.time()
    integer_time = t1-t0    
    solution = recover_solution_empty(cluster_layout, selected_patterns, x)
    solution = tune_solution(solution, cluster_layout, means, variances, demands, dalpha)
    return solution, patterns, selected_patterns, x, dual_probelm_time, sub_problem_time, integer_time, price, reduced_cost, dlpm

def recover_solution_empty(cluster_layout, patterns, x):
    cluster_states = cluster_layout.values
    solution = np.zeros(cluster_states.shape)
    x = x.copy()
   
    nz_pids = np.where(x > 0)[0] 
    
    node_pids = []
    for i in nz_pids:
        node_pids = node_pids + [i]*round((x[i])) 
    for i, pid in enumerate(node_pids):
        pattern = patterns[:, pid] 
        solution[i, :] = pattern - cluster_states[i, :]
    
    return solution 

def tune_solution(solution, cluster_layout, means, variances, demands, dalpha):
    """ solution may over-provisioning beyond the demands, thus we tune it.
    """
    cluster_states = cluster_layout.values.copy() #may change, so copy one.
    cluster_states = cluster_states[:, :] + solution 
    over_pros = np.sum(solution, axis=0) - demands
    for j in range(len(over_pros)): #per service
        if over_pros[j] > 0:
            nodes = np.nonzero(solution[:, j])[0]
            ucacs = [compute_node_ucac(means, variances, dalpha, cluster_states[node, :]) 
                     for node in nodes]
            sorted_indices = np.argsort(ucacs) 
            for idx in sorted_indices:
                node = nodes[idx]
                if solution[node, j] >= over_pros[j]:
                    solution[node, j] -= over_pros[j]
                    cluster_states[node, j] -= over_pros[j]
                    over_pros[j] = 0
                    break
                else:
                    over_pros[j] -= solution[node, j]
                    cluster_states[node, j] -= solution[node, j]
                    solution[node, j] = 0       
    return solution

# wrappers
def solve_offline_sbpp_ensemble(cluster_layout, means, variances, demands, capacity, dalpha, patterns, kwargs):
    return solve_sbpp_empty(cluster_layout, means, variances,  demands, capacity, dalpha, patterns, kwargs)
