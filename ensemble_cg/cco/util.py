import numpy as np
import pandas as pd

def compute_node_ucac(means, variances, dalpha, node): 
    ucac = node @ means + dalpha * np.sqrt(node @ variances)
    return ucac

def compute_cluster_ucac(means, variances, dalpha, cluster_layout):
    cluster_layout_values = cluster_layout.values
    u = cluster_layout_values @ means + dalpha * np.sqrt(cluster_layout_values @ variances)
    return np.array(u) 

def compute_solution_ucac(means, variances, demands, dalpha, solution, num_nodes): 
    num_services = len(means)
    cluster_layout = pd.DataFrame(np.zeros((num_nodes, num_services)))
    cnt = [0] * num_services
    node_idx = 0
    for ms, n in solution:
        for i in range(int(n)):  
            for j, m in enumerate(ms):
                if cnt[j] < demands[j]:
                    items = min(m, demands[j] - cnt[j])
                    cluster_layout.iloc[node_idx, j] += items
                    cnt[j] += items
            node_idx += 1

    nodes_ucac = compute_cluster_ucac(means, variances, dalpha, cluster_layout) 
    ucac = sum(nodes_ucac)
            
    return ucac, nodes_ucac


def compute_pattern_ucac(means, variances, dalpha, patterns): 
    u = []
    num_patterns = patterns.shape[1] 
    for i in range(num_patterns):
        pattern = patterns[:, i]
        ucac = pattern @ means + dalpha * np.sqrt(pattern @ variances)
        ucac = np.round(ucac, 2)
        u.append(ucac)
    return np.array(u)

def solution_to_metrics(cluster_layout, means, variances, dalpha, solution):
    new_cluster_layout = pd.DataFrame(cluster_layout.values + solution)
    bin = ((new_cluster_layout.sum(axis=1)) > 0).sum()
    node_usages = compute_cluster_ucac(means, variances, dalpha, new_cluster_layout)
    ucac = sum(node_usages)
    return bin, node_usages, ucac
