import numpy as np
import random

# TODO: delete.
def generate_random_distributions(n, trace_means, trace_stds, trace_demands):
    demands = []
    means = []
    variances = []
    for i in range(n):
        idx = np.random.choice(len(trace_means))
        # mean = max(0.01, trace_means[idx] * (0.8 + 0.4 * random.random()))
        mean = trace_means[idx]
        std = max(0.1, trace_stds[idx] * (0.8 + 0.4 * random.random()))
        demand = max(1, trace_demands[idx] * (0.8 + 0.4 * random.random()))
        means.append(mean)
        variances.append(std ** 2)
        demands.append(int(demand))
        
    means = np.round(means, 2)
    variances = np.round(variances, 2)   
    demands = np.array(demands)
    return means, variances, demands


def generate_instances(mean, var, n, lower=0.01, upper=16):
    instances = []
    for i in range(n):
        instance = max(lower, np.random.normal(mean, np.sqrt(var)))
        instance = min(instance, upper)
        instances.append(instance)
    return np.array(instances)


def generate_instances_list(means, variances, demands, lowers, uppers, seed=42):
    #np.random.seed(seed)
    instances_list = []
    for j in range(len(means)):
        instances_list.append(generate_instances(means[j], variances[j], demands[j], 
                                                 lowers[j], uppers[j]))
    return instances_list
