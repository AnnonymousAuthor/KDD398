import numpy as np
import pandas as pd
import random
import os

from scipy.stats import norm
from .util import compute_node_ucac, compute_cluster_ucac
from .generate_random import generate_instances_list, generate_instances

class SyntheticEnv:
    num_nodes = 4000

    def __init__(self, num_services=5, alpha=0.999, seed=42, experiment_setting='nonlinear_real'):
        """docstring
        """
        random.seed(seed)
        np.random.seed(seed)
        if experiment_setting == 'nonlinear_real':
            base_means = np.array([6.18, 2.47, 1.07, 4.12, 1.06, 0.73, 1.94, 2.48, 2.42, 2.49, 
                           0.97, 2.46, 2.52, 1.06, 2.59, 1.96, 3.33])            
            base_stds = np.array([1.73, 0.47, 0.43, 2.69, 0.85, 0.19, 0.9 , 0.82, 0.97, 0.62, 
                          0.31, 0.62, 0.84, 0.57, 0.7 , 0.55, 0.9 ])
            base_nums = np.array([270, 55, 1618, 904, 576, 1085, 1035, 118, 1450, 313, 44, 544, 
                          697, 427, 363, 360, 701])
            base_vcores = np.array([16, 8, 4, 16, 4, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8])  
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_real_ext1':
            base_means = np.array([6.18, 2.47, 1.07, 4.12, 1.06, 0.73, 1.94, 2.48, 2.42, 2.49, 
                           0.97, 2.46, 2.52, 1.06, 2.59, 1.96, 3.33, 2.45, 2.82, 1.9, 2.41, 2.5, 1.88, 2.12, 2.27])    
            base_stds = np.array([1.73, 0.47, 0.43, 2.69, 0.85, 0.19, 0.9 , 0.82, 0.97, 0.62, 
                          0.31, 0.62, 0.84, 0.57, 0.7 , 0.55, 0.9, 0.92, 0.97, 0.67, 0.8, 0.96, 0.79, 0.75, 0.91])
            base_nums = np.array([270, 55, 1618, 904, 576, 1085, 1035, 118, 1450, 313, 44, 544, 
                          697, 427, 363, 360, 701, 468, 578, 532, 453, 558, 683, 567, 676])
            base_vcores = np.array([16, 8, 4, 16, 4, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8])  
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_real_ext': 
            base_means = np.array( [6.18, 2.47, 1.07, 4.12, 1.06, 0.73, 1.94, 2.48, 2.42, 2.49, 0.97, 
                                    2.46, 2.52, 1.06, 2.59, 1.96, 3.33, 1.86, 1.88, 4.55, 0.52, 1.11, 3.26, 0.91, 2.47] )
            base_stds = np.array( [1.73, 0.47, 0.43, 2.69, 0.85, 0.19, 0.9, 0.82, 0.97, 0.62, 0.31, 
                                   0.62, 0.84, 0.57, 0.7, 0.55, 0.9, 0.68, 0.99, 2.88, 0.42, 0.91, 1.08, 0.34, 0.65] )
            base_nums = np.array( [270, 55, 1618, 904, 576, 1085, 1035, 118, 1450, 313, 44, 544, 697, 
                                   427, 363, 360, 701, 808, 952, 953, 1082, 611, 165, 1356, 83] )
            base_vcores = np.array( [16, 8, 4, 16, 4, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8, 4, 8, 16, 
                                     2, 4, 8, 2, 8] )
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_real_ext2':
            base_means = np.array([[6.18, 2.47, 1.07, 4.12, 1.06, 0.73, 1.94, 2.48, 2.42, 2.49, 0.97, 2.46, 
                                    2.52, 1.06, 2.59, 1.96, 3.33, 5.31, 1.66, 2.16, 4.91, 1.06, 1.22, 1.72, 4.03]])            
            base_stds = np.array([1.73, 0.47, 0.43, 2.69, 0.85, 0.19, 0.9 , 0.82, 0.97, 0.62, 0.31, 0.62, 
                                  0.84, 0.57, 0.7 , 0.55, 0.9 , 1.59, 0.78, 0.43, 1.53, 0.95, 1.21, 1.27, 1.43])
            base_nums = np.array([270,   55, 1618,  904,  576, 1085, 1035,  118, 1450,  313,   44, 544,  697,  
                                  427,  363,  360,  701,  647, 1256,  353,  553,  607, 996, 1021,  407])
            base_vcores = np.array([16,  8,  4, 16,  4,  4,  8,  8,  8,  8,  4,  8,  8,  4,  8,  8,  8, 
                                    16,  4,  4, 16,  4,  8,  8, 16])
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_synthetic':
            real_means = np.array([6.18, 2.47, 1.07, 4.12, 1.06, 0.73, 1.94, 2.48, 2.42, 2.49, 
                           0.97, 2.46, 2.52, 1.06, 2.59, 1.96, 3.33])     
            real_stds = np.array([1.73, 0.47, 0.43, 2.69, 0.85, 0.19, 0.9 , 0.82, 0.97, 0.62, 
                          0.31, 0.62, 0.84, 0.57, 0.7 , 0.55, 0.9 ])  
            real_variable_coeff = np.round(real_means/real_stds,2)
            base_means = np.round(np.random.beta(2, 10, 20)*16,2)
            base_variable_coeff = np.array(random.choices(real_variable_coeff,k=20))
            base_stds = np.round(base_means/base_variable_coeff,2) 
            base_nums = np.round(np.random.beta(2, 6, 20) * 1500,0)
            base_vcores = np.ones_like(base_means)*16
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_easy':
            base_means = np.round(np.random.beta(2, 2, 50)*16,2) 
            base_stds = np.round(np.random.beta(2, 6, 50)*2,2) 
            base_nums = np.round(np.random.beta(2, 8, 50) * 500,0)
            base_vcores = np.ones_like(base_means)*20
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'nonlinear_easy_2':
            base_means = np.round(np.random.uniform(0.1, 10, 50),2) 
            base_stds = np.round(np.random.uniform(0.1, 2, 50),2) 
            base_nums = np.round(np.random.beta(2, 8, 50) * 500,0)
            base_vcores = np.ones_like(base_means)*20
            self.node_capacity = [31.58, 256*1024]
        elif experiment_setting == 'linear_extention': 
            base_means = np.random.randint(1,150,100) 
            base_stds = np.zeros_like(base_means)
            base_nums = np.random.randint(5,100,size=100) 
            base_vcores = np.ones_like(base_means)*150
            self.node_capacity = [299.58, 256*1024]
        elif experiment_setting == 'linear_synthetic_set1':
            base_means = np.random.randint(1,40,size=30)
            base_stds = np.zeros_like(base_means)
            base_nums = np.random.randint(50,200,size=30)
            base_vcores = np.ones_like(base_means)*40
            self.node_capacity = [63.58, 256*1024]
        elif experiment_setting == 'linear_synthetic_set2':
            base_means = np.random.randint(1,70,size=60)
            base_stds = np.zeros_like(base_means)
            base_nums = np.random.randint(50,200,size=60)
            base_vcores = np.ones_like(base_means)*70
            self.node_capacity = [127.58, 256*1024]

        self._dalpha = norm.ppf(alpha, loc=0, scale=1)
        self._num_base = len(base_means)
        self._num_services = num_services
        self._layout = pd.DataFrame(np.zeros((self.num_nodes, self._num_services), dtype=int))
        
        indices = random.sample(range(self._num_base), k=min(num_services, self._num_base))
        if num_services > self._num_base:
            num_left = num_services-self._num_base
            while num_left > 0:
                indices = indices + random.sample(range(self._num_base), k=min(num_left, self._num_base))
                num_left -= self._num_base
        self._means = base_means[indices]
        self._stds = base_stds[indices] * [random.uniform(0.9, 1.1) for i in range(num_services)]
        self._stds = np.round(self._stds, 2)
        self._vcores = base_vcores[indices]
        
        ratio = np.sum(base_means * base_nums) / np.sum(base_nums[indices]*self._means)
        self._default_max_nums = (base_nums[indices] * ratio).astype(int) # recale according to mean cores make sure that nodes num less than 4000
       
        self._instances = None        
    
    def reset(self, seed=42):
        """ reset to empty nodes
        """
        random.seed(seed)
        self._instances = None
        for col in self._layout.columns: self._layout[col].values[:] = 0
        return self._layout
               
    def allocate(self, solution):
        """ execute the bin packing solution and update
        """
        solution = np.array(solution).astype(int)
        self._layout.loc[:,:] = self._layout.values + solution
        
        # add instances
        self._instances = generate_instances_list(self._means, self._stds**2, 
                                                  np.sum(self._layout.values, axis=0),
                                                  lowers=[0.01]*self._num_services, uppers=self._vcores)
        
        metrics = self._get_metrics()
        return self._layout, metrics
    
    def get_demands(self, inc_ratio = 1.0):
        demands = self._default_max_nums * inc_ratio - np.sum(self._layout.values, axis=0)
        return demands.astype(int)
       
    def get_metrics(self):
        """metrics: UCaC, #Nodes, Violations
        """
        return self._get_metrics()
    
    def get_node_usages(self):
        return self._get_node_usages()
    
    def get_service_means_stds(self):
        return self._means, self._stds
    
    def get_node_capacity(self):
        return self.node_capacity
    
    @property
    def layout(self):
        return self._layout
       
    
    def _get_metrics(self):
        num_used_nodes = np.sum(np.sum(self._layout, axis=1) > 0)
        
        ucac = compute_cluster_ucac(self._means, self._stds**2, self._dalpha, self._layout)
        
        node_usages = self._get_node_usages()      
        num_violations = np.sum(node_usages > self.node_capacity[0])

        return {'num_used_nodes':num_used_nodes, 'ucac':ucac, 'num_violations':num_violations}
    
    def _get_node_usages(self):
        node_usages = np.zeros(self.num_nodes) #[0]*num_nodes
        for j in range(self._num_services):
            container_usages = self._instances[j]
            allocs = self._layout[j].values.astype(int)
            nzidxs = np.nonzero(allocs)[0]
            base = 0
            for idx in nzidxs:
                node_usages[idx] += np.sum(container_usages[base : base+allocs[idx]])
                base += allocs[idx]                
        return node_usages
