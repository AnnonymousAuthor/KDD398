import numpy as np
import pandas as pd
import random
import argparse

import matplotlib.pyplot as plt
import scipy.stats as stat
from datetime import datetime

import time
import os
import sys
from scipy.stats import norm

from cco import SyntheticEnv
from cco import default_seeds
from cco import solve_offline_sbpp_ensemble

import logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format = fmt)
logger = logging.getLogger('exp-ensemble-{datetime}'.format(datetime=time.time()))
logger.setLevel(logging.INFO) # enable debug when necessary

configs = {'convergence':True, 'num_ensembles':8, 'max_depth':20, 
           'min_num_ensembles':1,
           'max_throttles': 3, 'num_cols':5,
           'max_dual_throttles': 3, 'dual_obj_threshold':0.01,
	   'dual_time_limit':100.0, 'csp_time_limit':500.0, 
           'selection_times':0}

def set_logger(args):
    if not os.path.exists('./logs-synthetic'):
        os.mkdir('./logs-synthetic')
    if not os.path.exists('./patterns'):
        os.mkdir('./patterns')
    if not os.path.exists('./Figure'):
        os.mkdir('./Figure')
    if not os.path.exists('./time'):
        os.mkdir('./time')
    if not os.path.exists('./price'):
        os.mkdir('./price')
    fn = f"log_{args.num_services}_{args.alpha}.txt"
    handler = logging.FileHandler("./logs-synthetic/" + fn)
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="verbosity")
    parser.add_argument("-o", "--output_path", type=str, default='./results-synthetic/',
                        help="output file path of traces")
    parser.add_argument("-n", "--num_services", type=int, default=100,
                        help="number of services to simulate")
    parser.add_argument("--alpha", type=float, default=0.999,
                        help="confidence level")
    parser.add_argument("--alpha1", type=float, default=0.999,
                        help="confidence level used for compute ucac")
    parser.add_argument('-d', "--depth", type=int, default=20, 
                        help="the max depth of a single tree")
    parser.add_argument('-e', "--ensembles",type=int,default=10, 
                        help="the number of ensembles")
    parser.add_argument('-t', "--throttles",type=int,default=3, 
                        help="the numbers of throttles to escape")
    parser.add_argument("--dual_throttles",type=int,default=4, 
                        help="the numbers of dual throttles to escape")
    parser.add_argument("--dual_obj_threshold",type=float,default=0.01, 
                        help="the numbers of dual throttles to escape")
    parser.add_argument("-c", "--columns", type=int,default=3,
                        help="the max number of columns returned each dual")
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="the index of test days")
    parser.add_argument("--dual_time_limit", type=float, default=100.0, 
                        help="dual time limit")
    parser.add_argument("--max_dual_time_limit", type=float, default=100.0,
                        help="max dual time limit")
    parser.add_argument("--csp_time_limit", type=float, default=500.0, 
                        help="csp time limit")
    parser.add_argument("--convergence",action='store_true',default=False, 
                        help="sampling based method converge")
    parser.add_argument("--selection_times",type=int,default=6,
                        help="number of patterns selected")  
    parser.add_argument("--experiment_setting",type=str, default='linear_extention',
                        help="the experiment setting")
    args = parser.parse_args()
    return args 

def plot_figure(x,y,z,ylabel,zlabel,x_label,y_index,name):
    plt.figure()
    plt.plot(x,y,'b*--',alpha=1.0, linewidth=1, label=ylabel)
    plt.plot(x,z,'r*--',alpha=1.0, linewidth=1, label=zlabel)
    plt.legend()  
    plt.xlabel(x_label) 
    plt.ylabel(y_index)
    plt.title(name)
    plt.savefig("./Figure/"+name+".png",dpi=500)
    plt.close()

def make_metrics_row(seed, kv, lower_bound, gap, abs_gap, patterns_num, selecetd_patterns_num, sub_time, integer_time, total_time):
    row = ['csp nodes', seed,  
           kv['num_used_nodes'],
           lower_bound,
           gap,
           abs_gap,
           patterns_num,
           selecetd_patterns_num,
           sub_time,
           integer_time,
           total_time]
    return row

def set_output_path(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        logger.info(f'the directory {args.output_path} not exist, so create one.')

def main(args):
    set_logger(args)
    set_output_path(args)
    timestamp = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")

    alpha = args.alpha
    alpha1 = args.alpha1
    dalpha = norm.ppf(alpha, loc=0, scale=1)
    dalpha1 = norm.ppf(alpha1, loc=0, scale=1)
    num_services = args.num_services
    experiment_setting = args.experiment_setting

    configs['convergence'] = args.convergence
    configs['num_ensembles'] = args.ensembles
    configs['max_depth'] = args.depth
    configs['max_throttles'] = args.throttles
    configs['max_dual_throttles'] = args.dual_throttles
    configs['dual_obj_threshold'] = args.dual_obj_threshold
    configs['dual_time_limit'] = args.dual_time_limit
    configs['csp_time_limit'] = args.csp_time_limit
    configs['num_cols'] = args.columns
    configs['selection_times'] = args.selection_times

    fpre = f'{experiment_setting}_converge-{args.convergence}_{num_services}_{args.ensembles}_{args.depth}_{args.throttles}_{args.columns}_{args.dual_time_limit}_{args.dual_throttles}_{args.dual_obj_threshold}'
    exp_metrics = []

    for i, seed in enumerate(default_seeds):
        logger.info(f"Test env with random seed {seed} and num_services {num_services}")

        # create a synthetic env
        env = SyntheticEnv(num_services=num_services, alpha=alpha, seed=seed, experiment_setting=experiment_setting)
        means, stds = env.get_service_means_stds()
        capacity = env.get_node_capacity()
        logger.info("Means: " + ",".join(map(str, list(means))))
        logger.info("Stds: " + ",".join(map(str, list(stds))))
        logger.info("Demands:" + ",".join(map(str, env.get_demands(inc_ratio=1.0))))
        logger.info("Created the env.")

        #solve the sbpp-nodes problem
        t_start = time.time()
        solution, patterns, selected_patterns, x, dual_probelm_time, sub_problem_time, integer_time, price, reduced_cost, dlpm = solve_offline_sbpp_ensemble(env.layout, means, stds**2,
                                            env.get_demands(inc_ratio=1.0), 
                                            capacity, dalpha, patterns=None, kwargs=configs) 
        t_end = time.time()

        layout, metrics = env.allocate(solution)
        logger.info('num_used_nodes: {val}'.format(val = np.round(metrics['num_used_nodes'])))

        gap = np.round(((metrics['num_used_nodes'] - dlpm[-1])/dlpm[-1])*100,2)
        abs_gap = np.round((metrics['num_used_nodes'] - dlpm[-1]),2)
        exp_metrics.append(make_metrics_row(seed, metrics, lower_bound=np.round(dlpm[-1],2), gap=gap, abs_gap=abs_gap, patterns_num=patterns.shape[1], selecetd_patterns_num=selected_patterns.shape[1], sub_time=np.round(sum(sub_problem_time),2),
                            integer_time=np.round(integer_time,2),total_time = np.round((t_end-t_start),2)))

        #dump metrics and results
        pd.DataFrame(patterns).to_csv(f"./patterns/{fpre}_{seed}_patterns.csv")
        pd.DataFrame(selected_patterns).to_csv(f"./patterns/{fpre}_{seed}_selected_patterns.csv")
        pd.DataFrame(x).to_csv(f"./patterns/{fpre}_{seed}_solutions.csv")
        metrics_columns = ['alg', 'seed', 'num_used_nodes', 'lower bound', 'gap', 'abs_gap', 'patterns','seleted patterns','sub time','integer time','total time']
        pd.DataFrame(columns=metrics_columns, data=exp_metrics).to_csv(args.output_path + f'{fpre}_{timestamp}.csv', index=False)
        pd.DataFrame({"dual_probelm_time":dual_probelm_time,"sub_problem_time":sub_problem_time}).to_csv(f"./time/{fpre}_{seed}.csv")
        pd.DataFrame(np.array(price)).to_csv(f"./price/{fpre}_{seed}_price.csv")
        pd.DataFrame(reduced_cost).to_csv(f"./price/{fpre}_{seed}_reduced_cost.csv")
        pd.DataFrame(dlpm).to_csv(f"./price/{fpre}_{seed}_dual_obj.csv")

        #draw figure
        plot_figure(x=[i for i in range(len(dual_probelm_time))],y=dual_probelm_time,z=sub_problem_time,
            ylabel="dual problem",zlabel="sub probelm",x_label="pattern index",y_index="conputation time",name=f"{fpre}_{seed}")

if __name__ == '__main__':
    args = parse()
    sys.exit(main(args))
