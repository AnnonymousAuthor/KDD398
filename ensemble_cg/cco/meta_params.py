default_seeds = [i for i in range(10,60)]
# default_seeds = [13, 17, 29, 51, 57, 63, 67, 79, 81, 97] #our used random seeds in experiments
import multiprocessing
default_gurobi_threads = multiprocessing.cpu_count()-1
