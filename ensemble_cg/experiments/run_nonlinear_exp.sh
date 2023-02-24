# Baseline
python run_synthetic30.py -n 10 -d 200 -e 1 -t 200 -c 1 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;
python run_synthetic30.py -n 15 -d 200 -e 1 -t 200 -c 1 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;
python run_synthetic30.py -n 20 -d 200 -e 1 -t 200 -c 1 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;

# MC3
python run_synthetic30.py -n 10 -d 200 -e 1 -t 200 -c 3 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;
python run_synthetic30.py -n 15 -d 200 -e 1 -t 200 -c 3 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;
python run_synthetic30.py -n 20 -d 200 -e 1 -t 200 -c 3 --dual_time_limit 1000  --csp_time_limit 1000 --experiment_setting nonlinear_real_ext --dual_throttles 100;

# MP4
python run_synthetic30.py -n 10 -d 10 -e 4 -t 1 -c 1 --dual_time_limit 30  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;
python run_synthetic30.py -n 15 -d 15 -e 4 -t 1 -c 1 --dual_time_limit 50  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;
python run_synthetic30.py -n 20 -d 15 -e 4 -t 1 -c 1 --dual_time_limit 60  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;

# MP4MC3
python run_synthetic30.py -n 10 -d 10 -e 4 -t 1 -c 3 --dual_time_limit 30  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;
python run_synthetic30.py -n 15 -d 15 -e 4 -t 1 -c 3 --dual_time_limit 50  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;
python run_synthetic30.py -n 20 -d 15 -e 4 -t 1 -c 3 --dual_time_limit 60  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_throttles 100;

# MP4MC3+ES
python run_synthetic30.py -n 10 -d 10 -e 4 -t 1 -c 3 --dual_time_limit 30  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_obj_threshold 0.005 --dual_throttles 4;
python run_synthetic30.py -n 15 -d 15 -e 4 -t 1 -c 3 --dual_time_limit 50  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_obj_threshold 0.005 --dual_throttles 4;
python run_synthetic30.py -n 20 -d 15 -e 4 -t 1 -c 3 --dual_time_limit 60  --csp_time_limit 500 --experiment_setting nonlinear_real_ext --convergence --dual_obj_threshold 0.005 --dual_throttles 4;
