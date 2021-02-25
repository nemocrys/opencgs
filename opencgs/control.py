import datetime
import multiprocessing
import os
import platform
import subprocess
import yaml
import numpy as np

# from opencgs.parameter_study import *

from pyelmer.execute import run_elmer_solver, run_elmer_grid
from pyelmer.post import scan_logfile


def load_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def execute_multiple(simulations):
    count = multiprocessing.cpu_count()
    if platform.system() == "Windows":
        count -= 1
    print("Working on ", count, " cores.")
    pool = multiprocessing.Pool(processes=count)
    pool.map(execute_sim, simulations)


def execute_sim(simulation):
    simulation.execute()


def execute(simulations):
    count = multiprocessing.cpu_count()
    if platform.system() == "Windows":
        count -= 1
    print("Working on ", count, " cores.")
    pool = multiprocessing.Pool(processes=count)
    pool.starmap(execute_simulation, simulations.items())


def execute_simulation(sim_name, sim_dir):
    print("Starting simulation ", sim_dir, " ...")
    run_elmer_grid(sim_dir, sim_name + ".msh")
    run_elmer_solver(sim_dir)
    # post_processing(sim_path)
    err, warn, stats = scan_logfile(sim_dir)
    print(err, warn, stats)
    print("Finished simulation ", sim_dir, " .")


# def create_simulations(config):
#     if type(config) is str:
#         with open(config) as f:
#             config = yaml.safe_load(f)
#     # create file system
#     if not os.path.exists('./simdata'):
#         os.mkdir('./simdata')
#     sim_dir = './simdata/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  + '_'
#     simulations = {}
#     if False:
#         pass  # TODO permutations for parameter study
#     else:
#         sim_dir += config['general']['name']
#         os.mkdir(sim_dir)
#         config['general'].update({'sim_dir': sim_dir})
#         sim_config = sim_dir + '/config.yml'
#         with open(sim_config, 'w') as f:
#             yaml.dump(config, f, sort_keys=False)
#         subprocess.run(['python', config['control']['setup'], sim_config])
#         simulations.update({config['general']['name']: sim_dir})
#         # TODO don't run this twice
#         # run_elmer_grid(sim_dir, config['general']['name'] + '.msh')

#     with open('./simdata/simulations.yml', 'w') as f:
#         yaml.dump({'simulations': simulations}, f, sort_keys=False)
#     return simulations

# def create_transient_simulation(config_file, l_min=2e-3, l_max=12e-2):
#     crys_len = [l_min]
#     while crys_len[-1] < l_max:
#         crys_len.append(crys_len[-1] + crys_len[-1]/3)
#     crys_len[-1] = l_max
#     print(crys_len)

#     with open(config_file) as f:
#         config = yaml.safe_load(f)
#     for i in range(len(crys_len)):
#         config['general']['name'] = 'cz_induction_' + str(i)
#         config['geometry'].update({'crystal': {'l': crys_len[i]}})
#         print(config)
#         # create_simulations(config)
#     print(crys_len)
#     with open('./simdata/time_vs_length.txt', 'w') as f:
#         f.write('time [s]    length [m]\n')
#         for length in crys_len:
#             time = (length - crys_len[0]) / (4  / 6e4)
#             f.write(f'{time}    {length}\n')

# def run_diameter_iteration():
#     pass

# def post_processing():
#     pass


if __name__ == "__main__":
    config = "./examples/config.yml"
    create_transient_simulation(config)
    # simulations = create_simulations('./examples/config.yml')
    # execute(simulations)
