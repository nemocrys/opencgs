import datetime
import multiprocessing
import os
import platform
import subprocess
import yaml

# from opencg.parameter_study import *

from pyelmer.execute import run_elmer_solver, run_elmer_grid
from pyelmer.post import scan_logfile


def create_simulations(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # create file system
    if not os.path.exists('./simdata'):
        os.mkdir('./simdata')
    sim_dir = './simdata/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  + '_'
    simulations = {}
    if False:
        pass  # TODO permutations for parameter study
    else:
        sim_dir += config['general']['name']
        os.mkdir(sim_dir)
        config['general'].update({'sim_dir': sim_dir})
        sim_config = sim_dir + '/config.yml'
        with open(sim_config, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        subprocess.run(['python', config['control']['setup'], sim_config])
        simulations.update({config['general']['name']: sim_dir})

    with open('./simdata/simulations.yml', 'w') as f:
        yaml.dump({'simulations': simulations}, f, sort_keys=False)
    return simulations

def execute(simulations):
    count = multiprocessing.cpu_count()
    if platform.system() == 'Windows':
        count -= 1
    print('Working on ', count, ' cores.')
    pool = multiprocessing.Pool(processes=count)
    pool.starmap(execute_simulation, simulations.items())

def execute_simulation(sim_name, sim_dir):
    print('Starting simulation ', sim_dir, ' ...')
    run_elmer_grid(sim_dir, sim_name + '.msh')
    run_elmer_solver(sim_dir)
    # postprocessing(sim_path)
    err, warn, stats = scan_logfile(sim_dir)
    print(err, warn, stats)
    print('Finished simulation ', sim_dir, ' .')

def run_diameter_iteration():
    pass

def post_processing():
    pass


if __name__ == "__main__":
    simulations = create_simulations('./examples/config.yml')
    execute(simulations)
