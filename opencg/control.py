from pyelmer.execute import run_elmer_solver
from pyelmer.post import scan_logfile

def run(sim_dir):
    run_elmer_solver(sim_dir)
    err, warn, stats = scan_logfile(sim_dir)
    print(err, warn, stats)

def run_parameter_study():
    pass

def run_diameter_iteration():
    pass

def post_processing():
    pass


if __name__ == "__main__":
    run('./simdata/_test/')
