import datetime
import multiprocessing
import os
import platform
import subprocess
import yaml
import numpy as np


def get_git_metadata():
    """Get information about installed version in form of a dictionary.

    Returns:
        dict: {"git origin": "url", "git label": "description of commit"}
    """
    git_origin = (
        subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
        .strip()
        .decode("utf-8")
    )
    git_label = (
        subprocess.check_output(["git", "describe", "--tags", "--dirty", "--always"])
        .strip()
        .decode("utf-8")
    )
    return {"git origin": git_origin, "git label": git_label}


def load_config(config_file):
    """Load configuration from yml-file.

    Args:
        config_file (str): file path

    Returns:
        dict: configuration from file
    """
    with open(config_file) as f:
        return yaml.safe_load(f)


def execute_multiple(simulations):
    """Execute multiple simulations in parallel. On Linux (cluster),
    all CPUs are used. On Windows (Laptop) n-1 CPUs are used.

    Args:
        simulations (lis): List of Simulation objects.
    """
    count = multiprocessing.cpu_count()
    if platform.system() == "Windows":
        count -= 1
    print("Working on ", count, " cores.")
    pool = multiprocessing.Pool(processes=count)
    pool.map(execute_sim, simulations)


def execute_sim(simulation):
    """Execute single simulation."""
    simulation.execute()
