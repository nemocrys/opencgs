from copy import deepcopy
from datetime import datetime
import inspect
from importlib.metadata import version
import matplotlib
import matplotlib.pyplot as plt
import meshio
import multiprocessing
import numpy as np
import os
import pandas as pd
import platform
import scipy.interpolate as interpolate
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml

import opencgs
from opencgs import post
from opencgs.geo import compute_current_crystal_surface
import pyelmer
from pyelmer.execute import run_elmer_solver, run_elmer_grid
from pyelmer.post import scan_logfile, dat_to_dataframe


# for documentation of applied versions
PACKAGES = ["gmsh", "matplotlib", "numpy", "pandas", "pyelmer", "objectgmsh", "opencgs"]


class OpencgsError(Exception):
    pass


class Simulation:
    """In this class base functionality for all different kinds of simulations is collected."""
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        mat_config,
        config_update,
        sim_name,
        sim_type,
        base_dir,
        with_date=True,
        metadata="",
    ):
        """Create a base simulation.

        Args:
            geo (function): function for geometry generation
            geo_config (dict): configuration for geo
            sim (function): function for simulation setup
            sim_config (dict): configuration for sim
            mat_config (dict): material configuration
            config_update (dict): changes for geo_config, sim_config,
                mat_config
            sim_name (str): simulation name
            sim_type (str): abbreviation of simulation type in two
                letters
            base_dir (str): path of base directory
            with_date (bool, optional): Include date in simulation
                directory name. Defaults to True.
            metadata (str, optional): Metadata to be saved, e.g git-hash
                of parent repository. Defaults to "".
        """
        self.geo = geo
        if "geometry" in config_update:
            geo_config = self._update_config(geo_config, config_update["geometry"])
        self.geo_config = geo_config
        self.sim = sim
        if "simulation" in config_update:
            sim_config = self._update_config(sim_config, config_update["simulation"])
        self.sim_config = sim_config
        if "materials" in config_update:
            mat_config = self._update_config(mat_config, config_update["materials"])
        self.mat_config = mat_config
        self.sim_name = sim_name
        self.sim_type = sim_type
        self._create_directories(base_dir, with_date)
        self._archive_input(metadata)
        self.results_file = f"{self.sim_dir}/case.result"

    def _create_directories(self, base_dir, with_date):
        if with_date:
            self.root_dir = f"{base_dir}/{datetime.now():%Y-%m-%d_%H-%M}_{self.sim_type}_{self.sim_name}"
        else:
            self.root_dir = f"{base_dir}/{self.sim_type}_{self.sim_name}"
        if os.path.exists(self.root_dir):
            i = 1
            self.root_dir += "_1"
            while os.path.exists(self.root_dir):
                i += 1
                self.root_dir = "_".join(self.root_dir.split("_")[:-1]) + f"_{i}"
        os.makedirs(self.root_dir)
        self.input_dir = f"{self.root_dir}/01_input"
        self.sim_dir = f"{self.root_dir}/02_simulation"
        self.res_dir = f"{self.root_dir}/03_results"
        self.plot_dir = f"{self.root_dir}/04_plots"
        os.mkdir(self.input_dir)
        os.mkdir(self.sim_dir)
        os.mkdir(self.res_dir)
        os.mkdir(self.plot_dir)

    def _create_setup(self, visualize=False):
        model = self.geo(self.geo_config, self.sim_dir, self.sim_name, visualize)
        self.sim(model, self.sim_config, self.sim_dir, self.mat_config)

    def _archive_input(self, metadata):
        with open(self.input_dir + "/geo.yml", "w") as f:
            yaml.dump(self.geo_config, f)
        with open(self.input_dir + "/sim.yml", "w") as f:
            yaml.dump(self.sim_config, f)
        with open(self.input_dir + "/mat.yml", "w") as f:
            yaml.dump(self.mat_config, f)
        geo_file = inspect.getfile(self.geo)
        sim_file = inspect.getfile(self.sim)
        if geo_file == sim_file:
            shutil.copy2(geo_file, self.input_dir + "/setup.py")
        else:
            shutil.copy2(geo_file, self.input_dir + "/setup_geo.py")
            shutil.copy2(geo_file, self.input_dir + "/setup_sim.py")
        metadada = {"parent metadata": metadata, "python": sys.version}
        metadada.update(
            {"opencgs": opencgs.__version__}
        )  # don't use importlib, it doesn't get the correct version number in editable mode
        for pkg in PACKAGES:
            metadada.update({pkg: version(pkg)})
        with open(self.input_dir + "/metadata.yml", "w") as f:
            yaml.dump(metadada, f, sort_keys=False)

    @staticmethod
    def _read_names_file(names_file, skip_rows=7):  # TODO move to pyelmer
        with open(names_file) as f:
            data = f.readlines()
        data = data[7:]  # remove header
        for i in range(len(data)):
            data[i] = data[i][6:-1]  # remove index and \n
        return data

    def _postprocessing_probes(self):
        sim_dir = self.sim_dir
        res_dir = self.res_dir

        with open(sim_dir + "/probes.yml") as f:
            probes = list(yaml.safe_load(f).keys())
        df = dat_to_dataframe(sim_dir + "/results/probes.dat")

        new_names_dict = {}
        probe_idx = -1
        for old_name in df.columns.tolist():
            if "value" in old_name:
                if "coordinate 1" in old_name:  # assumes same probe sorting as in yaml-file
                    probe_idx += 1
                new_name = probes[probe_idx] + " " + old_name[7:].split(" in element ")[0]
            if "res" in old_name:
                new_name = "res " + old_name[5:]
            new_names_dict[old_name] = new_name
        df.rename(columns=new_names_dict, inplace=True)

        # keep only temperature and "res" columns
        columns_list = []
        for column in df.columns.tolist():
            if "temperature" in column and not "temperature " in column:
                columns_list.append(column)
            if "res " in column:
                columns_list.append(column)
        df = df[columns_list]

        data = {}
        for column in df.iteritems():
            data.update({column[0]: float(column[1].iloc[-1])})
        with open(res_dir + "/probes.yml", "w") as f:
            yaml.dump(data, f)
        self.probe_data = data
        df.to_csv(res_dir + "/probes.csv", index=False, sep=";")

    def execute(self):
        """Execute the simulation."""
        print("Starting simulation ", self.root_dir, " ...")
        run_elmer_grid(self.sim_dir, "case.msh")
        run_elmer_solver(self.sim_dir)
        # post_processing(sim_path)
        err, warn, stats = scan_logfile(self.sim_dir)
        if err != []:
            raise OpencgsError(f"The simulation {self.sim_name} was not successful.\nThe following errors were found in the log: {err}")
        else:
            print(err, warn, stats)
            print("Finished simulation ", self.root_dir, " .")
            print("Post processing...")
            self._postprocessing_probes()
            self.post()
            print("Finished post processing.")

    def post(self):
        """Run the post-processing."""
        err, warn, stats = scan_logfile(self.sim_dir)
        with open(self.res_dir + "/elmer_summary.yml", "w") as f:
            yaml.dump(
                {"Errors": err, "Warnings": warn, "Statistics": stats},
                f,
                sort_keys=False,
            )

    @staticmethod
    def _update_config(base_config, config_update):
        if config_update is not None:
            for param, update in config_update.items():
                if type(update) is dict:
                    base_config[param] = Simulation._update_config(
                        base_config[param], config_update[param]
                    )
                else:
                    base_config[param] = update
        return base_config

    @property
    def vtu_files(self):
        """List of file-path's to vtus with simulation results."""
        return [f for f in os.listdir(self.sim_dir) if f.split(".")[-1] == "vtu"]

    @property
    def last_interation(self):
        """number of iterations"""
        return max([int(f[-8:-4]) for f in self.vtu_files])

    @property
    def phase_interface(self):
        """List of phase boundary coordinates."""
        header = ["iteration", "BC", "NodeIdx", "x", "y", "z", "T"]
        df = pd.read_table(
            f"{self.sim_dir}/results/phase-if.dat",
            names=header,
            sep=" ",
            skipinitialspace=True,
        )
        return [[line["x"], line["y"]] for _, line in df.iterrows()]

    @property
    def mesh_files(self):
        """List of Elmer mesh files."""
        files = []
        for f in os.listdir(self.sim_dir):
            e = f.split(".")[-1]
            if e == "boundary" or e == "elements" or e == "header" or e == "nodes":
                files.append(f)
        return files

    @property
    def last_heater_current(self):
        """Heater current in last iteration."""
        I_base = self.sim_config["heating_induction"]["current"]
        pwr_scaling = self.probe_data["res heater power scaling"]
        return I_base * pwr_scaling ** 0.5


class SteadyStateSim(Simulation):
    """In this class functionality for steady state simulations is
    collected."""
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        mat_config,
        config_update={},
        sim_name="opencgs-simulation",
        base_dir="./simdata",
        visualize=False,
        with_date=True,
        metadata="",
        **_,
    ):
        """Create a steady-state simulation.

        Args:
            geo (function): function for geometry generation
            geo_config (dict): configuration for geo
            sim (function): function for simulation setup
            sim_config (dict): configuration for sim
            mat_config (dict): material configuration
            config_update (dict, optional): changes for geo_config,
                sim_config, mat_config. Defaults to {}.
            sim_name (str, optional): simulation name.
                Defaults to "opencgs-simulation".
            base_dir (str, optional): path of base directory. Defaults
                to "./simdata".
            visualize (bool, optional): run GUI (of mesh generator).
                Defaults to False.
            with_date (bool, optional): Include date in simulation
                directory name. Defaults to True.
            metadata (str, optional): Metadata to be saved, e.g git-hash
                of parent repository. Defaults to "".
        """
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update,
            sim_name,
            "ss",
            base_dir,
            with_date,
            metadata,
        )
        self._create_setup(visualize)

    def post(self):
        """Run post-processing."""
        super().post()
        print("evaluating heat fluxes")
        try:
            post.heat_flux(self.sim_dir, self.res_dir)
        except Exception as exc:
            print("Could not evaluate heat fluxes :(")
            print(exc)

    @property
    def T_tp(self):
        """Temperature at triple-point. In case of an error 0.0 is
        returned."""
        try:
            with open(f"{self.res_dir}/probes.yml", "r") as f:
                res = yaml.safe_load(f)
            return res["res triple point temperature"]
        except FileNotFoundError:
            return 0.0


class ParameterStudy(Simulation):
    """In this class functionality for parameter studies is collected."""
    def __init__(
        self,
        SimulationClass,
        geo,
        geo_config,
        sim,
        sim_config,
        mat_config,
        study_params,
        config_update={},
        create_permutations=False,
        sim_name="opencgs-parameter-study",
        base_dir="./simdata",
        with_date=True,
        metadata="",
        **kwargs,
    ):
        """Create a parameter study.

        Args:
            SimulationClass (Simulation): class of simulations that is
                investigated in parameter study.
            geo (function): function for geometry generation
            geo_config (dict): configuration for geo
            sim (function): function for simulation setup
            sim_config (dict): configuration for sim
            mat_config (dict): material configuration
            study_params (dict): list of parameter updates for parameter
                study.
            config_update (dict, optional): changes for geo_config,
                sim_config, mat_config. Defaults to {}.
            create_permutations (bool, optional): If True: create
                permutations of all given parameters. If False: other
                parameters will be left at the default. Defaults to
                False.
            sim_name (str, optional): simulation name.
                Defaults to "opencgs-simulation".
            base_dir (str, optional): path of base directory. Defaults
                to "./simdata".
            visualize (bool, optional): run GUI (of mesh generator).
                Defaults to False.
            with_date (bool, optional): Include date in simulation
                directory name. Defaults to True.
            metadata (str, optional): Metadata to be saved, e.g git-hash
                of parent repository. Defaults to "".
        """
        if SimulationClass == SteadyStateSim:
            type_str = "ps"
        elif SimulationClass == DiameterIteration:
            type_str = "pd"
        else:
            raise TypeError(
                f"SimulationClass {SimulationClass} not supported in parameter study."
            )
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update,
            sim_name,
            type_str,
            base_dir,
            with_date=with_date,
            metadata=metadata,
        )
        with open(self.input_dir + "/study_params.yml", "w") as f:
            yaml.dump(study_params, f)
        self.sims = []
        self.create_sims(SimulationClass, study_params, create_permutations, kwargs)

    def create_sims(self, SimulationClass, study_params, create_permutations, kwargs):
        """Create simulations for parameter study"""
        keys, vals = self._create_param_lists(study_params)
        print("updating the following parameters:", keys, vals)
        if create_permutations:
            # TODO create permutations!
            # use numpy meshgrid?
            NotImplementedError(
                "Permutations for parameter studies are not available yet."
            )
        else:
            for i in range(len(keys)):
                key_list = keys[i]
                for val in vals[i]:
                    config_update = {key_list[-1]: val}
                    for j in range(len(key_list) - 1):
                        config_update = {key_list[-(j + 2)]: config_update}
                    sim_name = ";".join(key_list) + f"={val}"
                    print(kwargs)
                    sim = SimulationClass(
                        geo=self.geo,
                        geo_config=deepcopy(self.geo_config),
                        sim=self.sim,
                        sim_config=deepcopy(self.sim_config),
                        mat_config=deepcopy(self.mat_config),
                        config_update=config_update,
                        sim_name=sim_name,
                        base_dir=self.sim_dir,
                        with_date=False,
                        **kwargs,
                    )
                    self.sims.append(sim)

    def _create_param_lists(self, study_params, previous_keys=[]):
        keys = []
        vals = []
        for key, val in study_params.items():
            if type(val) is dict:
                k, v = self._create_param_lists(val, previous_keys + [key])
                keys += k
                vals += v
            elif type(val) is list:
                keys.append(previous_keys + [key])
                vals.append(val)
            else:
                raise ValueError("Wrong format in study_parameters!")
        return keys, vals

    def execute(self):
        """Execute parameter study in parallel. Linux (cluster): on all
        CUPs, Windows (Laptop) on n-1 CPUs."""
        count = multiprocessing.cpu_count()
        # TODO not sure if that's a good idea...
        if platform.system() == "Windows":
            count -= 1
        print("Working on ", count, " cores.")
        pool = multiprocessing.Pool(processes=count)
        pool.map(ParameterStudy._execute_sim, self.sims)
        self.post()

    @staticmethod
    def _execute_sim(simulation):
        simulation.execute()

    def post(self):
        """Run post-processing."""
        for sim in self.sims:
            for f in os.listdir(sim.res_dir):
                ext = f.split(".")[-1]
                if ext in ["yaml", "yml"]:
                    shutil.copy2(
                        f"{sim.res_dir}/{f}",
                        f"{self.res_dir}/{f[:-(len(ext) + 1)]}_{sim.sim_name}.{ext}",
                    )
        post.parameter_study(self.sim_dir, self.plot_dir)


class DiameterIteration(Simulation):
    """In this class functionality for simulations with iterative
    crystal diameter computation is collected."""
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        mat_config,
        config_update={},
        T_tp=505,
        r_min=0.001,
        r_max=0.02,
        max_iterations=10,
        dT_max=0.01,
        sim_name="opencgs-diameter-iteration",
        base_dir="./simdata",
        with_date=True,
        metadata="",
        **_,
    ):
        """Run a simulation with iterative crystal diameter computation,
        consisting of multiple steady-state simulations.

        Args:
            geo (function): function for geometry generation
            geo_config (dict): configuration for geo
            sim (function): function for simulation setup
            sim_config (dict): configuration for sim
            mat_config (dict): material configuration
            config_update (dict, optional): changes for geo_config,
                sim_config, mat_config. Defaults to {}.
            T_tp (float, optional): Triple point temperature. Defaults
                to 505 (tin).
            r_min (float, optional): Minimum allowable crystal radius.
                Defaults to 0.001.
            r_max (float, optional): Maximum allowable crystal radius.
                Defaults to 0.02.
            max_iterations (int, optional): Maximum number of
               iterations. Defaults to 10.
            dT_max (float, optional): Convergence criterion: maximum
                allowable difference in triple point temperature.
                Defaults to 0.01.
            sim_name (str, optional): simulation name.
                Defaults to "opencgs-simulation".
            base_dir (str, optional): path of base directory. Defaults
                to "./simdata".
            with_date (bool, optional): Include date in simulation
                directory name. Defaults to True.
            metadata (str, optional): Metadata to be saved, e.g git-hash
                of parent repository. Defaults to "".
        """
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update,
            sim_name,
            "di",
            base_dir,
            with_date,
            metadata,
        )
        if (
            self.sim_config["smart-heater"]
            and not self.sim_config["smart-heater"]["control-point"]
        ):
            raise ValueError(
                "Smart heater with control at triple point. Iteration useless."
            )
        with open(self.input_dir + "/di_params.yml", "w") as f:
            yaml.dump(
                {
                    "T_tp": T_tp,
                    "r_min": r_min,
                    "r_max": r_max,
                    "max_iterations": max_iterations,
                    "dT_max": dT_max,
                },
                f,
            )
        self.T_tp = T_tp
        self.r_min = r_min
        self.r_max = r_max
        self.max_iterations = max_iterations
        self.dT_max = dT_max

    def execute(self):
        """Run iterative diameter computation process, consisting of
        multiple steady-state simulations."""
        # initial simulations
        geo_config = deepcopy(self.geo_config)
        geo_config["crystal"]["r"] = self.r_min
        sim_r_min = SteadyStateSim(
            self.geo,
            geo_config,
            self.sim,
            self.sim_config,
            self.mat_config,
            sim_name=f"r={self.r_min}",
            base_dir=self.sim_dir,
            with_date=False,
        )
        sim_r_min.execute()
        T_rmin = sim_r_min.T_tp
        geo_config = deepcopy(self.geo_config)
        geo_config["crystal"]["r"] = self.r_max
        sim_r_max = SteadyStateSim(
            self.geo,
            geo_config,
            self.sim,
            self.sim_config,
            self.mat_config,
            sim_name=f"r={self.r_max}",
            base_dir=self.sim_dir,
            with_date=False,
        )
        sim_r_max.execute()
        T_rmax = sim_r_max.T_tp
        # evaluate
        print(f"r-min: {self.r_min} m - T = {T_rmin:5f} K")
        print(f"r-max: {self.r_max} m - T = {T_rmax:5f} K")
        Ttp_r = {}  # triple point temperature : radius
        Ttp_r.update({T_rmin: self.r_min})
        Ttp_r.update({T_rmax: self.r_max})
        self.export_Ttp_r(Ttp_r)
        if not T_rmin <= self.T_tp <= T_rmax:
            print("Diameter fitting impossible with current setup.")
            print("Interpolated radius would be", self.compute_new_r(Ttp_r), "m.")
            with open(f"{self.res_dir}/iteration-summary.yml", "w") as f:
                yaml.dump(self.T_tp, f)
        # iteration
        converged = False
        for i in range(self.max_iterations):
            print("Diameter iteration", i + 1)
            r_new = self.compute_new_r(Ttp_r)
            print("new radius:", r_new)
            if not (self.r_min < r_new < self.r_max):
                print("ERROR: Interpolation not possible.")
                break
            geo_config = deepcopy(self.geo_config)
            geo_config["crystal"]["r"] = r_new
            sim = SteadyStateSim(
                self.geo,
                geo_config,
                self.sim,
                self.sim_config,
                self.mat_config,
                sim_name=f"r={r_new}",
                base_dir=self.sim_dir,
                with_date=False,
            )
            sim.execute()
            Ttp_new = sim.T_tp
            print("corresponding TP temperature:", Ttp_new)
            Ttp_r.update({Ttp_new: r_new})
            self.export_Ttp_r(Ttp_r)
            if np.abs(Ttp_new - self.T_tp) <= self.dT_max:
                print("Iteration finished.")
                print("Crystal radius =", r_new, "m.")
                print("TP Temperature =", Ttp_new, "K")
                converged = True
                break

        results = {
            "converged": converged,
            "iterations": i + 1,
            "dT at TP": Ttp_new - self.T_tp,
            "Ttp_r": Ttp_r,
        }
        with open(f"{self.res_dir}/iteration-summary.yml", "w") as f:
            yaml.dump(results, f)
        if converged:
            results.update({"radius": r_new})
            shutil.copytree(sim.root_dir, f"{self.res_dir}/{sim.sim_name}")
        self.plot(Ttp_r)

    def plot(self, Ttp_r):
        """Plot convergence

        Args:
            Ttp_r (dict): Temperature @TP: crystal radius
        """
        fig, ax = plt.subplots(1, 2, figsize=(5.75, 3))
        ax[0].plot(list(Ttp_r.keys()), "x-")
        ax[0].set_xlabel("simulation")
        ax[0].set_ylabel("temperature at triple point [K]")
        ax[0].grid()
        T_tps_sorted = {
            k: v for k, v in sorted(Ttp_r.items(), key=lambda item: item[1])
        }
        ax[1].plot(list(T_tps_sorted.values()), list(T_tps_sorted.keys()), "x-")
        ax[1].set_xlabel("crystal radius [m]")
        ax[1].set_ylabel("temperature at triple point [K]")
        ax[1].grid()
        fig.tight_layout()
        fig.savefig(f"{self.res_dir}/diameter-iteration.png")
        plt.close(fig)

    def compute_new_r(self, Ttp_r):
        """Compute new crystal radius for next iteration.

        Args:
            Ttp_r (dict): Temperature @TP: crystal radius
                (from previous iterations)
        """
        T_tps = np.fromiter(Ttp_r.keys(), float)
        rs = np.fromiter(Ttp_r.values(), float)
        # ignore failed simulations (if possible)
        T_tps_new = np.array([T for T in T_tps if T > 0.0])
        if len(T_tps_new) >= 2:
            T_tps = T_tps_new
        # try to INTERpolate
        T1 = 0.0
        T2 = self.T_tp * 2
        for T in T_tps:
            if T1 < T < self.T_tp:
                T1 = T
            else:
                T2 = T
        try:
            r1 = Ttp_r[T1]
            r2 = Ttp_r[T2]
            r_new = r1 + (r2 - r1) / (T2 - T1) * (self.T_tp - T1)
            r_new = round(float(r_new), 8)
            if r_new == rs[-1]:
                print(
                    "WARNING: Non-linearity, tanking value from last iteration for interpolation."
                )
                if T_tps[-1] < self.T_tp:
                    T1 = T_tps[-1]
                    r1 = Ttp_r[T1]
                else:
                    T2 = T_tps[-1]
                    r2 = Ttp_r[T2]
        except KeyError:
            print("Warning: Could not interpolate!")
        # EXTRApolate if necessary
        if T1 == 0.0:
            T_diff = sorted(T_tps - self.T_tp)
            T1 = T_diff[0]
            T2 = T_diff[1]
        if T2 == self.T_tp * 2:
            T_diff = sorted(T_tps - self.T_tp)
            T1 = T_diff[-2]
            T2 = T_diff[-1]
        r_new = r1 + (r2 - r1) / (T2 - T1) * (self.T_tp - T1)
        r_new = round(float(r_new), 8)
        print("selected points for interpolation")
        print(Ttp_r)
        print("T1 =", T1)
        print("T2 =", T2)
        return r_new

    def export_Ttp_r(self, Ttp_r):
        """Write dictionary with triple point temperature and radius to
        file."""
        with open(f"{self.res_dir}/Ttp_r.yml", "w") as f:
            yaml.dump(Ttp_r, f, sort_keys=False)


class QuasiTransientSim(Simulation):
    def __init__(
            self,
            quasi_transient,
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update={},
            sim_name="opencgs-quasi-transient",
            base_dir="./simdata",
            with_date=True,
            metadata="",
            simulation_class=SteadyStateSim,
            **kwargs
    ):
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update,
            sim_name,
            "qt",
            base_dir,
            with_date=with_date,
            metadata=metadata,
        )

        with open(f"{self.input_dir}/quasi_transient.yml", "w") as f:
            yaml.dump(quasi_transient, f)
        
        self.simulation_class = simulation_class
        self.sims = []
        self.create_sims(quasi_transient, kwargs)

    def create_sims(self, quasi_transient, kwargs):
        for step in quasi_transient:
            crystal_length = step["length"]
            v_pull = step["v_pull"]
            config_update = {
                "geometry": {"crystal": {"current_length": crystal_length}},
                "simulation": {"general": {"v_pull": v_pull}},
            }
            sim_name = f"length={crystal_length}_vpull={v_pull}"
            sim = self.simulation_class(
                geo=self.geo,
                geo_config=deepcopy(self.geo_config),
                sim=self.sim,
                sim_config=deepcopy(self.sim_config),
                mat_config=deepcopy(self.mat_config),
                config_update=config_update,
                sim_name=sim_name,
                base_dir=self.sim_dir,
                with_date=False,
                **kwargs,
            )
            self.sims.append(sim)

    def execute(self):
        """Execute parameter study in parallel. Linux (cluster): on all
        CUPs, Windows (Laptop) on n-1 CPUs."""
        count = multiprocessing.cpu_count()
        # TODO not sure if that's a good idea...
        if platform.system() == "Windows":
            count -= 1
        print("Working on ", count, " cores.")
        pool = multiprocessing.Pool(processes=count)
        pool.map(ParameterStudy._execute_sim, self.sims)
        self.post()

    @staticmethod
    def _execute_sim(simulation):
        simulation.execute()

    def post(self):
        """Run post-processing."""
        for sim in self.sims:
            for f in os.listdir(sim.res_dir):  # collect yaml files
                ext = f.split(".")[-1]
                if ext in ["yaml", "yml"]:
                    shutil.copy2(
                        f"{sim.res_dir}/{f}",
                        f"{self.res_dir}/{f[:-(len(ext) + 1)]}_{sim.sim_name}.{ext}",
                    )
                if f == "NOT_CONVERGED":
                    shutil.copy2(
                        f"{sim.res_dir}/{f}",
                        f"{self.res_dir}/{f}_{sim.sim_name}",
                    )

        # post.parameter_study(self.sim_dir, self.plot_dir)
        self.create_plots()
        self.create_pvd()

    def create_plots(self):
        probe_values = {}
        for file in os.listdir(self.res_dir):
            if file.split("_")[0] == "probes":
                length = float(file.split("=")[1].split("_")[0])
                with open(f"{self.res_dir}/{file}") as f:
                    data = yaml.safe_load(f)
                data.update({"length": length})
                for key, value in data.items():
                    if key in probe_values:
                        probe_values[key].append(value)
                    else:
                        probe_values[key] = [value]
        lengths = np.array(probe_values.pop("length"))
        sort_indices = np.argsort(lengths)
        for probe, values in probe_values.items():
            fig, ax = plt.subplots()
            ax.plot(lengths[sort_indices], np.array(values)[sort_indices])
            ax.set_xlabel("crystal length")
            ax.set_ylabel(probe)
            fig.tight_layout()
            fig.savefig(f"{self.plot_dir}/{probe}.png")
            plt.close(fig)

    def create_pvd(self):
        length_simulation = {}
        for simulation in os.listdir(self.sim_dir):
            length = float(simulation.split("=")[1].split("_")[0])
            length_simulation.update({length: simulation})

        length_simulation = {k: length_simulation[k] for k in sorted(length_simulation.keys())}

        main_tree = None
        for length, simulation in length_simulation.items():
            try:  # SteadyStateSim
                simulation_subdir = "02_simulation"
                tree = ET.parse(f'{self.sim_dir}/{simulation}/{simulation_subdir}/case.pvd')
            except FileNotFoundError:  # CoupledSim
                elmer_dirs = [x for x in os.listdir(f'{self.sim_dir}/{simulation}/02_simulation') if "elmer_" in x]
                simulation_subdir = f"02_simulation/{sorted(elmer_dirs)[-1]}"
                tree = ET.parse(f'{self.sim_dir}/{simulation}/{simulation_subdir}/case.pvd')
            
            root = tree.getroot()
            
            # Modify the "file" entry
            for dataset in root.iter('DataSet'):
                current_file = dataset.attrib['file']
                dataset.attrib['file'] = f"{simulation}/{simulation_subdir}/{current_file}"
            
            # Modify the "timestep" entry
            for dataset in root.iter('DataSet'):
                dataset.set('timestep', str(length))
            
            if main_tree is None:
                main_tree = tree
                # main_tree.write(f"{sim_dir}/case.pvd")
            else:
                # main_tree = ET.parse(f"{sim_dir}/case.pvd")
                main_root = main_tree.getroot()
                main_collection = main_root.find("Collection")
                new_collection = root.find("Collection")
                for dataset in new_collection.findall('DataSet'):
                    main_collection.append(dataset)
                main_tree = ET.ElementTree(main_root)
        main_tree.write(f"{self.sim_dir}/case.pvd")


class CoupledSim(Simulation):
    def __init__(
        self,
        geo,
        geo_config,
        sim,  # Elmer
        sim_config,  # Elmer
        setup_openfoam,
        setup_openfoam_config,
        coupling_config,
        mat_config,
        config_update={},  # works currently only for Elmer
        sim_name="opencgs-coupled-simulation",
        base_dir="./simdata",
        with_date=True,
        metadata="",
        **kwargs,
    ):
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            mat_config,
            config_update,  # only for Elmer
            sim_name,
            "eo",
            base_dir,
            with_date=with_date,
            metadata=metadata,
        )

        self.setup_openfoam = setup_openfoam
        self.setup_openfoam_config = setup_openfoam_config
        self.coupling_config = coupling_config
        with open(self.input_dir + "/openfoam.yml", "w") as f:
            yaml.dump(setup_openfoam_config, f)
        with open(self.input_dir + "/coupling.yml", "w") as f:
            yaml.dump(coupling_config, f)
        of_file = inspect.getfile(self.geo)
        shutil.copy2(of_file, self.input_dir + "/setup_openfoam.py")

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def clean_openfoam_meshio_data(mesh):
        # TODO these fixes could be implemented in a cleaner way
        mesh.point_data["U"][
            :, 2
        ] = 0  # remove z-component, it's wrong in point_data (correct in cell_data)
        mesh.points[
            :, 1
        ] += 1e-5  # minimal shift to get better interpolation at melt surface

    @staticmethod
    def vtk_to_msh(vtk_file, foam_dir):
        mesh = meshio.read(vtk_file)
        CoupledSim.clean_openfoam_meshio_data(mesh)
        mesh.write(f"{foam_dir}/VTK/of_result.msh", file_format="gmsh22", binary=False)

    @staticmethod
    def vtk_to_msh_update_variable_name(input_file_vtk, output_file_msh, old_name, new_name):
        mesh = meshio.read(input_file_vtk)
        CoupledSim.clean_openfoam_meshio_data(mesh)
        mesh.point_data[new_name] = mesh.point_data.pop(old_name)
        mesh.cell_data[new_name] = mesh.cell_data.pop(old_name)
        mesh.write(output_file_msh, file_format="gmsh22", binary=False)

    @staticmethod
    def vtk_to_msh_with_u_old(vtk_file, vtk_file_old, foam_dir, flow_variable):
        # this function assumes that both OpenFOAM simulations where run with a similar 
        # structured mesh
        mesh = meshio.read(vtk_file)
        CoupledSim.clean_openfoam_meshio_data(mesh)
        mesh_old = meshio.read(vtk_file_old)
        CoupledSim.clean_openfoam_meshio_data(mesh_old)
        mesh.point_data[f"{flow_variable}_old"] = mesh_old.point_data[flow_variable]
        mesh.write(f"{foam_dir}/VTK/of_result.msh", file_format="gmsh22", binary=False)


    @staticmethod
    def relax_phase_if(
        old_interface_interp=None,
        new_interface_file="simdata_elmer/resutls/save_line_melt_crys.dat",
        relaxation_factor=0.5,
        visualize=False,
        fix_coordinates=False,
        crystal_radius=0.0,
    ):
        df = dat_to_dataframe(new_interface_file)
        df = df.loc[df["Call count"] == df["Call count"].max()]
        df = df.sort_values("coordinate 1")
        df["coordinate 1"] = df["coordinate 1"].round(20)
        df["coordinate 2"] = df["coordinate 2"].round(20)
        if fix_coordinates:
            # fix coordinates, required since mgdyn slightly shifts them
            df.at[df['coordinate 1'].idxmin(), 'coordinate 1'] = 0
            if crystal_radius != 0.0:
                df.at[df['coordinate 1'].idxmax(), 'coordinate 1'] = crystal_radius
        x = np.linspace(0, df["coordinate 1"].max(), 100)
        if old_interface_interp is None:
            old_interface_interp = interpolate.interp1d(
                [x[0], x[-1]], 2 * [df["coordinate 2"][df.index[-1]]], kind="linear", fill_value="extrapolate"
            )
        new_interface_interp = interpolate.interp1d(
            df["coordinate 1"], df["coordinate 2"], kind="linear", fill_value="extrapolate"
        )
        interface_relaxed = lambda x: relaxation_factor * new_interface_interp(x) + (
            1 - relaxation_factor
        ) * old_interface_interp(x)

        if visualize:
            fig, ax = plt.subplots()
            # ax.plot(df["coordinate 1"], df["coordinate 2"], "x-", label="sorted raw data")
            ax.plot(x, new_interface_interp(x), "x-", label="new interface")
            ax.plot(x, old_interface_interp(x), "x-", label="old interface")
            ax.plot(x, interface_relaxed(x), "x-", label="relaxed interface")
            ax.legend()
            plt.show()
            plt.close(fig)

        shutil.copy(
            f"{new_interface_file[:-4]}.dat.names",
            f"{new_interface_file[:-4]}_relaxed.dat.names",
        )
        df["coordinate 2"] = interface_relaxed(df["coordinate 1"])
        df.to_csv(
            f"{new_interface_file[:-4]}_relaxed.dat", header=False, sep=" ", index=False
        )

        return interface_relaxed

    @staticmethod
    def relax_temperatures_heatfluxes(
        old_elmer_result,
        new_elmer_result,
        relaxation_factor=0.5,
        visualize=False,
        coordinate_tolerance=1e-4,
    ):
        # This only works if the mesh at the boundaries doesn't change betweeen the iterations
        # It's not guaranteed but should be working, if not there will be an error
        df_old = dat_to_dataframe(old_elmer_result)
        df_old = df_old.loc[df_old["Call count"] == df_old["Call count"].max()]
        df_old["coordinate 1"] = df_old.loc[:, "coordinate 1"].round(20)
        df_old["coordinate 2"] = df_old.loc[:, "coordinate 2"].round(20)
        df_old = df_old.drop_duplicates(subset=["Node index"]).sort_values(["coordinate 1", "coordinate 2"], ignore_index=True)

        df_new = dat_to_dataframe(new_elmer_result)
        df_new = df_new.loc[df_new["Call count"] == df_new["Call count"].max()]
        df_new["coordinate 1"] = df_new.loc[:, "coordinate 1"].round(20)
        df_new["coordinate 2"] = df_new.loc[:, "coordinate 2"].round(20)
        df_new = df_new.drop_duplicates(subset=["Node index"]).sort_values(["coordinate 1", "coordinate 2"], ignore_index=True)

        coord_diff = (
            abs(df_old["coordinate 1"] - df_new["coordinate 1"]).sum()
            + abs(df_old["coordinate 2"] - df_new["coordinate 2"]).sum()
        )
        if coord_diff > coordinate_tolerance:
            raise ValueError(
                f"The coordinates at the boundary do not agree, there is a total difference of {coord_diff:.2e} between {old_elmer_result} and {new_elmer_result}. Relaxation not possible."
            )
        df_relaxed = deepcopy(df_new)
        for value in ["temperature", "temperature grad 1", "temperature grad 2"]:
            df_relaxed[value] = (
                relaxation_factor * df_new[value]
                + (1 - relaxation_factor) * df_old[value]
            )
        if visualize:
            fig, ax = plt.subplots(2, 3)
            ax[0, 0].plot(df_old["coordinate 1"], df_old["temperature"], label="old")
            ax[0, 0].plot(df_new["coordinate 1"], df_new["temperature"], label="new")
            ax[0, 0].plot(
                df_relaxed["coordinate 1"], df_relaxed["temperature"], label="relaxed"
            )
            ax[0, 0].set_xlabel("x in m")
            ax[0, 0].set_ylabel("T in K")
            ax[0, 0].grid(linestyle=":")
            ax[0, 0].legend()

            ax[0, 1].plot(
                df_old["coordinate 1"], df_old["temperature grad 1"], label="old"
            )
            ax[0, 1].plot(
                df_new["coordinate 1"], df_new["temperature grad 1"], label="new"
            )
            ax[0, 1].plot(
                df_relaxed["coordinate 1"],
                df_relaxed["temperature grad 1"],
                label="relaxed",
            )
            ax[0, 1].set_xlabel("x in m")
            ax[0, 1].set_ylabel("heat flux x in W/m^2")
            ax[0, 1].grid(linestyle=":")

            ax[0, 2].plot(
                df_old["coordinate 1"], df_old["temperature grad 2"], label="old"
            )
            ax[0, 2].plot(
                df_new["coordinate 1"], df_new["temperature grad 2"], label="new"
            )
            ax[0, 2].plot(
                df_relaxed["coordinate 1"],
                df_relaxed["temperature grad 2"],
                label="relaxed",
            )
            ax[0, 2].set_xlabel("x in m")
            ax[0, 2].set_ylabel("heat flux y in W/m^2")
            ax[0, 2].grid(linestyle=":")

            ax[1, 0].plot(df_old["coordinate 2"], df_old["temperature"], label="old")
            ax[1, 0].plot(df_new["coordinate 2"], df_new["temperature"], label="new")
            ax[1, 0].plot(
                df_relaxed["coordinate 2"], df_relaxed["temperature"], label="relaxed"
            )
            ax[1, 0].set_xlabel("x in m")
            ax[1, 0].set_ylabel("T in K")
            ax[1, 0].grid(linestyle=":")

            ax[1, 1].plot(
                df_old["coordinate 2"], df_old["temperature grad 1"], label="old"
            )
            ax[1, 1].plot(
                df_new["coordinate 2"], df_new["temperature grad 1"], label="new"
            )
            ax[1, 1].plot(
                df_relaxed["coordinate 2"],
                df_relaxed["temperature grad 1"],
                label="relaxed",
            )
            ax[1, 1].set_xlabel("x in m")
            ax[1, 1].set_ylabel("heat flux x in W/m^2")
            ax[1, 1].grid(linestyle=":")

            ax[1, 2].plot(
                df_old["coordinate 2"], df_old["temperature grad 2"], label="old"
            )
            ax[1, 2].plot(
                df_new["coordinate 2"], df_new["temperature grad 2"], label="new"
            )
            ax[1, 2].plot(
                df_relaxed["coordinate 2"],
                df_relaxed["temperature grad 2"],
                label="relaxed",
            )
            ax[1, 2].set_xlabel("x in m")
            ax[1, 2].set_ylabel("heat flux y in W/m^2")
            ax[1, 2].grid(linestyle=":")

            fig.tight_layout()
            plt.show()

        df_relaxed.to_csv(
            f"{new_elmer_result[:-4]}_relaxed.dat", header=False, sep=" ", index=False
        )

    @staticmethod
    def check_interface_convergence(
        old_interface, new_interface, x_max, number_of_points
    ):
        x = np.linspace(0, x_max, number_of_points)
        x = np.round(
            x, 20
        )  # this has to be lower than the rounding in relax_phase_if to avoid out of range error
        y_old = old_interface(x)
        y_new = new_interface(x)
        avg_change = np.abs(y_old - y_new).mean()
        max_change = np.abs(y_old - y_new).max()
        return avg_change, max_change, y_new[0]

    @staticmethod
    def extract_phase_if(input_file="simdata_elmer/resutls/save_line_melt_crys.dat"):
        df = dat_to_dataframe(input_file)
        df = df.loc[df["Call count"] == df["Call count"].max()]

        _x = df["coordinate 1"].to_numpy()
        _y = df["coordinate 2"].to_numpy()

        x = [_x[i] for i in _x.argsort()]
        y = [_y[i] for i in _x.argsort()]

        points = []
        for i in range(len(x)):
            points.append([float(round(x[-i - 1], 20)), float(round(y[-i - 1], 20))])
        uniquie_points = list(
            set(map(tuple, points))
        )  # remove duplicates (ChatGPT code)
        return {"phase_if": points}

    @staticmethod
    def read_and_update_pvd(base_dir, elmer_dir, timestep):
        # Parse the XML files
        tree = ET.parse(f"{base_dir}/{elmer_dir}/case.pvd")
        root = tree.getroot()

        # Modify the "file" entry
        for dataset in root.iter("DataSet"):
            current_file = dataset.attrib["file"]
            dataset.attrib["file"] = f"{elmer_dir}/{current_file}"

        # Modify the "timestep" entry
        for dataset in root.iter("DataSet"):
            dataset.set("timestep", str(timestep))

        pvd_file = f"{base_dir}/elmer-result.pvd"
        if timestep == 0:  # write directly
            tree.write(pvd_file)
        else:  # update old tree, write then
            old_tree = ET.parse(pvd_file)
            old_root = old_tree.getroot()
            collection_old = old_root.find("Collection")
            collection_update = root.find("Collection")
            for dataset in collection_update.findall("DataSet"):
                collection_old.append(dataset)
            new_tree = ET.ElementTree(old_root)
            new_tree.write(pvd_file)

    @staticmethod
    def is_converged(openfoam_logfile):
        with open(openfoam_logfile) as log_file:
            log_lines = log_file.readlines()[-15:]  # check last lines only
        return any("solution converged" in line for line in reversed(log_lines))

    def execute(self):
        last_foam_dir = None
        last_vtk_file = None
        last_elmer_dir = None
        if self.coupling_config["transient_flow"]:
            flow_variable = "UMean_avg300s"
        else:
            flow_variable = "U"
        openfoam_converged = True  # used for steady state simulations only
        last_openfoam_converged = True
        coupling_converged = False
        for i in range(self.coupling_config["maximum_iterations"]):
            if i > 0:
                last_elmer_dir = elmer_dir
                last_foam_dir = foam_dir
                last_vtk_file = vtk_file
            elmer_dir = f"{self.sim_dir}/elmer_{i+1:02}"
            if not os.path.exists(elmer_dir):
                os.makedirs(elmer_dir)
            if i > 0:
                shutil.copy(
                    f"{foam_simdir}/VTK/of_result.msh", f"{elmer_dir}/of_result.msh"
                )
            foam_dir = f"{self.sim_dir}/openfoam_{i+1:02}"
            with open(f"{elmer_dir}/config_geo.yml", "w") as f:
                yaml.safe_dump(self.geo_config, f, sort_keys=False)
            # run elmer
            model = self.geo(deepcopy(self.geo_config), elmer_dir)
            if i in self.coupling_config["flow_scaling"]:
                flow_scaling = self.coupling_config["flow_scaling"][i]
            else:
                flow_scaling = 1
            if i in self.coupling_config["heat_conductivity_scaling"]:
                heat_conductivity_scaling = self.coupling_config["heat_conductivity_scaling"][i]
            else:
                heat_conductivity_scaling = 1
            if i == 0:
                self.sim(model, self.sim_config, elmer_dir, self.mat_config, heat_conductivity_scaling=heat_conductivity_scaling)
            elif i == 1:  # flow relaxation not yet possible
                self.sim(
                        model,
                        self.sim_config,
                        elmer_dir,
                        self.mat_config,
                        with_flow=True,
                        u_scaling=flow_scaling,
                        flow_variable=flow_variable,
                        heat_conductivity_scaling=heat_conductivity_scaling,
                    )
            else:  # with flow relaxation
                self.sim(
                    model,
                    self.sim_config,
                    elmer_dir,
                    self.mat_config,
                    with_flow=True,
                    u_scaling=flow_scaling,
                    flow_variable=flow_variable,
                    flow_variable_old=f"{flow_variable}_old",
                    flow_relaxation_factor=self.coupling_config["flow_relaxation_factor"],
                )
            print("Running ElmerGrid...")
            run_elmer_grid(elmer_dir, "case.msh")
            print("Running ElmerSolver...")
            run_elmer_solver(elmer_dir)
            err, warn, stats = scan_logfile(elmer_dir)
            print("Errors:", err)
            print("Warnings:", warn)
            print("Statistics:", stats)
            self._postprocessing_probes_2(elmer_dir, elmer_dir)
            self.read_and_update_pvd(self.sim_dir, f"elmer_{i+1:02}", i)
            print("Relaxing interface, checking for convergence")
            if i == 0:
                old_interface = self.relax_phase_if(
                    None,
                    f"{elmer_dir}/results/save_line_melt_crys.dat",
                    self.coupling_config["relaxation_factor"],
                )
            else:
                for file in ["save_line_crc_melt.dat", "save_line_melt_surf.dat"]:
                    self.relax_temperatures_heatfluxes(
                        f"{last_elmer_dir}/results/{file}",
                        f"{elmer_dir}/results/{file}",
                        self.coupling_config["relaxation_factor"],
                    )
                new_interface = self.relax_phase_if(
                    old_interface,
                    f"{elmer_dir}/results/save_line_melt_crys.dat",
                    self.coupling_config["relaxation_factor"],
                )
                # crystal_surface, _ = compute_current_crystal_surface(
                #     **self.geo_config["crystal"]
                # )
                try:  # as applied in Si simulation with complex crystal shape
                    crystal_radius = self.geo_config["process_condition"]["crystal_radius"]
                except KeyError:
                    try:  # as applied in Sn simulation with simple cylindrical / conical crystal shape
                        crystal_radius = self.geo_config["crystal"]["r"]
                    except KeyError:  # as applied in CsI simulation with complex crystal shape
                        compute_current_crystal_surface(**self.geo_config["crystal"])[-1, 0]

                avg_change, max_change, y_new_0 = self.check_interface_convergence(
                    old_interface,
                    new_interface,
                    crystal_radius,
                    self.coupling_config["interface_number_of_points"],
                )
                print(f"Average change at phase interface: {avg_change:.3e}")
                print("***************************************")
                print(f"Maximum change at phase interface: {max_change:.3e}")
                print("***************************************")
                print(f"Deflection at r = 0: {y_new_0:.3e}")
                print("***************************************")
                with open(f"{self.sim_dir}/interface_convergence_avg.log", "a") as f:
                    f.write(f"{i+1:02}    {avg_change:.3e}\n")
                with open(f"{self.sim_dir}/interface_convergence_max.log", "a") as f:
                    f.write(f"{i+1:02}    {max_change:.3e}\n")
                with open(f"{self.sim_dir}/interface_deflection_r=0.log", "a") as f:
                    f.write(f"{i+1:02}    {y_new_0:.3e}\n")
                if (
                    max_change < self.coupling_config["tolerance_phase_interface"]
                    and openfoam_converged  # checked for steady state only
                    and last_openfoam_converged  # checked for steady state only
                    and i > max(self.coupling_config["flow_scaling"])
                    and i > max(self.coupling_config["heat_conductivity_scaling"])
                ):
                    coupling_converged = True
                    print("Converged!")
                    break
                old_interface = new_interface

            # run openFoam
            if self.coupling_config["OpenFOAM_3D"]:
                if not self.coupling_config["transient_flow"]:
                    raise ValueError("2D-3D-coupling only available with transient flow simulation.")
                shutil.copytree("./azimuthalAverage", foam_dir)
                self.setup_openfoam(
                    3,
                    f"{foam_dir}/3D-case",
                    f"{elmer_dir}/results",
                    self.setup_openfoam_config["template_3D"],
                    highres=False,
                )
                self.setup_openfoam(
                    2,
                    f"{foam_dir}/2D-case",
                    f"{elmer_dir}/results",
                    self.setup_openfoam_config["template_transient"],
                    highres=False,
                )

                foam_simdir = f"{foam_dir}/3D-case"
                last_foam_simdir = f"{last_foam_dir}/3D-case"
            elif self.coupling_config["transient_flow"]:
                self.setup_openfoam(
                    2,
                    foam_dir,
                    f"{elmer_dir}/results",
                    self.setup_openfoam_config["template_transient"],
                )
                foam_simdir = foam_dir
                last_foam_simdir = last_foam_dir                
            else:
                self.setup_openfoam(
                    2,
                    foam_dir,
                    f"{elmer_dir}/results",
                    self.setup_openfoam_config["template_steady"],
                )
                foam_simdir = foam_dir
                last_foam_simdir = last_foam_dir
            if (
                last_foam_dir is not None
            ):  # copy old results to initialize new simulation
                print("Obtaining last simulation results for initialization.")
                highest_number = 0
                for item in os.listdir(last_foam_simdir):
                    item_path = os.path.join(last_foam_simdir, item)
                    if os.path.isdir(item_path) and self.is_float(item):
                        number = float(item)
                        if number > highest_number:
                            highest_number = number
                            highest_directory = item_path
                print(f"Found results in '{highest_directory}'")
                os.remove(f"{foam_simdir}/0.orig/U")
                os.remove(f"{foam_simdir}/0.orig/T")
                shutil.copy(f"{highest_directory}/U", f"{foam_simdir}/0.orig/U")
                shutil.copy(f"{highest_directory}/T", f"{foam_simdir}/0.orig/T")
            print("Running OpenFOAM...")
            with open(os.path.join(foam_simdir, "allrun.log"), "w") as f:
                subprocess.run(["chmod", "+x", "Allrun"], cwd=foam_simdir)
                subprocess.run(["./Allrun"], cwd=foam_simdir, stdout=f, stderr=f)
            if self.coupling_config["OpenFOAM_3D"]:
                with open(os.path.join(foam_dir, "runAzimuthalAveraging.log"), "w") as f:
                    subprocess.run(["chmod", "+x", "runAzimuthalAveraging"], cwd=foam_dir)
                    subprocess.run(["./runAzimuthalAveraging"], cwd=foam_dir, stdout=f, stderr=f)
                foam_simdir = f"{foam_dir}/simdata-2D-averaged"
                last_foam_simdir = f"{last_foam_dir}/simdata-2D-averaged"
            print("Converting Foam to VTK...")
            with open(os.path.join(foam_dir, "f2vtk.log"), "w") as f:
                subprocess.run(
                    ["foamToVTK", "-latestTime", "-legacy", "-ascii"],
                    cwd=foam_simdir,
                    stdout=f,
                    stderr=f,
                )
            print("Converting VTK to msh...")
            files = os.listdir(f"{foam_simdir}/VTK")
            for file in files:
                if file.split(".")[-1] == "vtk":
                    vtk_file = file
            if i == 0:
                self.vtk_to_msh(f"{foam_simdir}/VTK/{vtk_file}", foam_simdir)
            else:
                self.vtk_to_msh_with_u_old(
                    f"{foam_simdir}/VTK/{vtk_file}",
                    f"{last_foam_simdir}/VTK/{last_vtk_file}",
                    foam_simdir,
                    flow_variable,
                )
            if not self.coupling_config["transient_flow"]:
                last_openfoam_converged = openfoam_converged
                openfoam_converged = self.is_converged(
                    f"{foam_dir}/log.buoyantBoussinesqSimpleFoam"
                )
                with open(f"{self.sim_dir}/openfoam_convergence.log", "a") as f:
                    f.write(f"{i+1:02}    {openfoam_converged}\n")

            phase_if = self.extract_phase_if(
                f"{elmer_dir}/results/save_line_melt_crys_relaxed.dat"
            )
            self.geo_config.update(phase_if)
        err, warn, stats = scan_logfile(elmer_dir)
        with open(self.res_dir + "/elmer_summary.yml", "w") as f:
            yaml.dump(
                {"Errors": err, "Warnings": warn, "Statistics": stats},
                f,
                sort_keys=False,
            )
        if coupling_converged:
            with open(f"{self.res_dir}/CONVERGED", 'w') as file: pass
        else:
            with open(f"{self.res_dir}/NOT_CONVERGED", 'w') as file: pass
        self._postprocessing_probes_2(elmer_dir)

    def _postprocessing_probes_2(self, sim_dir=None, res_dir=None):
        # this is a re-implementation of _postprocessing_probes, supposed to be more stable
        if sim_dir is None:
            sim_dir = self.sim_dir
        if res_dir is None:
            res_dir = self.res_dir

        with open(sim_dir + "/probes.yml") as f:
            probes = list(yaml.safe_load(f).keys())
        df = dat_to_dataframe(sim_dir + "/results/probes.dat")

        new_names_dict = {}
        probe_idx = -1
        for old_name in df.columns.tolist():
            if "value" in old_name:
                if "coordinate 1" in old_name:  # assumes same probe sorting as in yaml-file
                    probe_idx += 1
                new_name = probes[probe_idx] + " " + old_name[7:].split(" in element ")[0]
            if "res" in old_name:
                new_name = "res " + old_name[5:]
            new_names_dict[old_name] = new_name
        df.rename(columns=new_names_dict, inplace=True)

        # keep only temperature and "res" columns
        columns_list = []
        for column in df.columns.tolist():
            if "temperature" in column and not "temperature " in column:
                columns_list.append(column)
            if "res " in column:
                columns_list.append(column)
        df = df[columns_list]

        data = {}
        for column in df.iteritems():
            data.update({column[0]: float(column[1].iloc[-1])})
        if sim_dir == res_dir:
            output_name = "probes_result"
        else:
            output_name = "probes"
        with open(f"{res_dir}/{output_name}.yml", "w") as f:
            yaml.dump(data, f)
        self.probe_data = data
        df.to_csv(f"{res_dir}/{output_name}.csv", index=False, sep=";")
