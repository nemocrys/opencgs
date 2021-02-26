from copy import deepcopy
from datetime import datetime
import inspect
import multiprocessing
import numpy as np
import os
import pandas as pd
import platform
import shutil
import yaml

import opencgs
from opencgs import post
import pyelmer
from pyelmer.execute import run_elmer_solver, run_elmer_grid
from pyelmer.post import scan_logfile


class Simulation:
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        config_update,
        sim_name,
        sim_type,
        base_dir,
        with_date=True,
    ):
        self.geo = geo
        if "geometry" in config_update:
            geo_config = self._update_config(geo_config, config_update["geometry"])
        self.geo_config = geo_config
        self.sim = sim
        if "simulation" in config_update:
            sim_config = self._update_config(sim_config, config_update["simulation"])
        self.sim_config = sim_config
        self.sim_name = sim_name
        self.sim_type = sim_type
        self._create_directories(base_dir, with_date)
        self._archive_input()
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
        self.sim(model, self.sim_config, self.sim_dir)

    def _archive_input(self):
        with open(self.input_dir + "/geo.yml", "w") as f:
            yaml.dump(self.geo_config, f)
        with open(self.input_dir + "/sim.yml", "w") as f:
            yaml.dump(self.sim_config, f)
        geo_file = inspect.getfile(self.geo)
        sim_file = inspect.getfile(self.sim)
        if geo_file == sim_file:
            shutil.copy2(geo_file, self.input_dir + "/setup.py")
        else:
            shutil.copy2(geo_file, self.input_dir + "/setup_geo.py")
            shutil.copy2(geo_file, self.input_dir + "/setup_sim.py")
        metadada = {"opencgs": opencgs.__version__, "pyelmer": pyelmer.__version__}
        with open(self.input_dir + "/metadata.yml", "w") as f:
            yaml.dump(metadada, f)

    @staticmethod
    def _read_names_file(names_file, skip_rows=7):  # TODO move to pyelmer
        with open(names_file) as f:
            data = f.readlines()
        data = data[7:]  # remove header
        for i in range(len(data)):
            data[i] = data[i][6:-1]  # remove index and \n
        return data

    def _postprocessing_probes(self):  # TODO copied from sim-elmerthermo. Dublecheck!
        with open(self.sim_dir + "/probes.yml") as f:
            probes = list(yaml.safe_load(f).keys())
        names_data = self._read_names_file(self.sim_dir + "/results/probes.dat.names")
        values = []
        res = []
        for column in names_data:
            if "value: " in column:
                values.append(
                    column[7:].split(" in element ")[0]
                )  # remove 'value: ' and 'in element XX'
            if "res: " in column:
                res.append("res " + column[5:])
        values = values[: int(len(values) / len(probes))]  # remove duplicates
        header = []
        for probe in probes:
            for value in values:
                header.append(probe + " " + value)
        header_shortened = []
        for column in header:
            if "temperature" in column and not "loads" in column:
                header_shortened.append(column)
        for column in header:
            if "magnetic flux density" in column and "MF" in column:
                header_shortened.append(column)
        header += res
        header_shortened += res
        df = pd.read_table(
            self.sim_dir + "/results/probes.dat",
            names=header,
            sep=" ",
            skipinitialspace=True,
        )
        df = df[header_shortened]

        data = {}
        for column in df.iteritems():
            data.update({column[0]: float(column[1].iloc[-1])})
        with open(self.res_dir + "/probes.yml", "w") as f:
            yaml.dump(data, f)
        self.probe_data = data
        df.to_csv(self.res_dir + "/probes.csv", index=False, sep=";")

    def execute(self):
        print("Starting simulation ", self.root_dir, " ...")
        run_elmer_grid(self.sim_dir, self.sim_name + ".msh")
        run_elmer_solver(self.sim_dir)
        # post_processing(sim_path)
        err, warn, stats = scan_logfile(self.sim_dir)
        print(err, warn, stats)
        print("Finished simulation ", self.root_dir, " .")
        print("Post processing...")
        self._postprocessing_probes()
        self.post()
        print("Finished post processing.")

    def post(self):
        """Simulation specific post processing."""
        pass

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
        return [f for f in os.listdir(self.sim_dir) if f.split(".")[-1] == "vtu"]

    @property
    def last_interation(self):
        return max([int(f[-8:-4]) for f in self.vtu_files])

    @property
    def phase_interface(self):
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
        files = []
        for f in os.listdir(self.sim_dir):
            e = f.split(".")[-1]
            if e == "boundary" or e == "elements" or e == "header" or e == "nodes":
                files.append(f)
        return files

    @property
    def last_heater_current(self):
        I_base = self.sim_config["heating_induction"]["current"]
        pwr_scaling = self.probe_data["res heater power scaling"]
        return I_base * pwr_scaling ** 0.5


class SteadyStateSim(Simulation):
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        config_update={},
        sim_name="opencgs-simulation",
        base_dir="./simdata",
        visualize=False,
        with_date=True,
        **_,
    ):
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            config_update,
            sim_name,
            "ss",
            base_dir,
            with_date,
        )
        self.sim_config["general"][
            "transient"
        ] = False  # TODO this may not appear in 01_input!
        self._create_setup(visualize)

    def post(self):
        print("evaluating heat fluxes")
        try:
            post.heat_flux(self.sim_dir, self.res_dir)
        except Exception as exc:
            print("Could not evaluate heat fluxes :(")
            print(exc)


class TransientSim(Simulation):
    def __init__(
        self,
        geo,
        geo_config,
        sim,
        sim_config,
        config_update={},
        sim_name="opencgs-simulation",
        base_dir="./simdata",
        l_start=0.01,
        l_end=0.1,
        d_l=3e-3,
        timesteps_per_dl=10,
        dt_out_factor=1,
        visualize=False,
        with_date=True,
        **_,
    ):
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            config_update,
            sim_name,
            "st",
            base_dir,
            with_date,
        )
        self.l_start = l_start
        self.l_end = l_end
        self.d_l = d_l
        self.v_pull = sim_config["general"]["v_pull"] / 6e4  # in m/s
        self.dt = d_l / self.v_pull / timesteps_per_dl
        print(f"Time step: {self.dt} s")
        if timesteps_per_dl % dt_out_factor != 0:
            raise ValueError("Bad dt_out_factor. Try again!")
        if dt_out_factor != 1:
            raise NotImplementedError(
                "The dt_out_factor option is not available yet :("
            )
        self.dt_out = self.dt * dt_out_factor
        print(f"Output time step: {self.dt_out} s")
        if "transient" not in self.sim_config:
            self.sim_config["transient"] = {}
        self.sim_config["transient"]["dt"] = self.dt
        self.sim_config["transient"]["dt_out"] = self.dt_out
        self.visualize = visualize
        self.vtu_idx = 0

    def execute(self):
        print("Initial steady state simulation...")
        geo_config = deepcopy(self.geo_config)
        geo_config["crystal"]["l"] = self.l_start
        sim_config = deepcopy(self.sim_config)
        sim_config["general"]["heat_control"] = True
        sim = SteadyStateSim(
            self.geo,
            geo_config,
            self.sim,
            sim_config,
            sim_name="initialization",
            base_dir=self.sim_dir,
            with_date=False,
            visualize=self.visualize,
        )
        sim.execute()
        self.collect_vtu_files(sim, only_last=True)
        print("Iterative transient simulation...")
        current_l = self.l_start
        old = sim
        i = 0
        while current_l < self.l_end:
            print("Starting new simulation. Current crystal length:", current_l)
            # compute end time
            t_end = 0
            l_sim_end = current_l
            while l_sim_end < min(current_l + self.d_l, self.l_end):
                t_end += self.dt
                l_sim_end += self.dt * self.v_pull
            sim_config = deepcopy(self.sim_config)
            sim_config["transient"]["t_max"] = t_end
            geo_config = deepcopy(self.geo_config)
            geo_config["crystal"]["l"] = current_l
            sim = TransientSubSim(
                old,
                t_end,
                self.geo,
                geo_config,
                self.sim,
                sim_config,
                sim_name=f"iteration_{i}",
                base_dir=self.sim_dir,
                visualize=self.visualize,
            )
            sim.execute()
            self.collect_vtu_files(sim)
            old = sim
            i += 1
            current_l = l_sim_end
            # params: length, if-shape, time, start-index

    def collect_vtu_files(self, sim, only_last=False):
        if only_last:
            vtus = [sim.vtu_files[-1]]
        else:
            vtus = sim.vtu_files
        for vtu in vtus:
            if self.vtu_idx < 10:
                prefix = "000"
            elif self.vtu_idx < 100:
                prefix = "00"
            elif self.vtu_idx < 1000:
                prefix = "0"
            else:
                prefix = ""
            name = f"case_t{prefix}{self.vtu_idx}.vtu"
            shutil.copy2(f"{sim.sim_dir}/{vtu}", f"{self.res_dir}/{name}")
            self.vtu_idx += 1


class TransientSubSim(Simulation):
    def __init__(
        self,
        old,
        t_end,
        geo,
        geo_config,
        sim,
        sim_config,
        sim_name,
        base_dir,
        visualize=False,
    ):
        super().__init__(
            geo, geo_config, sim, sim_config, {}, sim_name, "ts", base_dir, False
        )
        self.sim_config["general"]["transient"] = True
        self.sim_config["transient"]["t_max"] = t_end
        # self.sim_config['heating_induction']['current'] = old.last_heater_current
        self.geo_config["phase_if"] = old.phase_interface
        shutil.copy2(f"{old.sim_dir}/result.msh", f"{self.sim_dir}/input.msh")
        self._create_setup(visualize)


class ParameterStudy(Simulation):
    def __init__(
        self,
        SimulationClass,
        geo,
        geo_config,
        sim,
        sim_config,
        config_update,
        study_params,
        create_permutations=False,
        sim_name="opencgs-simulation",
        base_dir="./simdata",
        with_date=True,
        **_,
    ):
        if SimulationClass == SteadyStateSim:
            type_str = "ps"
        elif SimulationClass == TransientSim:
            type_str = "pt"
        else:
            raise TypeError(
                f"SimulationClass {SimulationClass} not supported in parameter study."
            )
        super().__init__(
            geo,
            geo_config,
            sim,
            sim_config,
            config_update,
            sim_name,
            type_str,
            base_dir,
            with_date=with_date,
        )
        with open(self.input_dir + "/study_params.yml", "w") as f:
            yaml.dump(study_params, f)
        self.sims = []
        self.create_sims(SimulationClass, study_params, create_permutations)

    def create_sims(self, SimulationClass, study_params, create_permutations):
        keys, vals = self._create_param_lists(study_params)
        print("updating the following parameters:", keys, vals)
        if create_permutations:
            # TODO create permutations!
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
                    sim = SimulationClass(
                        self.geo,
                        self.geo_config,
                        self.sim,
                        self.sim_config,
                        config_update,
                        sim_name,
                        self.sim_dir,
                        with_date=False,
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
        count = multiprocessing.cpu_count()
        # TODO not sure if that's a good idea...
        if platform.system() == "Windows":
            count -= 1
        print("Working on ", count, " cores.")
        pool = multiprocessing.Pool(processes=count)
        pool.map(ParameterStudy._execute_sim, self.sims)
        for sim in self.sims:
            for f in os.listdir(sim.res_dir):
                ext = f.split(".")[-1]
                if ext in ["yaml", "yml"]:
                    shutil.copy2(
                        f"{sim.res_dir}/{f}",
                        f"{self.res_dir}/{f[:-(len(ext) + 1)]}_{sim.sim_name}.{ext}",
                    )

    @staticmethod
    def _execute_sim(simulation):
        simulation.execute()
