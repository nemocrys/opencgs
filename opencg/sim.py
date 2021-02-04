from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import numpy as np
import os
import pandas as pd
import shutil
import yaml

from opencg.post import HeatfluxSurf

import pyelmer.elmer as elmer
from pyelmer.execute import run_elmer_solver, run_elmer_grid
from pyelmer.post import scan_logfile

SOLVER_FILE = os.path.dirname(os.path.realpath(__file__)) + '/data/solvers.yml'
MATERIAL_FILE = os.path.dirname(os.path.realpath(__file__)) + '/data/materials.yml'


class ElmerSetupCz:
    def __init__(self, heat_control, heat_convection, heating_induction, phase_change, transient,
                 heating, sim_dir, v_pull=0, smart_heater={}, probes={}, transient_setup={}):
        self.heat_control = heat_control
        self.heat_convection = heat_convection
        self.heating_induction = heating_induction
        self.phase_change = phase_change
        self.transient = transient
        self.sim_dir = sim_dir
        self.v_pull = v_pull
        if phase_change:
            self.mesh_update = True
        elif transient:
            self.mesh_update = True
        else:
            self.mesh_update = False

        if transient:
            self.sim = self._setup_transient_sim(**transient_setup)
            try:
                self.smart_heater_t = transient_setup['smart_heater_t']
            except KeyError:
                self.smart_heater_t = 1
        else:
            self.sim = elmer.load_simulation('axi-symmetric_steady')

        self.heating = heating
        self.probes = probes
        self._set_equations()

        if self.heat_control and smart_heater == {}:
            raise ValueError('The smart heater settings are missing!')
        if self.heating_induction:
            self.smart_heater = smart_heater
            self._set_joule_heat()
        else:
            pass # TODO resistance heating body force

        self._crystal = None
        self._heat_flux_dict = {}

    def __getitem__(self, name):
        for body in self.sim.bodies:
            if body == name:
                return self.sim.bodies[name]
        for boundary in self.sim.boundaries:
            if boundary == name:
                return self.sim.boundaries[name]

    def _set_equations(self):
        # solvers
        if self.heating_induction:
            omega = 2 * np.pi * self.heating['frequency']
            solver_statmag = elmer.load_solver('StatMagSolver', self.sim, SOLVER_FILE)
            solver_statmag.data.update({'Angular Frequency': omega})
        solver_heat = elmer.load_solver('HeatSolver', self.sim, SOLVER_FILE)
        if self.transient and self.heat_control:
            solver_heat.data.update({'Smart Heater Time Scale': self.smart_heater_t})
        if self.phase_change:
            if self.transient:
                solver_phase_change = elmer.load_solver('TransientPhaseChange', self.sim, SOLVER_FILE)
            else:
                solver_phase_change = elmer.load_solver('SteadyPhaseChange', self.sim, SOLVER_FILE)
                solver_phase_change.data['Triple Point Fixed'] = 'Logical True'
        if self.mesh_update:
            solver_mesh = elmer.load_solver('MeshUpdate', self.sim, SOLVER_FILE)
        if self.probes != {}:
            solver_probe_scalars = elmer.load_solver('probe-scalars', self.sim, SOLVER_FILE)
            n = 0
            point_str = 'Real '
            for key in self.probes:
                if n != 0:
                    point_str += ' \\\n    '
                point_str += f'{self.probes[key][0]} {self.probes[key][1]}'
                n += 1
            solver_probe_scalars.data.update({f'Save Coordinates({n},2)': point_str})
        if self.transient:
            elmer.load_solver('Mesh2Mesh', self.sim, SOLVER_FILE)
        elmer.load_solver('SaveMaterials', self.sim, SOLVER_FILE)
        elmer.load_solver('ResultOutputSolver', self.sim, SOLVER_FILE)
        elmer.load_solver('SaveLine', self.sim, SOLVER_FILE)

        # equations
        if self.heating_induction:
            equation_main = elmer.Equation(self.sim, 'equation_main', [solver_statmag, solver_heat])
        else:
            equation_main = elmer.Equation(self.sim, 'main_equation', [solver_heat])
        if self.transient or self.phase_change:
            equation_main.solvers.append(solver_mesh)
        self._eqn_main = equation_main
        if self.phase_change:
            equation_phase_change = elmer.Equation(self.sim, 'equation_phase_change',
                                                   [solver_phase_change])
            self._eqn_phase_change = equation_phase_change
        else:
            self._eqn_phase_change = None

    def _set_joule_heat(self):
        joule_heat = elmer.BodyForce(self.sim, 'joule_heat')
        joule_heat.joule_heat = True
        if self.heat_control:
            joule_heat.smart_heat_control = True
            if self.smart_heater['control-point']:
                joule_heat.smart_heater_control_point = [self.smart_heater['x'],
                                                         self.smart_heater['y'],
                                                         self.smart_heater['z']]
                joule_heat.smart_heater_T = self.smart_heater['T']
        self._joule_heat = joule_heat

    def _setup_transient_sim(self, dt, dt_out, t_max, smart_heater_t, restart=False,
                             restart_file='', restart_time=0):
        sim = elmer.load_simulation('axi-symmetric_transient')
        # TODO implement restart in pyelmer
        sim.settings.update({'Timestep Sizes': dt})
        sim.settings.update({'Output Intervals': round(dt_out / dt)})
        sim.settings.update({'Timestep Intervals': round(t_max / dt)})
        if restart:
            # sim.settings.update({'Restart File': restart_file})
            # sim.settings.update({'Restart Variable 1': 'Temperature'})
            # sim.settings.update({'Restart Variable 2': 'PhaseSurface'})
            sim.settings.update({'Restart Time': restart_time})
            sim.settings.update({'Restart Error Continue': 'Logical True',
                                 'Use Mesh Projector': 'Logical False'})
        return sim

    @property
    def joule_heat(self):
        return self._joule_heat

    @property
    def movement(self):
        if self.transient:
            return [0, f'Variable Time\n    real MATC "{self.v_pull / 6e4 }*tx"']
        else:
            return [0, 0]

    @property
    def distortion(self):
        return [0, None]

    def add_crystal(self, shape, material='', force=None):
        self._crystal = self.add_body(shape, material, force)

    def add_inductor(self, shape, material=''):
        self._current = elmer.BodyForce(self.sim, 'Current Density')
        self._current.current_density = self.heating['current'] / shape.params.area
        self._inductor = self.add_body(shape, material, self._current)

    def add_body(self, shape, material='', force=None):
        body = elmer.Body(self.sim, shape.name, [shape.ph_id])
        if material == '':
            material = self.add_material(shape.params.material)
        body.material = material
        body.equation = self._eqn_main
        body.initial_condition = elmer.InitialCondition(self.sim, f'T-{shape.name}',
                                                        {'Temperature': shape.params.T_init})
        if force is not None:
            body.body_force = force
        return body

    def add_radiation_boundary(self, shape, movement=[0, 0], htc=0, T_ext=293.15, 
                               rad_s2s=True):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        if rad_s2s:
            boundary.radiation = True
        else:
            boundary.radiation_idealized = True
            boundary.T_ext = T_ext
        if htc != 0 and self.heat_convection:
            boundary.heat_transfer_coefficient = htc
            boundary.T_ext = T_ext
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def add_temperature_boundary(self, shape, T, movement=[0, 0]):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.fixed_temperature = T
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def add_heatflux_boundary(self, shape, heatflux, movement=[0, 0]):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.fixed_heatflux = heatflux
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def add_material(self, name, setup_file=''):
        if name in self.sim.materials:
            return self.sim.materials[name]
        if setup_file == '':
            setup_file = MATERIAL_FILE
        material = elmer.load_material(name, self.sim, setup_file)
        # remove properties not required for simulation
        if 'Beta' in material.data:
            material.data.pop('Beta')
        if 'Surface Tension' in material.data:
            material.data.pop('Surface Tension')
        return material
         
    def add_phase_interface(self, shape, crystal):
        if not self.phase_change:
            raise ValueError('This Simulation does not include a phase change model.')
        if self._eqn_phase_change is None:
            raise ValueError('Equation is not initialized.')
        if self._crystal is None:
            raise ValueError('No crystal was added to the model.')
        # Body for phase change solver
        phase_if = elmer.Body(self.sim, shape.name, [shape.ph_id])
        phase_if.material = self._crystal.material
        phase_if.equation = self._eqn_phase_change
        phase_if.initial_condition = elmer.InitialCondition(self.sim, 't0_phase_change')
        phase_if.initial_condition.data = {'Temperature': self._crystal.material.data['Melting Point'],
                                           'PhaseSurface': 'Real 0.0'}
        # Boundary condition
        bc_phase_if = elmer.Boundary(self.sim, 'melt_crystal_if', [shape.ph_id])
        bc_phase_if.save_line = True
        bc_phase_if.normal_target_body = self.sim.bodies[crystal.name]
        if self.heat_control and not self.smart_heater['control-point']:
            bc_phase_if.smart_heater = True
            bc_phase_if.smart_heater_T = self.smart_heater['T']
        if self.phase_change:
            if self.transient:
                bc_phase_if.phase_change_transient = True
            else:
                bc_phase_if.phase_change_steady = True
            bc_phase_if.phase_change_vel = self.v_pull / 6e4  # mm/min to m/s
            bc_phase_if.material = self._crystal.material
            bc_phase_if.phase_change_body = phase_if

    def add_interface(self, shape, movement=[0, 0]):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        if self.mesh_update:
            boundary.mesh_update = movement

    def export(self):
        self.sim.write_startinfo(self.sim_dir)
        self.sim.write_sif(self.sim_dir)
        self.sim.write_boundary_ids(self.sim_dir)
        with open(self.sim_dir + '/post_processing.yml', 'w') as f:
            yaml.dump(self._heat_flux_dict, f)
        with open(self.sim_dir + '/probes.yml', 'w') as f:
            yaml.dump(self.probes, f, sort_keys=False)

    def heat_flux_computation(self, body, boundary):
        # TODO that's a way too complicated!
        hfs = HeatfluxSurf(boundary.surface_ids[0], body.body_ids, body.material.data['Heat Conductivity'])
        self._heat_flux_dict.update({f'{body.name}_{boundary.name}': asdict(hfs)})


class Simulation:
    def __init__(self, geo, geo_config, sim, sim_config, config_update, sim_name, sim_type,
                 base_dir, with_date=True):
        self.geo = geo
        if 'geometry' in config_update:
            geo_config = self._update_config(geo_config, config_update['geometry'])
        self.geo_config = geo_config
        self.sim = sim
        if 'simulation' in config_update:
            sim_config = self._update_config(sim_config, config_update['simulation'])
        self.sim_config = sim_config
        self.sim_name = sim_name
        self.sim_type = sim_type
        self.sim_dir = self._create_directories(base_dir, with_date)
        self.elmer_dir = f'{self.sim_dir}/02_simulation'
        self.results_file = f'{self.elmer_dir}/case.result'

    def _create_directories(self, base_dir, with_date):
        if with_date:
            sim_dir = f'{base_dir}/{datetime.now():%Y-%m-%d_%H-%M}_{self.sim_type}_{self.sim_name}'
        else:
            sim_dir = f'{base_dir}/{self.sim_type}_{self.sim_name}'
        if os.path.exists(sim_dir):
            i = 1
            sim_dir += '_1'
            while os.path.exists(sim_dir):
                i += 1
                sim_dir = '_'.join(sim_dir.split('_')[:-1]) + f'_{i}'
        os.makedirs(sim_dir)
        os.mkdir(f'{sim_dir}/01_input')
        os.mkdir(f'{sim_dir}/02_simulation')
        os.mkdir(f'{sim_dir}/03_results')
        os.mkdir(f'{sim_dir}/04_plots')
        return sim_dir

    def _create_setup(self, visualize=False):
        # TODO copy / write setup to input directory
        model = self.geo(self.geo_config, self.elmer_dir, self.sim_name, visualize)
        self.sim(model, self.sim_config, self.elmer_dir)
    
    def execute(self):
        print('Starting simulation ', self.elmer_dir, ' ...')
        run_elmer_grid(self.elmer_dir, self.sim_name + '.msh')
        run_elmer_solver(self.elmer_dir)
        # post_processing(sim_path)
        err, warn, stats = scan_logfile(self.elmer_dir)
        print(err, warn, stats)
        print('Finished simulation ', self.elmer_dir, ' .')

    @staticmethod
    def _update_config(base_config, config_update):
        if config_update is not None:
            for param, update in config_update.items():
                if type(update) is dict:
                    base_config[param] = Simulation._update_config(base_config[param], config_update[param])
                else:
                    base_config[param] = update
        return base_config
    
    @property
    def vtu_files(self):
        return [f for f in os.listdir(self.elmer_dir) if f.split('.')[-1] == 'vtu']

    @property
    def last_interation(self):
        return max([int(f[-8:-4]) for f in self.vtu_files])

    @property
    def phase_interface(self):
        header = ['iteration', 'BC', 'NodeIdx', 'x', 'y', 'z', 'T']
        df = pd.read_table(f'{self.elmer_dir}/results/phase-if.dat', names=header, sep= ' ',
                           skipinitialspace=True)
        return [[line['x'], line['y']] for _, line in df.iterrows()]
    
    @property
    def mesh_files(self):
        files = []
        for f in os.listdir(self.elmer_dir):
            e = f.split('.')[-1]
            if e == 'boundary' or e == 'elements' or e == 'header' or e == 'nodes':
                files.append(f)
        return files
            

class SteadyStateSim(Simulation):
    def __init__(self, geo, geo_config, sim, sim_config, config_update={}, sim_name='opencgs-simulation',
                 base_dir='./simdata', visualize=False, with_date=True, **kwargs):
        super().__init__(geo, geo_config, sim, sim_config, config_update, sim_name, 'ss', base_dir, with_date)
        self.sim_config['general']['transient'] = False
        self._create_setup(visualize)

    def post():
        pass

class TransientSim(Simulation):
    def __init__(self, geo, geo_config, sim, sim_config, config_update={}, sim_name='opencgs-simulation',
                 base_dir='./simdata', l_start = 0.01, l_end = 0.1, **kwargs):
        super().__init__(geo, geo_config, sim, sim_config, config_update, sim_name, 'st', base_dir)
        self.l_start = l_start
        self.l_end = l_end

    def execute(self):
        print('Starting with steady state simulation')
        geo_config = deepcopy(self.geo_config)
        geo_config['crystal']['l'] = self.l_start
        sim_config = deepcopy(self.sim_config)
        sim_config['smart-heater']['control-point'] = True
        sim = SteadyStateSim(self.geo, geo_config, self.sim, sim_config,
                             sim_name='initialization', base_dir=self.elmer_dir, with_date=False)
        sim.execute()
        current_l = self.l_start
        old = sim
        i = 0
        # while current_l < self.l_start:
        t_end = 10
        sim = TransientSubSim(old, t_end, self.geo, deepcopy(geo_config), self.sim,
                                deepcopy(self.sim_config), sim_name=f'iteration_{i}',
                                base_dir=self.elmer_dir)
        sim.execute()
        old = sim
        i += 1
            # params: length, if-shape, time, start-index

class TransientSubSim(Simulation):
    def __init__(self, old, t_end, geo, geo_config, sim, sim_config, sim_name, base_dir):
        super().__init__(geo, geo_config, sim, sim_config, {}, sim_name, 'ts', base_dir, False)
        self.sim_config['general']['transient'] = True
        self.sim_config['transient']['t_max'] = t_end
        self.sim_config['transient']['restart'] = True
        self.sim_config['transient']['restart_file'] = 'restart.result'
        self.sim_config['transient']['restart_time'] = old.last_interation
        self.geo_config['phase_if'] = old.phase_interface
        os.mkdir(f'{self.elmer_dir}/old_mesh')
        shutil.copy2(old.results_file, f'{self.elmer_dir}/old_mesh/restart.result')
        for f in old.mesh_files:
            shutil.copy2(f'{old.elmer_dir}/{f}', f'{self.elmer_dir}/old_mesh/{f}')
        self._create_setup()

