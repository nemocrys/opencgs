import numpy as np
import pyelmer.elmer as elmer


SOLVER_FILE = './restruct/data/solvers.yml'
MATERIAL_FILE = './restruct/data/materials.yml'


class ElmerSimulation:
    def __init__(self, config, transient=False, phase_change=True, heat_control=True):
        self.config = config
        self.transient = transient
        self.phase_change = phase_change
        self.heat_control = heat_control

        self._joule_heat = None
        self._eqn_phase_change = None
        self._eqn_main = None
        self._crystal = None
        self._material_crystal = None

        if self.transient:
            self.sim = elmer.load_simulation('axi-symmetric_transient')
        else:
            self.sim = elmer.load_simulation('axi-symmetric_steady')

    def set_equations(self, heat=True, induction=True, probes=True):
        if heat:
            solver_heat = elmer.load_solver('HeatSolver', self.sim, SOLVER_FILE)
        if induction:
            omega = 2 * np.pi * self.config['induction']['frequency']
            solver_statmag = elmer.load_solver('StatMagSolver', self.sim, SOLVER_FILE)
            solver_statmag.data.update({'Angular Frequency': omega})
        if self.phase_change:
            if self.transient:
                solver_phase_change = elmer.Solver('TransientPhaseChange', self.sim, SOLVER_FILE)
            else:
                solver_phase_change = elmer.load_solver('SteadyPhaseChange', self.sim, SOLVER_FILE)
                if self.config['phase-change']['fixed-tp']:
                    solver_phase_change.data['Triple Point Fixed'] = 'Logical True'
                else:
                    solver_phase_change.data['Triple Point Fixed'] = 'Logical False'
        if self.phase_change or self.transient:
            solver_mesh = elmer.load_solver('MeshUpdate', self.sim, SOLVER_FILE)
        if probes:
            solver_probe_scalars = elmer.load_solver('probe-scalars', self.sim, SOLVER_FILE)
            n = 0
            point_str = 'Real '
            for key in self.config['probes']:
                if n != 0:
                    point_str += ' \\\n    '
                point_str += f'{self.config["probes"][key][0]} {self.config["probes"][key][1]}'
                n += 1
            solver_probe_scalars.data.update({f'Save Coordinates({n},2)': point_str})

        elmer.Solver(self.sim, 'SaveMaterials', self.config['solvers']['SaveMaterials'])
        elmer.Solver(self.sim, 'ResultOutputSolver', self.config['solvers']['ResultOutputSolver'])
        elmer.Solver(self.sim, 'boundary_scalars', self.config['solvers']['boundary-scalars'])

        if induction:
            equation_main = elmer.Equation(self.sim, 'equation_main', [solver_statmag, solver_heat])
        else:
            equation_main = elmer.Equation(self.sim, 'main_equation', [solver_heat])
        if self.transient or self.phase_change:
            equation_main.solvers.append(solver_mesh)
        if self.phase_change:
            equation_phase_change = elmer.Equation(self.sim, 'equation_phase_change',
                                                   [solver_phase_change])
            self._eqn_phase_change = equation_phase_change
        self._eqn_main = equation_main
        
    @property
    def joule_heat(self):
        if self._joule_heat is None:
            joule_heat = elmer.BodyForce(self.sim, 'joule_heat')
            joule_heat.joule_heat = True
            if self.config['settings']['smart-heater']:
                joule_heat.smart_heat_control = True
                if self.config['smart-heater']['control-point']:
                    joule_heat.smart_heater_control_point = [self.config['smart-heater']['x'],
                                                            self.config['smart-heater']['y'],
                                                            self.config['smart-heater']['z']]
                    joule_heat.smart_heater_T = self.config['smart-heater']['T']
            self._joule_heat = joule_heat
        return self._joule_heat

    def add_current(self, inductor):
        current = elmer.BodyForce(self.sim, 'Current Density')
        current.current_density = self.config['induction']['current'] / inductor.params.area
        return current

    def add_crystal(self, shape, material, force=None):
        self._crystal = self.add_body(shape, material, force)
        self._material_crystal = material

    def add_body(self, shape, material, force=None):
        body = elmer.Body(self.sim, shape.name, [shape.ph_id])
        body.material = material
        body.equation = self._eqn_main
        body.initial_condition = elmer.InitialCondition(self.sim, f'T-{shape.name}',
                                                        {'Temperature': shape.params.T_init})
        if force is not None:
            body.body_force = force
        return body

    def add_radiation_boundary(self, shape, mesh_update=[0, 0]):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.radiation = True
        boundary.save_scalars = True
        if self.transient or self.phase_change:
            boundary.mesh_update = mesh_update
        return boundary

    def add_temperature_boundary(self, shape, T, mesh_update=[0, 0]):
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.save_scalars = True
        boundary.fixed_temperature = T
        if self.transient or self.phase_change:
            boundary.mesh_update = mesh_update
        return boundary

    def add_material(self, name, setup_file=''):
        if setup_file == '':
            setup_file = MATERIAL_FILE
        return elmer.load_material(name, self.sim, setup_file)

    def add_phase_interface(self, shape):
        if not self.phase_change:
            raise ValueError('This Simulation does not include a phase change model.')
        if self._eqn_phase_change is None:
            raise ValueError('Equation is not initialized.')
        if self._crystal is None:
            raise ValueError('No crystal was added to the model.')
        # Body for phase change solver
        phase_if = elmer.Body(self.sim, shape.name, [shape.ph_id])
        phase_if.material = self._material_crystal
        phase_if.equation = self._eqn_phase_change
        phase_if.initial_condition = elmer.InitialCondition(self.sim, 't0_phase_change')
        phase_if.initial_condition.data = {'Temperature': self._material_crystal.data['Melting Point'],
                                           'PhaseSurface': 'Real 0.0'}
        # Boundary condition
        bc_phase_if = elmer.Boundary(self.sim, 'melt_crystal_if', [shape.ph_id])
        bc_phase_if.save_scalars = True
        if self.heat_control and not self.config['smart-heater']['control-point']:
            bc_phase_if.smart_heater = True
            bc_phase_if.smart_heater_T = self.config['smart-heater']['T']
        if self.phase_change:
            if self.transient:
                bc_phase_if.phase_change_transient = True
            else:
                bc_phase_if.phase_change_steady = True
            bc_phase_if.phase_change_vel = self.config['settings']['v-pull'] / 6e4  # mm/min to m/s
            bc_phase_if.material = self._material_crystal
            bc_phase_if.normal_target_body = self._crystal #TODO
            bc_phase_if.phase_change_body = phase_if

    def export(self):
        self.sim.write_startinfo('./simdata/_test/')
        self.sim.write_sif('./simdata/_test/')
        self.sim.write_boundary_ids('./simdata/_test/')
