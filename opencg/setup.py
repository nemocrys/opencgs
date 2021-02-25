from dataclasses import asdict
import numpy as np
import os
import yaml

from opencg.post import HeatfluxSurf
import pyelmer.elmer as elmer


SOLVER_FILE = os.path.dirname(os.path.realpath(__file__)) + "/data/solvers.yml"
MATERIAL_FILE = os.path.dirname(os.path.realpath(__file__)) + "/data/materials.yml"


class ElmerSetupCz:
    def __init__(
        self,
        heat_control,
        heat_convection,
        heating_induction,
        phase_change,
        transient,
        heating,
        sim_dir,
        v_pull=0,
        smart_heater={},
        probes={},
        transient_setup={},
    ):
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
                self.smart_heater_t = transient_setup["smart_heater_t"]
            except KeyError:
                self.smart_heater_t = transient_setup["dt"]
        else:
            self.sim = elmer.load_simulation("axi-symmetric_steady")
            self.sim.settings.update({"Output Intervals": 0})

        self.heating = heating
        self.probes = probes
        self._set_equations()

        if self.heat_control and smart_heater == {}:
            raise ValueError("The smart heater settings are missing!")
        if self.heating_induction:
            self.smart_heater = smart_heater
            self._set_joule_heat()
        else:
            pass  # TODO resistance heating body force

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
            omega = 2 * np.pi * self.heating["frequency"]
            solver_statmag = elmer.load_solver("StatMagSolver", self.sim, SOLVER_FILE)
            solver_statmag.data.update({"Angular Frequency": omega})
        solver_heat = elmer.load_solver("HeatSolver", self.sim, SOLVER_FILE)
        if self.transient and self.heat_control:
            solver_heat.data.update({"Smart Heater Time Scale": self.smart_heater_t})
        if self.phase_change:
            if False:
                # if self.transient: TODO
                solver_phase_change = elmer.load_solver(
                    "TransientPhaseChange", self.sim, SOLVER_FILE
                )
            else:
                solver_phase_change = elmer.load_solver(
                    "SteadyPhaseChange", self.sim, SOLVER_FILE
                )
                solver_phase_change.data["Triple Point Fixed"] = "Logical True"
        if self.mesh_update:
            solver_mesh = elmer.load_solver("MeshUpdate", self.sim, SOLVER_FILE)
        if self.probes != {}:
            solver_probe_scalars = elmer.load_solver(
                "probe-scalars", self.sim, SOLVER_FILE
            )
            n = 0
            point_str = "Real "
            for key in self.probes:
                if n != 0:
                    point_str += " \\\n    "
                point_str += f"{self.probes[key][0]} {self.probes[key][1]}"
                n += 1
            solver_probe_scalars.data.update({f"Save Coordinates({n},2)": point_str})
        if self.transient:
            # elmer.load_solver('Mesh2Mesh', self.sim, SOLVER_FILE)
            elmer.load_solver("gmsh-input", self.sim, SOLVER_FILE)
        elmer.load_solver("SaveMaterials", self.sim, SOLVER_FILE)
        elmer.load_solver("ResultOutputSolver", self.sim, SOLVER_FILE)
        elmer.load_solver("SaveLine", self.sim, SOLVER_FILE)
        elmer.load_solver("gmsh-output", self.sim, SOLVER_FILE)

        # equations
        if self.heating_induction:
            equation_main = elmer.Equation(
                self.sim, "equation_main", [solver_statmag, solver_heat]
            )
        else:
            equation_main = elmer.Equation(self.sim, "main_equation", [solver_heat])
        if self.transient or self.phase_change:
            equation_main.solvers.append(solver_mesh)
        self._eqn_main = equation_main
        if self.phase_change:
            equation_phase_change = elmer.Equation(
                self.sim, "equation_phase_change", [solver_phase_change]
            )
            self._eqn_phase_change = equation_phase_change
        else:
            self._eqn_phase_change = None

    def _set_joule_heat(self):
        joule_heat = elmer.BodyForce(self.sim, "joule_heat")
        joule_heat.joule_heat = True
        if self.heat_control:
            joule_heat.smart_heat_control = True
            if self.smart_heater["control-point"]:
                joule_heat.smart_heater_control_point = [
                    self.smart_heater["x"],
                    self.smart_heater["y"],
                    self.smart_heater["z"],
                ]
                joule_heat.smart_heater_T = self.smart_heater["T"]
        self._joule_heat = joule_heat

    def _setup_transient_sim(self, dt, dt_out, t_max):
        sim = elmer.load_simulation("axi-symmetric_transient")
        sim.settings.update({"Timestep Sizes": dt})
        sim.settings.update({"Output Intervals": round(dt_out / dt)})
        sim.settings.update({"Timestep Intervals": round(t_max / dt)})
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

    def add_crystal(self, shape, material="", force=None):
        self._crystal = self.add_body(shape, material, force)

    def add_inductor(self, shape, material=""):
        self._current = elmer.BodyForce(self.sim, "Current Density")
        self._current.current_density = self.heating["current"] / shape.params.area
        self._inductor = self.add_body(shape, material, self._current)

    def add_body(self, shape, material="", force=None):
        body = elmer.Body(self.sim, shape.name, [shape.ph_id])
        if material == "":
            material = self.add_material(shape.params.material)
        body.material = material
        body.equation = self._eqn_main
        body.initial_condition = elmer.InitialCondition(
            self.sim, f"T-{shape.name}", {"Temperature": shape.params.T_init}
        )
        if force is not None:
            body.body_force = force
        return body

    def add_radiation_boundary(
        self, shape, movement=[0, 0], htc=0, T_ext=293.15, rad_s2s=True
    ):
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

    def add_material(self, name, setup_file=""):
        if name in self.sim.materials:
            return self.sim.materials[name]
        if setup_file == "":
            setup_file = MATERIAL_FILE
        material = elmer.load_material(name, self.sim, setup_file)
        # remove properties not required for simulation
        if "Beta" in material.data:
            material.data.pop("Beta")
        if "Surface Tension" in material.data:
            material.data.pop("Surface Tension")
        return material

    def add_phase_interface(self, shape, crystal):
        if not self.phase_change:
            raise ValueError("This Simulation does not include a phase change model.")
        if self._eqn_phase_change is None:
            raise ValueError("Equation is not initialized.")
        if self._crystal is None:
            raise ValueError("No crystal was added to the model.")
        # Body for phase change solver
        phase_if = elmer.Body(self.sim, shape.name, [shape.ph_id])
        phase_if.material = self._crystal.material
        phase_if.equation = self._eqn_phase_change
        phase_if.initial_condition = elmer.InitialCondition(self.sim, "t0_phase_change")
        phase_if.initial_condition.data = {
            "Temperature": self._crystal.material.data["Melting Point"],
            "PhaseSurface": "Real 0.0",
        }
        # Boundary condition
        bc_phase_if = elmer.Boundary(self.sim, "melt_crystal_if", [shape.ph_id])
        bc_phase_if.save_line = True
        bc_phase_if.normal_target_body = self.sim.bodies[crystal.name]
        if self.heat_control and not self.smart_heater["control-point"]:
            bc_phase_if.smart_heater = True
            bc_phase_if.smart_heater_T = self.smart_heater["T"]
        if self.phase_change:
            if False:
                # if self.transient: # TODO
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
        with open(self.sim_dir + "/post_processing.yml", "w") as f:
            yaml.dump(self._heat_flux_dict, f)
        with open(self.sim_dir + "/probes.yml", "w") as f:
            yaml.dump(self.probes, f, sort_keys=False)

    def heat_flux_computation(self, body, boundary):
        # TODO that's a way too complicated!
        hfs = HeatfluxSurf(
            boundary.surface_ids[0],
            body.body_ids,
            body.material.data["Heat Conductivity"],
        )
        self._heat_flux_dict.update({f"{body.name}_{boundary.name}": asdict(hfs)})
