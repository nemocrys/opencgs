from dataclasses import asdict
import numpy as np
import os
import yaml

from opencgs.post import HeatfluxSurf
import pyelmer.elmerkw as elmer

SIMULATION_FILE = os.path.dirname(os.path.realpath(__file__)) + "/data/simulations.yml"
SOLVER_FILE = os.path.dirname(os.path.realpath(__file__)) + "/data/solvers.yml"
MATERIAL_FILE = os.path.dirname(os.path.realpath(__file__)) + "/data/materials.yml"



class ElmerSetupCz:
    def __init__(
        self,
        heat_control,
        heat_convection,
        phase_change,
        heating,
        sim_dir,
        v_pull=0,
        heating_induction=False,
        heating_resistance=False,
        smart_heater={},
        probes={},
        solver_update={},
        materials_dict={},
    ):
        """Setup for Czochralski growth simulation with Elmer

        Args:
            heat_control (bool): modify heating to get the melting point
                temperature at the triple point.
            heat_convection (bool): include a heat convection model with
                heat transfer coefficients
            phase_change (bool): include latent heat release
            heating (dict): configuration of respective heating 
            sim_dir (str): simulation directory file path
            v_pull (int, optional): pulling velocity. Defaults to 0.
            heating_resistance (bool, optional): furnace with resistance
                heating. Defaults to False (= inductive heating ).
            smart_heater (dict, optional): heater control parameters.
                Defaults to {}.
            probes (dict, optional): probe locations where temperatures
                etc. are evaluated. Defaults to {}.
            solver_update (dict, optional): modified solver parameters.
                Defaults to {}.
            materials_dict (dict, optional): material properties.
                Defaults to {}.
        """

        self.heat_control = heat_control
        self.heat_convection = heat_convection
        self.phase_change = phase_change
        self.sim_dir = sim_dir
        self.v_pull = v_pull
        self.heating_induction = heating_induction
        self.heating_resistance = heating_resistance
        self.solver_update = solver_update
        self.materials_dict = materials_dict
        if phase_change:
            self.mesh_update = True
        else:
            self.mesh_update = False

        self.sim = elmer.load_simulation("axi-symmetric_steady", SIMULATION_FILE)
        self.sim.settings.update({"Output Intervals": 0})

        self.heating = heating
        self.probes = probes
        self._set_equations()

        if self.heat_control and smart_heater == {}:
            raise ValueError("The smart heater settings are missing!")
        if self.heating_induction:
            self.smart_heater = smart_heater
            self._set_joule_heat()
        if self.heating_resistance:
            if self.heating_induction:
                raise ValueError("Simultaneous incuction and resistance heating not supported.")
            self.smart_heater = smart_heater

        self._crystal = None
        self._heat_flux_dict = {}

    def __getitem__(self, name):
        for boundary in self.sim.boundaries:
            if boundary == name:
                return self.sim.boundaries[name]
        for body in self.sim.bodies:
            if body == name:
                return self.sim.bodies[name]

    def _set_equations(self):
        # solvers
        if self.heating_induction:
            omega = 2 * np.pi * self.heating["frequency"]
            self.solver_statmag = elmer.load_solver(
                "StatMagSolver", self.sim, SOLVER_FILE
            )
            self.solver_statmag.data.update({"Angular Frequency": omega})
        self.solver_heat = elmer.load_solver("HeatSolver", self.sim, SOLVER_FILE)
        if self.phase_change:
            self.solver_phase_change = elmer.load_solver(
                "SteadyPhaseChange", self.sim, SOLVER_FILE
            )
            self.solver_phase_change.data["Triple Point Fixed"] = "Logical True"
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
        elmer.load_solver("SaveMaterials", self.sim, SOLVER_FILE)
        elmer.load_solver("ResultOutputSolver", self.sim, SOLVER_FILE)
        elmer.load_solver("SaveLine", self.sim, SOLVER_FILE)

        # equations
        if self.heating_induction:
            equation_main = elmer.Equation(
                self.sim, "equation_main", [self.solver_statmag, self.solver_heat]
            )
        else:
            equation_main = elmer.Equation(
                self.sim, "main_equation", [self.solver_heat]
            )
        if self.phase_change:
            equation_main.solvers.append(solver_mesh)
        self._eqn_main = equation_main
        if self.phase_change:
            equation_phase_change = elmer.Equation(
                self.sim, "equation_phase_change", [self.solver_phase_change]
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

    @property
    def joule_heat(self):
        """Joule heat body force."""
        return self._joule_heat

    @property
    def distortion(self):
        """Boundary condition for distorted boundaries."""
        return [0, None]

    def add_crystal(self, shape, material="", force=None):
        """Add the crystal to the simulation.

        Args:
            shape (Shape): objectgmsh shape object.
            material (str, optional): material name. Defaults to "".
            force (elmer.BodyForce, optional): body force, e.g. joule
                heat. Defaults to None.

        Returns:
            Body: pyelmer Body object.
        """
        self._crystal = self.add_body(shape, material, force)
        return self._crystal

    def add_inductor(self, shape, material=""):
        """Add the inductor to the simulation (inductive heating).

        Args:
            shape (int): ID of the body
            material (str, optional): material name. Defaults to "".
        
        Returns:
            Body: pyelmer Body object.
        """
        self._current = elmer.BodyForce(self.sim, "Current Density")
        self._current.current_density = self.heating["current"] / shape.params.area
        self._inductor = self.add_body(shape, material, self._current)
        return self._inductor

    def add_resistance_heater(self, shape, material=""):
        """Add the resistance heater to the simulation. A constant
        heating power will be set in the volume.

        Args:
            shape (Shape): objectgmsh shape object.
            material (str, optional): material name. Defaults to "".

        Returns:
            Body: pyelmer Body object.
        """
        self._resistance_heating = elmer.BodyForce(self.sim, "resistance_heating")
        self._resistance_heating.heat_source = 1  # TODO set proper power_per_kilo?
        self._resistance_heating.integral_heat_source = self.heating["power"]
        if self.heat_control:
            self._resistance_heating.smart_heat_control = True
            if self.smart_heater["control-point"]:
                self._resistance_heating.smart_heater_control_point = [
                    self.smart_heater["x"],
                    self.smart_heater["y"],
                    self.smart_heater["z"],
                ]
                self._resistance_heating.smart_heater_T = self.smart_heater["T"]
        self._heater = self.add_body(shape, material, self._resistance_heating)
        return self._heater

    def add_body(self, shape, material="", force=None):
        """Add a body to the simulation.

        Args:
            shape (Shape): objectgmsh shape object.
            material (str, optional): material name. Defaults to "".
            force (BodyForce, optional): Body force, e.g., joule heat.
                Defaults to None.

        Returns:
            Body: pyelmer Body object.
        """
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
        """Add a boundary with Robin-BC for radiation and convective
        cooling.

        Args:
            shape (Shape): objectgmsh shape object.
            movement (list, optional): Movement of the boundary.
                Defaults to [0, 0].
            htc (float, optional): Heat transfer coefficient. Defaults
                to 0.
            T_ext (float, optional): External temperature for convective
                cooling model and radiation to ambient. Defaults to
                293.15.
            rad_s2s (bool, optional): use surface-to-surface radiation
                modeling. Defaults to True.

        Returns:
            Boundary: pyelmer Boundary object.
        """
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
        """Add a boundary with Dirichlet-BC for fixed temperature.

        Args:
            shape (Shape): objectgmsh shape object.
            T (float): boundary temperature.
            movement (list, optional): Movement of the boundary.
                Defaults to [0, 0].

        Returns:
            Boundary: pyelmer Boundary object.
        """
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.fixed_temperature = T
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def add_heatflux_boundary(self, shape, heatflux, movement=[0, 0]):
        """Add a boundary with Neumann-BC for fixed heat flux.

        Args:
            shape (Shape): objectgmsh shape object.
            heatflux (float): boundary heat flux.
            movement (list, optional): Movement of the boundary.
                Defaults to [0, 0].

        Returns:
            Boundary: pyelmer Boundary object.
        """
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        boundary.fixed_heatflux = heatflux
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def add_material(self, name, setup_file=""):
        """Add material to the simulation.

        Args:
            name (str): name of the material
            setup_file (str, optional): path to yml-file with material
                parameters. Defaults to "".

        Returns:
            Material: pyelmer Material object.
        """
        if name in self.sim.materials:
            return self.sim.materials[name]
        if name in self.materials_dict:
            print(f"using material {name} from self.materials_dict")
            material = elmer.Material(self.sim, name, self.materials_dict[name])
        else:
            if setup_file == "":
                setup_file = MATERIAL_FILE
            material = elmer.load_material(name, self.sim, setup_file)
        # remove properties not required for simulation
        if "Beta" in material.data:
            material.data.pop("Beta")
        if "Surface Tension" in material.data:
            material.data.pop("Surface Tension")
        return material

    def add_phase_interface(self, shape):
        """Add crystallization front to the simulation

        Args:
            shape (Shape): objectgmsh shape object.

        Returns:
            (Body, Boundary): pyelmer Body, Boundary object.
        """
        if not self.phase_change:
            raise ValueError("This Simulation does not include a phase change model.")
        if self._eqn_phase_change is None:
            raise ValueError("Equation is not initialized.")
        if self._crystal is None:
            raise ValueError("No crystal was added to the model.")
        # Body for phase change solver
        phase_if = elmer.Body(self.sim, "melt_crystal_if", [shape.ph_id])
        phase_if.material = self._crystal.material
        phase_if.equation = self._eqn_phase_change
        phase_if.initial_condition = elmer.InitialCondition(self.sim, "t0_phase_change")
        phase_if.initial_condition.data = {
            "Temperature": self._crystal.material.data["Melting Point"],
            "PhaseSurface": "Real 0.0",
        }
        # Boundary condition
        bc_phase_if = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        bc_phase_if.save_line = True
        bc_phase_if.normal_target_body = self.sim.bodies[self._crystal.name]
        if self.heat_control and not self.smart_heater["control-point"]:
            bc_phase_if.smart_heater = True
            bc_phase_if.smart_heater_T = self.smart_heater["T"]
        if self.phase_change:
            bc_phase_if.phase_change_steady = True
            bc_phase_if.phase_change_vel = self.v_pull / 6e4  # mm/min to m/s
            bc_phase_if.material = self._crystal.material
            bc_phase_if.phase_change_body = phase_if
        return phase_if, bc_phase_if

    def add_interface(self, shape, movement=[0, 0]):
        """Add a interface between to bodies to the simulation

        Args:
            shape (Shape): objectgmsh shape object.
            movement (list, optional): Movement of the interface.
                Defaults to [0, 0].

        Returns:
            Boundary: pyelmer Boundary object.
        """
        boundary = elmer.Boundary(self.sim, shape.name, [shape.ph_id])
        if self.mesh_update:
            boundary.mesh_update = movement
        return boundary

    def export(self):
        """Create simulation setup and write Elmer sif file."""
        if "global" in self.solver_update:
            self.sim.settings.update(self.solver_update["global"])
        if "all-solvers" in self.solver_update:
            self.solver_heat.data.update(self.solver_update["all-solvers"])
            if self.heating_induction:
                self.solver_statmag.data.update(self.solver_update["all-solvers"])
            if self.phase_change:
                self.solver_phase_change.data.update(self.solver_update["all-solvers"])
        if "solver-heat" in self.solver_update:
            self.solver_heat.data.update(self.solver_update["solver-heat"])
        if "solver-statmag" in self.solver_update:
            self.solver_statmag.data.update(self.solver_update["solver-statmag"])
        if "solver-phase-change" in self.solver_update:
            self.solver_phase_change.data.update(
                self.solver_update["solver-phase-change"]
            )

        self.sim.write_startinfo(self.sim_dir)
        self.sim.write_sif(self.sim_dir)
        self.sim.write_boundary_ids(self.sim_dir)
        with open(self.sim_dir + "/post_processing.yml", "w") as f:
            yaml.dump(self._heat_flux_dict, f)
        with open(self.sim_dir + "/probes.yml", "w") as f:
            yaml.dump(self.probes, f, sort_keys=False)

    def heat_flux_computation(self, body, boundary):
        """Define a body and boundary for evaluation of heat fluxes.

        Args:
            body (Body): pyelmer Body object
            boundary (Boundary): pyelmer Boundary object
        """
        # TODO that's a way too complicated!
        hfs = HeatfluxSurf(
            boundary.geo_ids[0],
            body.body_ids,
            body.material.data["Heat Conductivity"],
        )
        self._heat_flux_dict.update({f"{body.name}_{boundary.name}": asdict(hfs)})
