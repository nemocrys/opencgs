import numpy as np
import yaml

import pyelmer.elmer as elmer
from pyelmer.gmsh_objects import Model, Shape, MeshControlConstant, MeshControlLinear, MeshControlExponential
from pyelmer.execute import run_elmer_grid, run_elmer_solver
from pyelmer.post import scan_logfile

import cz_geometry
import opencg


# TODO logging

def geometry(config):
    # TODO Geometry in separate classes? -> Crucible, Inductor, Melt
    dim = config['settings']['dimension']

    # initialize gmsh
    model = Model('cz-induction')

    # geometry
    # TODO initial condition
    crucible = cz_geometry.crucible(model, dim, config['bodies']['crucible'])
    melt = cz_geometry.melt(model, dim, config['bodies']['melt'], crucible)
    crystal = cz_geometry.crystal(model, dim, config['bodies']['crystal'], melt)
    inductor = cz_geometry.inductor(model, dim, config['bodies']['inductor'])
    air = cz_geometry.air(model, dim, [0, -0.1, 0], [0.3, 0.5])
    model.synchronize()

    # boundaries
    # TODO Heat Transfer Coefficients
    if_crucible_melt = Shape(model, dim - 1, 'if_crucible_melt', crucible.get_interface(melt))
    if_melt_crystal = Shape(model, dim - 1,  'if_melt_crystal', melt.get_interface(crystal))
    if_crystal_air = Shape(model, dim - 1, 'if_crystal_air', crystal. get_interface(air))
    if_crucible_air = Shape(model, dim - 1, 'if_crucible_air', crucible.get_interface(air))
    if_melt_air = Shape(model, dim - 1, 'if_melt_air', melt.get_interface(air))
    if_inductor_air = Shape(model, dim - 1, 'if_inductor_air', inductor.get_interface(air))
    bnd_air_outside = Shape(model, dim - 1, 'bnd_air_outside',
                            [air.top_boundary, air.right_boundary, air.bottom_boundary])

    model.make_physical()

    # mesh
    # model.deactivate_characteristic_length()
    model.set_characteristic_length(0.1)
    MeshControlConstant(model, 0.01, shapes=[crucible, crystal])
    MeshControlLinear(model, if_crystal_air, 0.01, 0.1, shapes=[air])
    MeshControlExponential(model, if_melt_air, 0.001, shapes=[melt, air])
    model.generate_mesh()

    # visualize
    model.show()
    model.write_msh('./simdata/_test/cz_induction.msh')

    return model


def elmer_setup(config, model):
    # TODO simulation configuration -> keep yaml file?

    # simulation
    sim = opencg.ElmerSimulation(config, phase_change=True, transient=False, heat_control=True)

    # solvers
    sim.set_equations(heat=True, induction=True, probes=True)
    
    # materials
    mat_crucible = sim.add_material('graphite-CZ3R6300')
    mat_melt = sim.add_material('tin-liquid')
    mat_crystal = sim.add_material('tin-solid')
    mat_inductor = sim.add_material('copper-coil')
    mat_air = sim.add_material('air')

    # forces
    frc_current = sim.add_current(model['inductor'])
    frc_joule_heat = sim.joule_heat

    # bodies
    sim.add_crystal(model['crystal'], mat_crystal, frc_joule_heat)
    sim.add_body(model['crucible'], mat_crucible, frc_joule_heat)
    sim.add_body(model['melt'], mat_melt, frc_joule_heat)
    sim.add_body(model['inductor'], mat_inductor, frc_current)
    sim.add_body(model['air'], mat_air)

    # phase interface
    sim.add_phase_interface(model['if_melt_crystal'])

    # boundaries
    sim.add_radiation_boundary(model['if_crucible_air'])
    sim.add_radiation_boundary(model['if_crystal_air'])
    sim.add_radiation_boundary(model['if_melt_air'])
    sim.add_radiation_boundary(model['if_inductor_air'])
    sim.add_temperature_boundary(model['bnd_air_outside'], 273.15)

    # export
    sim.export()
    # TODO export post-processing information

    return sim

if __name__ == "__main__":
    with open('./base_parameters.yml') as f:
        config = yaml.safe_load(f)
    model = geometry(config)
    run_elmer_grid('./simdata/_test/', 'cz_induction.msh')
    elmer_setup(config, model)
    run_elmer_solver('./simdata/_test/')
    err, warn, stats = scan_logfile('./simdata/_test/')
    print(err, warn, stats)
