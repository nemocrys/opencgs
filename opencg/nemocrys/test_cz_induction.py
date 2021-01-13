import numpy as np
import os
import yaml

import pyelmer.elmer as elmer
from pyelmer.gmsh_objects import Model, Shape, MeshControlConstant, MeshControlLinear, MeshControlExponential

import opencg.sim
import opencg.geo.czochralski as cz


# TODO logging
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def geometry(config, dim=2):
    # load base configuration
    with open(THIS_DIR + '/test_cz_geo.yml') as f:
        base_config = yaml.safe_load(f)
    config.update(base_config)

    # initialize geometry model
    model = Model('cz-induction')

    # geometry
    crucible = cz.crucible(model, dim, **config['crucible'])
    melt = cz.melt(model, dim, crucible, **config['melt'])
    crystal = cz.crystal(model, dim, **config['crystal'], melt=melt)
    inductor = cz.inductor(model, dim, **config['inductor'])
    air = cz.air(model, dim, **config['air'])
    model.synchronize()

    # boundaries
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
    # simulation
    sim = opencg.sim.ElmerSimulation(config, phase_change=True, transient=False, heat_control=True)

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
    # TODO Heat Transfer Coefficients

    # export
    sim.export()
    # TODO export post-processing information

    return sim

if __name__ == "__main__":
    from pyelmer.execute import run_elmer_grid, run_elmer_solver
    from pyelmer.post import scan_logfile

    with open('./examples/config.yml') as f:
        config = yaml.safe_load(f)
    model = geometry(config)
    run_elmer_grid('./simdata/_test/', 'cz_induction.msh')
    elmer_setup(config, model)
    run_elmer_solver('./simdata/_test/')
    err, warn, stats = scan_logfile('./simdata/_test/')
    print(err, warn, stats)
