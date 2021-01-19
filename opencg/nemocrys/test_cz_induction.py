import numpy as np
import os
import yaml

import pyelmer.elmer as elmer
from pyelmer.gmsh_objects import Model, Shape, MeshControlConstant, MeshControlLinear, MeshControlExponential

import opencg.sim
import opencg.geo.czochralski as cz


# TODO logging
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def geometry(config_update, dim=2):
    # load base configuration
    with open(THIS_DIR + '/test_cz_geo.yml') as f:
        config = yaml.safe_load(f)
    if config_update is not None:
        for body, update in config_update.items():
            config[body].update(update)

    # initialize geometry model
    model = Model('cz-induction')

    # geometry
    crucible = cz.crucible(model, dim, **config['crucible'])
    melt = cz.melt(model, dim, crucible, **config['melt'])
    crystal = cz.crystal(model, dim, **config['crystal'], melt=melt)
    inductor = cz.inductor(model, dim, **config['inductor'])
    air = cz.surrounding_box(model, dim, **config['air'])
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
    model.deactivate_characteristic_length()
    model.set_characteristic_length(0.1)
    for shape in [crucible, melt, crystal, inductor, air]:
        # print(shape.name, shape.mesh_size)
        MeshControlConstant(model, shape.mesh_size, [shape])
    MeshControlConstant(model, 0.01, shapes=[crucible, crystal])
    MeshControlLinear(model, if_crystal_air, 0.01, 0.1, shapes=[air])
    MeshControlExponential(model, if_melt_air, 0.001, shapes=[melt, air])
    model.generate_mesh()

    # visualize
    model.show()
    model.write_msh('./simdata/_test/cz_induction.msh')

    return model


def elmer_setup(config_update, model):
    with open(THIS_DIR + '/test_cz_sim.yml') as f:
        config = yaml.safe_load(f)
    if config_update is not None:
        for param, update in config_update.items():
            config[param].update(update)

    # simulation
    sim = opencg.sim.ElmerSimulationCz(**config['general'],
                                       probes=config['probes'],
                                       heating=config['heating_induction'],
                                       smart_heater=config['smart-heater'])
    # forces
    joule_heat = sim.joule_heat

    # bodies
    sim.add_crystal(model['crystal'], force=joule_heat)
    sim.add_inductor(model['inductor'])
    sim.add_body(model['crucible'], force=joule_heat)
    sim.add_body(model['melt'], force=joule_heat)
    sim.add_body(model['air'])

    # phase interface
    sim.add_phase_interface(model['if_melt_crystal'], model['crystal'])

    # boundaries
    sim.add_radiation_boundary(model['if_crucible_air'], htc=10, rad_s2s=False)
    sim.add_radiation_boundary(model['if_crystal_air'], rad_s2s=False)
    sim.add_radiation_boundary(model['if_melt_air'], rad_s2s=False)
    sim.add_radiation_boundary(model['if_inductor_air'], rad_s2s=False)
    sim.add_heatflux_boundary(model['bnd_air_outside'], 273.15)

    # export
    sim.export()
    # TODO export post-processing information

    return sim

if __name__ == "__main__":
    from pyelmer.execute import run_elmer_grid, run_elmer_solver
    from pyelmer.post import scan_logfile

    with open('./examples/config.yml') as f:
        config = yaml.safe_load(f)
    model = geometry(config['geometry'])
    run_elmer_grid('./simdata/_test/', 'cz_induction.msh')
    elmer_setup(config['simulation'], model)
