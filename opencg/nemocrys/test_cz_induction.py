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
    melt = cz.melt(model, dim, crucible, **config['melt'], crystal_radius=config['crystal']['r'])
    crystal = cz.crystal(model, dim, **config['crystal'], melt=melt)
    inductor = cz.inductor(model, dim, **config['inductor'])
    seed = cz.seed(model, dim, **config['seed'], crystal=crystal)
    ins = cz.crucible_support(model, dim,**config['insulation'], top_shape=crucible, name='insulation')
    adp = cz.crucible_adapter(model, dim, **config['crucible_adapter'], top_shape=ins)
    ax_bt = cz.crucible_support(model, dim, **config['axis_bt'], top_shape=adp, name='axis_bt')
    vessel = cz.vessel(model, dim, **config['vessel'], adjacent_shapes=[ax_bt])
    ax_top = cz.axis_top(model, dim, **config['axis_top'], seed=seed, vessel=vessel)
    filling = cz.filling(model, dim, **config['filling'], vessel=vessel)
    filling_in_inductor = Shape(model, dim, 'filling_in_inductor', filling.get_part_in_box(
        [inductor.params.X0[0] - inductor.params.d_in / 2, inductor.params.X0[0] + inductor.params.d_in / 2],
        [inductor.params.X0[1] - inductor.params.d_in, 1e6]
    ))
    filling -= filling_in_inductor
    model.synchronize()

    # boundaries
    bnd_melt = Shape(model, dim - 1, 'bnd_melt', melt.get_interface(filling))
    bnd_crystal = Shape(model, dim - 1, 'bnd_crystal', crystal.get_interface(filling))
    bnd_seed = Shape(model, dim - 1, 'bnd_seed', seed.get_interface(filling))
    bnd_axtop = Shape(model, dim -1, 'bnd_axtop', ax_top.get_interface(filling))
    
    bnd_crucible_bt = Shape(model, dim - 1, 'bnd_crucible_bt', crucible.get_boundaries_in_box(
        [0, ins.params.r_in], [crucible.params.X0[1], crucible.params.X0[1]]
    ))
    bnd_crucible_outside = Shape(model, dim - 1, 'bnd_crucible_outside', crucible.get_interface(filling))
    bnd_crucible_outside -= bnd_crucible_bt

    bnd_ins = Shape(model, dim -1, 'bnd_ins', ins.get_interface(filling))
    bnd_adp = Shape(model, dim -1, 'bnd_adp', adp.get_interface(filling))
    bnd_axbt = Shape(model, dim -1, 'bnd_axbt', ax_bt.get_interface(filling))
    bnd_vessel_inside = Shape(model, dim - 1, 'bnd_vessel_inside', [
        vessel.get_boundaries_in_box([ax_bt.params.r_out, vessel.params.r_in], [ax_bt.params.X0[1], ax_bt.params.X0[1]], one_only=True),  # bottom
        vessel.get_boundaries_in_box([ax_top.params.r, vessel.params.r_in], [ax_bt.params.X0[1] + vessel.params.h_in, ax_bt.params.X0[1] + vessel.params.h_in], one_only=True),  # top
        vessel.get_boundaries_in_box([vessel.params.r_in, vessel.params.r_in], [ax_bt.params.X0[1], ax_bt.params.X0[1] + vessel.params.h_in], one_only=True)  # wall
    ])
    bnd_vessel_outside = Shape(model, dim - 1, 'bnd_vessel_outside', [vessel.bottom_boundary, vessel.top_boundary, vessel.right_boundary])
    bnd_inductor_outside = Shape(model, dim - 1, 'bnd_inductor_outside', inductor.get_interface(filling))
    bnd_inductor_inside = Shape(model, dim - 1, 'bnd_inductor_inside', inductor.get_interface(filling_in_inductor))
    model.remove_shape(filling_in_inductor)

    # interfaces
    if_crucible_melt = Shape(model, dim - 1, 'if_crucible_melt', crucible.get_interface(melt))
    if_melt_crystal = Shape(model, dim - 1,  'if_melt_crystal', melt.get_interface(crystal))
    if_crystal_seed = Shape(model, dim -1, 'if_crystal_seed', crystal.get_interface(seed))
    if_seed_axtop = Shape(model, dim - 1, 'if_seed_axtop', seed.get_interface(ax_top))
    if_axtop_vessel = Shape(model, dim - 1, 'if_axtop_vessel', ax_top.get_interface(vessel))
    if_crucible_ins = Shape(model, dim - 1, 'if_crucible_ins', crucible.get_interface(ins))
    if_ins_adp = Shape(model, dim - 1, 'if_ins_adp', ins.get_interface(adp))
    if_adp_axbt = Shape(model, dim - 1, 'if_adp_axbt', adp.get_interface(ax_bt))
    if_axbt_vessel = Shape(model, dim - 1, 'if_axbt_vessel', ax_bt.get_interface(vessel))

    model.make_physical()

    # mesh
    model.deactivate_characteristic_length()
    model.set_const_mesh_sizes()
    for shape in [melt, crystal, seed, ax_top, crucible, ins, adp, ax_bt, vessel]:
        MeshControlLinear(model, shape, shape.mesh_size, filling.mesh_size)
    MeshControlExponential(model, if_melt_crystal, crystal.params.r / 30, exp=1.6, fact=3)
    MeshControlExponential(model, bnd_melt, melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(model, if_crucible_melt, melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(model, inductor, inductor.mesh_size)
    MeshControlExponential(model, bnd_crucible_outside, crucible.mesh_size / 3, exp=1.6, fact=3)
    model.generate_mesh()

    # visualize
    # model.show()
    model.write_msh('./simdata/_test/cz_induction.msh')
    print(model)
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
    sim.add_inductor(model['inductor'])
    sim.add_crystal(model['crystal'], force=joule_heat)
    sim.add_body(model['melt'], force=joule_heat)
    sim.add_body(model['crucible'], force=joule_heat)
    sim.add_body(model['insulation'], force=joule_heat)
    sim.add_body(model['crucible_adapter'], force=joule_heat)
    sim.add_body(model['axis_bt'], force=joule_heat)
    sim.add_body(model['vessel'], force=joule_heat)
    sim.add_body(model['seed'], force=joule_heat)
    sim.add_body(model['axis_top'], force=joule_heat)
    sim.add_body(model['filling'], force=joule_heat)

    # phase interface
    sim.add_phase_interface(model['if_melt_crystal'], model['crystal'])

    # boundaries
    sim.add_radiation_boundary(model['bnd_crystal'], **config['boundaries']['crystal'])
    sim.add_radiation_boundary(model['bnd_melt'], **config['boundaries']['melt'])
    sim.add_radiation_boundary(model['bnd_crucible_outside'], **config['boundaries']['crucible_outside'])
    for bnd in ['bnd_crucible_bt', 'bnd_ins', 'bnd_adp', 'bnd_axbt', 'bnd_vessel_inside',
                'bnd_inductor_outside', 'bnd_seed', 'bnd_axtop']:
        sim.add_radiation_boundary(model[bnd])
    for bnd in ['if_crucible_melt', 'if_axtop_vessel', 'if_crucible_ins', 'if_ins_adp',
                'if_adp_axbt', 'if_axbt_vessel']:
        pass  # TODO add_interface()
    for bnd in ['if_crystal_seed', 'if_seed_axtop']:
        pass
    sim.add_temperature_boundary(model['bnd_inductor_inside'], **config['boundaries']['inductor_inside'])
    sim.add_temperature_boundary(model['bnd_vessel_outside'], **config['boundaries']['vessel_outside'])

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
