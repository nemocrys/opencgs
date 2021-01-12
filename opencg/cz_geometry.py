from pyelmer.gmsh_objects import Shape
from pyelmer.gmsh_utils import *

def crucible(model, dim, config):
    crc = Shape(model, dim, 'crucible')

    crc.params.h = config['h']
    crc.params.r_in = config['r-in']
    crc.params.r_out = config['r-out']
    crc.params.t_bt = config['t-bt']
    crc.params.T_init = config['T-init']
    crc.params.X0 = [0, -crc.params.t_bt]
    crc.mesh_size = crc.params.r_out / config['t-lc']

    body = cylinder(0, crc.params.X0[1], 0, crc.params.r_out, crc.params.h, dim)
    hole = cylinder(0, 0, 0, crc.params.r_in, crc.params.h -crc.params.t_bt, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()

    crc.geo_ids = [body]

    return crc
    
def melt(model, dim, config, crucible, crystal_radius=0):
    melt = Shape(model, dim, 'melt')
    melt.params.h = config['h']
    melt.params.T_init = config['T-init']
    melt.params.X0 = [0, 0]
    melt.mesh_size = melt.params.h / config['t-lc']

    if crystal_radius == 0:  # no meniscus
        melt.geo_ids = [cylinder(0, 0, 0, crucible.params.r_in, melt.params.h, dim)]
        melt.set_interface(crucible)
        melt.params.y_max = melt.params.h
    else:  # with meniscus
        # TODO with meniscus here!
        pass

    return melt

def crystal(model, dim, config, melt=None):
    # TODO implement as own child class of Shape
    crys = Shape(model, dim, 'crystal')
    crys.params.r = config['r']
    crys.params.l = config['l']
    crys.params.T_init = config['T-init']
    crys.mesh_size = crys.params.r / config['t-lc']

    if melt is None:  # detached crystal
        pass
    else:  # in contact with melt
        crys.params.X0 = [0, melt.params.y_max]
        crys.geo_ids = [cylinder(0, crys.params.X0[1], 0, crys.params.r, crys.params.l, dim)]
        crys.set_interface(melt)
    
    return crys

def inductor(model, dim, config):
    ind = Shape(model, dim, 'inductor')
    ind.params.d = config['d']
    ind.params.d_in = config['d-in']
    ind.params.g = config['g']
    ind.params.n = config['n']
    ind.params.X0 = [config['x'], config['h-over-floor'] - 0.075]  # TODO
    ind.params.T_init = config['T-init']
    ind.params.area = np.pi * (ind.params.d**2 - ind.params.d_in**2) / 4
    ind.mesh_size = ind.params.d / config['t-lc']

    x = ind.params.X0[0] + ind.params.d / 2
    y = ind.params.X0[1] + ind.params.d / 2
    for i in range(ind.params.n):
        circle_1d = factory.addCircle(x, y, 0, ind.params.d / 2)
        circle = factory.addSurfaceFilling(factory.addCurveLoop([circle_1d]))
        hole_1d = factory.addCircle(x, y, 0, ind.params.d_in / 2)
        hole = factory.addSurfaceFilling(factory.addCurveLoop([hole_1d]))
        factory.synchronize()
        factory.cut([(2, circle)], [(2, hole)])
        if dim == 3:
            circle = rotate(circle)
        ind.geo_ids.append(circle)
        y += (ind.params.g + ind.params.d)

    return ind

def air(model, dim, X_min, X_max):
    shapes = model.get_shapes(2)
    dim_tags = []
    for shape in shapes:
        dim_tags += shape.dimtags

    air = Shape(model, dim, 'air')
    air.params.X_min = X_min
    air.params.X_max = X_max

    tag = factory.addRectangle(X_min[0], X_min[1], X_min[2], X_max[0], X_max[1])
    air.geo_ids = cut([(2, tag)], dim_tags, False)

    return air
    
