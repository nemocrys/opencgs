from pyelmer.gmsh_objects import Shape
from pyelmer.gmsh_utils import *

def crucible(model, dim, h, r_in, r_out, t_bt, char_l=0, T_init=273.15):
    crc = Shape(model, dim, 'crucible')

    crc.params.h = h
    crc.params.r_in = r_in
    crc.params.r_out = r_out
    crc.params.t_bt = t_bt
    crc.params.T_init = T_init
    crc.params.X0 = [0, -crc.params.t_bt]
    if char_l == 0:
        crc.mesh_size = r_out / 20
    else:
        crc.mesh_size = char_l

    body = cylinder(0, crc.params.X0[1], 0, crc.params.r_out, crc.params.h, dim)
    hole = cylinder(0, 0, 0, crc.params.r_in, crc.params.h -crc.params.t_bt, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()

    crc.geo_ids = [body]

    return crc
    
def melt(model, dim, crucible, h, char_l=0, T_init=273.15, crystal_radius=0):
    melt = Shape(model, dim, 'melt')
    melt.params.h = h
    melt.params.T_init = T_init
    melt.params.X0 = [0, 0]
    if char_l == 0:
        melt.mesh_size = melt.params.h / 10
    else:
        melt.mesh_size = char_l

    if crystal_radius == 0:  # no meniscus
        melt.geo_ids = [cylinder(0, 0, 0, crucible.params.r_in, melt.params.h, dim)]
        melt.set_interface(crucible)
        melt.params.y_max = melt.params.h
    else:  # with meniscus
        # TODO with meniscus here!
        pass

    return melt

def crystal(model, dim, r, l, char_l=0, T_init=273.15, melt=None):
    crys = Shape(model, dim, 'crystal')
    crys.params.r = r
    crys.params.l = l
    crys.params.T_init = T_init
    if char_l == 0:
        crys.mesh_size = crys.params.r / 10
    else:
        crys.mesh_size = char_l

    if melt is None:  # detached crystal
        pass
    else:  # in contact with melt
        crys.params.X0 = [0, melt.params.y_max]
        crys.geo_ids = [cylinder(0, crys.params.X0[1], 0, crys.params.r, crys.params.l, dim)]
        crys.set_interface(melt)
    
    return crys

def inductor(model, dim, d, d_in, g, n, X0, char_l=0, T_init=273.15):
    ind = Shape(model, dim, 'inductor')
    ind.params.d = d
    ind.params.d_in = d_in
    ind.params.g = g
    ind.params.n = n
    ind.params.X0 = X0  # TODO
    ind.params.T_init = T_init
    ind.params.area = np.pi * (d**2 - d_in**2) / 4
    if char_l == 0:
        ind.mesh_size = ind.params.d / 10
    else:
        ind.mesh_size = char_l

    x = ind.params.X0[0] + d / 2
    y = ind.params.X0[1] + d / 2
    for _ in range(ind.params.n):
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

def air(model, dim, X_min, X_max, T_init=273.15):
    shapes = model.get_shapes(2)
    dim_tags = []
    for shape in shapes:
        dim_tags += shape.dimtags

    air = Shape(model, dim, 'air')
    air.params.X_min = X_min
    air.params.X_max = X_max
    air.params.T_init = T_init

    tag = factory.addRectangle(X_min[0], X_min[1], X_min[2], X_max[0], X_max[1])
    air.geo_ids = cut([(2, tag)], dim_tags, False)

    return air
    
