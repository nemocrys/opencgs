from pyelmer.gmsh_objects import Shape
from pyelmer.gmsh_utils import *


# TODO stl import

def crucible(model, dim, h, r_in, r_out, t_bt, char_l=0, T_init=273.15, material='',
             name='crucible'):
    crc = Shape(model, dim, name)

    crc.params.h = h
    crc.params.r_in = r_in
    crc.params.r_out = r_out
    crc.params.t_bt = t_bt
    crc.params.T_init = T_init
    crc.params.X0 = [0, -crc.params.t_bt]
    crc.params.material=material

    if char_l == 0:
        crc.mesh_size = r_out / 20
    else:
        crc.mesh_size = char_l

    body = cylinder(0, crc.params.X0[1], 0, r_out, h, dim)
    hole = cylinder(0, 0, 0, r_in, h -t_bt, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()

    crc.geo_ids = [body]

    return crc
    
def melt(model, dim, crucible, h, char_l=0, T_init=273.15, material='', crystal_radius=0,
         name='melt'):
    melt = Shape(model, dim, name)
    melt.params.h = h
    melt.params.T_init = T_init
    melt.params.X0 = [0, 0]
    melt.params.material= material
    if char_l == 0:
        melt.mesh_size = melt.params.h / 10
    else:
        melt.mesh_size = char_l

    if crystal_radius == 0:  # no meniscus
        melt.geo_ids = [cylinder(0, 0, 0, crucible.params.r_in, h, dim)]
        melt.set_interface(crucible)
    else:  # with meniscus
        # TODO with meniscus here!
        pass

    return melt

def crystal(model, dim, r, l, char_l=0, T_init=273.15, material='', melt=None, name='crystal'):
    crys = Shape(model, dim, name)
    crys.params.r = r
    crys.params.l = l
    crys.params.T_init = T_init
    crys.params.material = material
    if char_l == 0:
        crys.mesh_size = r / 10
    else:
        crys.mesh_size = char_l

    if melt is None:  # detached crystal
        pass
    else:  # in contact with melt
        crys.params.X0 = [0, melt.params.X0[1] + melt.params.h]
        crys.geo_ids = [cylinder(0, crys.params.X0[1], 0, r, l, dim)]
        crys.set_interface(melt)

    return crys

def inductor(model, dim, d, d_in, g, n, X0, char_l=0, T_init=273.15, material='', name='inductor'):
    ind = Shape(model, dim, name)
    ind.params.d = d
    ind.params.d_in = d_in
    ind.params.g = g
    ind.params.n = n
    ind.params.X0 = X0  # TODO
    ind.params.T_init = T_init
    ind.params.material = material
    ind.params.area = np.pi * (d**2 - d_in**2) / 4
    if char_l == 0:
        ind.mesh_size = d / 10
    else:
        ind.mesh_size = char_l

    x = X0[0] + d / 2
    y = X0[1] + d / 2
    for _ in range(n):
        circle_1d = factory.addCircle(x, y, 0, d / 2)
        circle = factory.addSurfaceFilling(factory.addCurveLoop([circle_1d]))
        hole_1d = factory.addCircle(x, y, 0, d_in / 2)
        hole = factory.addSurfaceFilling(factory.addCurveLoop([hole_1d]))
        factory.synchronize()
        factory.cut([(2, circle)], [(2, hole)])
        if dim == 3:
            circle = rotate(circle)
        ind.geo_ids.append(circle)
        y += g + d

    return ind

def crucible_support(model, dim, r_in, r_out, h, top_shape, char_l=0, T_init=273.15, material='',
                     name='crucible_support'):
    sup = Shape(model, dim, name)
    sup.params.r_in = r_in
    sup.params.r_out = r_out
    sup.params.h = h
    sup.params.X0 = [r_in, top_shape.params.X0[1] - h]
    sup.params.T_init = T_init
    sup.params.material = material

    body = cylinder(0, sup.params.X0[1], 0, r_out, h, dim)
    if r_in != 0:
        hole = cylinder(0, sup.params.X0[1], 0, r_in, h, dim)
        factory.cut([(dim, body)], [(dim, hole)])
        factory.synchronize()
    sup.geo_ids = [body]
    sup.set_interface(top_shape)
    return sup

def crucible_adapter(model, dim, r_in_top, r_in_bt, r_out, h_top, h_bt, top_shape, char_l=0,
                     T_init=273.15, material='', name='crucible_adapter'):
    adp = Shape(model, dim, name)
    adp.params.r_in_top = r_in_top
    adp.params.r_in_bt = r_in_bt
    adp.params.r_out = r_out
    adp.params.h_top = h_top
    adp.params.h_bt = h_bt
    adp.params.X0 = [r_in_bt, top_shape.params.X0[1] - h_top]
    adp.params.T_init = T_init
    adp.params.material = material

    body = cylinder(0, adp.params.X0[1] - h_bt, 0, r_out, h_top + h_bt, dim)
    holes = []
    if r_in_top != 0:
        hole1 = cylinder(0, adp.params.X0[1], 0, r_in_top, h_top, dim)
        holes.append((dim, hole1))
    if r_in_bt != 0:
        hole2 = cylinder(0, adp.params.X0[1] - h_bt, 0, r_in_bt, h_bt, dim)
        holes.append((dim, hole2))
    if holes != []:
        factory.cut([(dim, body)], holes)
        factory.synchronize()
    adp.geo_ids = [body]
    adp.set_interface(top_shape)
    return adp

def seed(model, dim, crystal, r, l, char_l=0, T_init=273.15, material='', name='seed'):
    seed = Shape(model, dim, name)
    seed.params.l = l
    seed.params.r = r
    seed.params.X0 = [0, crystal.params.X0[1] + crystal.params.l]
    seed.params.T_init = T_init
    seed.params.material = material

    seed.geo_ids = [cylinder(0, seed.params.X0[1], 0, r, l, dim)]
    seed.set_interface(crystal)

    return seed

def axis_top(model, dim, seed, r, l=0, vessel=None, char_l=0, T_init=273.15, material='',
             name='axis_top'):
    ax = Shape(model, dim, name)
    ax.params.r = r
    ax.params.X0 = [0, seed.params.X0[1] + seed.params.l]
    if l == 0:
        if vessel is None:
            raise ValueError('If l=0 a vessel shape must be provided.')
        l = vessel.params.X0[1] + vessel.params.t + vessel.params.h_in - ax.params.X0[1]
    ax.params.l = l

    ax.geo_ids = [cylinder(0, ax.params.X0[1], 0, r, l, dim)]
    ax.set_interface(seed)
    if vessel is not None:
        ax.set_interface(vessel)
    
    return ax

def vessel(model, dim, r_in, h_in, t, adjacent_shapes, char_l=0, T_init=273.15,
           material='', name='vessel'):
    vsl = Shape(model, dim, name)
    vsl.params.r_in = r_in
    vsl.params.h_in = h_in
    vsl.params.t = t
    y0 = min([shape.params.X0[1] for shape in adjacent_shapes])
    vsl.params.X0 = [0, y0 - t]

    body = cylinder(0, y0 - t, 0, r_in + t, h_in + 2 * t, dim)
    hole = cylinder(0, y0, 0, r_in, h_in, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()
    vsl.geo_ids = [body]

    for shape in adjacent_shapes:
        vsl.set_interface(shape)
    
    return vsl

def filling(model, dim, vessel, char_l=0.1, T_init=273.15, material='', name='filling'):
    return surrounding(model, dim, [0, vessel.params.X0[1] + vessel.params.t],
                       vessel.params.r_in, vessel.params.h_in, char_l, T_init,
                       material, name)

def surrounding(model, dim, X0, r, h, char_l=0.1, T_init=273.15, material='',
                    name='surrounding'):
    # get all other shapes first
    shapes = model.get_shapes(2)
    # create this shape afterwards
    sur = Shape(model, dim, name)
    sur.params.X0 = X0
    sur.params.r = r
    sur.params.h = h    
    sur.params.T_init = T_init
    sur.params.material = material
    sur.mesh_size=char_l

    body = cylinder(X0[0], X0[1], 0, r, h, dim)
    dim_tags = []
    for shape in shapes:
        dim_tags += shape.dimtags
    sur.geo_ids = cut([(2, body)], dim_tags, False)

    for shape in shapes:
        sur.set_interface(shape)

    return sur



def resistance_heater():
    pass

def resistance_heating_insulation():
    pass
