import yaml

from pyelmer.gmsh_objects import Shape
from pyelmer.gmsh_utils import *

from opencg import setup

# TODO stl import


def crucible(
    model,
    dim,
    h,
    r_in,
    r_out,
    t_bt,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible",
):
    crc = Shape(model, dim, name)

    crc.params.h = h
    crc.params.r_in = r_in
    crc.params.r_out = r_out
    crc.params.t_bt = t_bt
    crc.params.T_init = T_init
    crc.params.X0 = [0, -crc.params.t_bt]
    crc.params.material = material

    if char_l == 0:
        crc.mesh_size = min([r_out - r_in, t_bt]) / 5
    else:
        crc.mesh_size = char_l

    body = cylinder(0, crc.params.X0[1], 0, r_out, h, dim)
    hole = cylinder(0, 0, 0, r_in, h - t_bt, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()

    crc.geo_ids = [body]

    return crc


def melt(
    model,
    dim,
    crucible,
    h,
    char_l=0,
    T_init=273.15,
    material="",
    name="melt",
    crystal_radius=0,
    phase_if=None,
    rho=0,
    gamma=0,
    beta=0,
    g=9.81,
    res=100,
):
    melt = Shape(model, dim, name)
    melt.params.h = h
    melt.params.T_init = T_init
    melt.params.X0 = [0, 0]
    melt.params.material = material
    if char_l == 0:
        melt.mesh_size = melt.params.h / 10
    else:
        melt.mesh_size = char_l

    if crystal_radius == 0:  # no meniscus, no phase interface
        melt.params.h_meniscus = melt.params.h
        melt.geo_ids = [cylinder(0, 0, 0, crucible.params.r_in, h, dim)]
    else:  # with meniscus, following Landau87
        if rho == 0:  # read material data from material file
            with open(setup.MATERIAL_FILE) as f:
                data = yaml.safe_load(f)[material]
            rho = data["Density"]
            gamma = data["Surface Tension"]
            beta = data["Beta"] / 360 * 2 * np.pi  # theta in Landau87
        a = (2 * gamma / (rho * g)) ** 0.5  # capillary constant
        h = a * (1 - 1 * np.sin(beta)) ** 0.5
        z = np.linspace(0, h, res)[1:]
        x0 = (
            -a / 2 ** 0.5 * np.arccosh(2 ** 0.5 * a / h)
            + a * (2 - h ** 2 / a ** 2) ** 0.5
        )  # x(z) in Landau has wrong sign, multiplied by -1 here
        x = (
            a / 2 ** 0.5 * np.arccosh(2 ** 0.5 * a / z)
            - a * (2 - z ** 2 / a ** 2) ** 0.5
            + x0
        )
        # convert Landau coordinates into global coordinates
        meniscus_x = x + crystal_radius
        meniscus_y = z + h
        if meniscus_x.max() >= crucible.params.r_in:  # meniscus longer than melt: cut
            melt_surface_line = False
            for i in range(len(meniscus_x)):
                if meniscus_x[-(i + 1)] > crucible.params.r_in:
                    break
            meniscus_x = meniscus_x[-(i + 1) :]
            meniscus_y = meniscus_y[-(i + 1) :]
            meniscus_x[0] = crucible.params.r_in
        else:  # meniscus shorter than melt
            melt_surface_line = True
        meniscus_y += -meniscus_y.min() + melt.params.h
        melt.params.h_meniscus = meniscus_y.max()
        meniscus_points = [
            factory.addPoint(meniscus_x[i], meniscus_y[i], 0)
            for i in range(len(meniscus_x))
        ]
        if phase_if is not None:
            meniscus_points[-1] = phase_if.right_boundary
        melt_meniscus = factory.addSpline(meniscus_points)
        if phase_if is not None:
            top_left = phase_if.left_boundary
        else:
            top_left = factory.addPoint(0, meniscus_y.max(), 0)
        bottom_left = factory.addPoint(0, 0, 0)
        bottom_right = factory.addPoint(crucible.params.r_in, 0, 0)
        if melt_surface_line:
            top_right = factory.addPoint(crucible.params.r_in, melt.params.h, 0)
        else:
            top_right = meniscus_points[0]
        if phase_if is None:
            melt_crystal_if = factory.addLine(meniscus_points[-1], top_left)
        else:
            melt_crystal_if = phase_if.geo_id
        melt_sym_ax = factory.addLine(top_left, bottom_left)
        melt_crc_bt = factory.addLine(bottom_left, bottom_right)
        melt_crc_side = factory.addLine(bottom_right, top_right)
        if melt_surface_line:
            melt_surface = factory.addLine(top_right, meniscus_points[0])
            loop = factory.addCurveLoop(
                [
                    melt_crystal_if,
                    melt_sym_ax,
                    melt_crc_bt,
                    melt_crc_side,
                    melt_surface,
                    melt_meniscus,
                ]
            )
        else:
            loop = factory.addCurveLoop(
                [
                    melt_crystal_if,
                    melt_sym_ax,
                    melt_crc_bt,
                    melt_crc_side,
                    melt_meniscus,
                ]
            )
        body = factory.addSurfaceFilling(loop)
        if dim == 3:
            body = rotate(body)
        melt.geo_ids = [body]
    melt.set_interface(crucible)
    if phase_if is not None:
        model.remove_shape(phase_if)
    return melt


def crystal(
    model,
    dim,
    r,
    l,
    char_l=0,
    T_init=273.15,
    X0=[0, 0],
    material="",
    melt=None,
    phase_if=None,
    name="crystal",
):
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
        crys.params.X0 = X0
        crys.geo_ids = [cylinder(X0[0], X0[1], 0, r, l, dim)]
    else:  # in contact with melt
        crys.params.X0 = [0, melt.params.X0[1] + melt.params.h_meniscus]
        if phase_if is None:  # use cylinder
            crys.geo_ids = [cylinder(0, crys.params.X0[1], 0, r, l, dim)]
        else:  # draw manually
            top_left = factory.addPoint(0, crys.params.X0[1] + l, 0)
            top_right = factory.addPoint(r, crys.params.X0[1] + l, 0)
            left = factory.addLine(phase_if.left_boundary, top_left)
            top = factory.addLine(top_left, top_right)
            right = factory.addLine(top_right, phase_if.right_boundary)
            loop = factory.addCurveLoop([left, top, right, phase_if.geo_id])
            crys.geo_ids = [factory.addSurfaceFilling(loop)]
            model.remove_shape(phase_if)
        crys.set_interface(melt)
    return crys


def inductor(
    model,
    dim,
    d,
    d_in,
    X0,
    g=0,
    n=1,
    char_l=0,
    T_init=273.15,
    material="",
    name="inductor",
):
    # X0: center of bottom winding
    ind = Shape(model, dim, name)
    ind.params.d = d
    ind.params.d_in = d_in
    ind.params.g = g
    ind.params.n = n
    ind.params.X0 = X0
    ind.params.T_init = T_init
    ind.params.material = material
    ind.params.area = np.pi * (d ** 2 - d_in ** 2) / 4
    if char_l == 0:
        ind.mesh_size = d / 10
    else:
        ind.mesh_size = char_l

    x = X0[0]
    y = X0[1]
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


def crucible_support(
    model,
    dim,
    r_in,
    r_out,
    h,
    top_shape,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible_support",
):
    sup = Shape(model, dim, name)
    sup.params.r_in = r_in
    sup.params.r_out = r_out
    sup.params.h = h
    sup.params.X0 = [r_in, top_shape.params.X0[1] - h]
    sup.params.T_init = T_init
    sup.params.material = material
    if char_l == 0:
        sup.mesh_size = min([h, r_out - r_in]) / 5
    else:
        sup.mesh_size = char_l

    body = cylinder(0, sup.params.X0[1], 0, r_out, h, dim)
    if r_in != 0:
        hole = cylinder(0, sup.params.X0[1], 0, r_in, h, dim)
        factory.cut([(dim, body)], [(dim, hole)])
        factory.synchronize()
    sup.geo_ids = [body]
    sup.set_interface(top_shape)
    return sup


def crucible_adapter(
    model,
    dim,
    r_in_top,
    r_in_bt,
    r_out,
    h_top,
    h_bt,
    top_shape,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible_adapter",
):
    adp = Shape(model, dim, name)
    adp.params.r_in_top = r_in_top
    adp.params.r_in_bt = r_in_bt
    adp.params.r_out = r_out
    adp.params.h_top = h_top
    adp.params.h_bt = h_bt
    adp.params.X0 = [r_in_bt, top_shape.params.X0[1] - h_top]
    adp.params.T_init = T_init
    adp.params.material = material
    if char_l == 0:
        adp.mesh_size = min([r_out - r_in_top, r_out - r_in_bt, h_top + h_bt]) / 5
    else:
        adp.mesh_size = char_l

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


def seed(model, dim, crystal, r, l, char_l=0, T_init=273.15, material="", name="seed"):
    seed = Shape(model, dim, name)
    seed.params.l = l
    seed.params.r = r
    seed.params.X0 = [0, crystal.params.X0[1] + crystal.params.l]
    seed.params.T_init = T_init
    seed.params.material = material
    if char_l == 0:
        seed.mesh_size = r / 5
    else:
        seed.mesh_size = char_l

    seed.geo_ids = [cylinder(0, seed.params.X0[1], 0, r, l, dim)]
    seed.set_interface(crystal)

    return seed


def axis_top(
    model,
    dim,
    seed,
    r,
    l=0,
    vessel=None,
    char_l=0,
    T_init=273.15,
    material="",
    name="axis_top",
):
    ax = Shape(model, dim, name)
    ax.params.r = r
    ax.params.X0 = [0, seed.params.X0[1] + seed.params.l]
    ax.params.material = material
    if l == 0:
        if vessel is None:
            raise ValueError("If l=0 a vessel shape must be provided.")
        l = vessel.params.X0[1] + vessel.params.t + vessel.params.h_in - ax.params.X0[1]
    ax.params.l = l
    if char_l == 0:
        ax.mesh_size = r / 4
    else:
        ax.mesh_size = char_l

    ax.geo_ids = [cylinder(0, ax.params.X0[1], 0, r, l, dim)]
    ax.set_interface(seed)
    if vessel is not None:
        ax.set_interface(vessel)

    return ax


def vessel(
    model,
    dim,
    r_in,
    h_in,
    t,
    adjacent_shapes,
    char_l=0,
    T_init=273.15,
    material="",
    name="vessel",
):
    vsl = Shape(model, dim, name)
    vsl.params.r_in = r_in
    vsl.params.h_in = h_in
    vsl.params.t = t
    y0 = min([shape.params.X0[1] for shape in adjacent_shapes])
    vsl.params.X0 = [0, y0 - t]
    vsl.params.material = material
    if char_l == 0:
        vsl.mesh_size = t / 2
    else:
        vsl.mesh_size = char_l

    body = cylinder(0, y0 - t, 0, r_in + t, h_in + 2 * t, dim)
    hole = cylinder(0, y0, 0, r_in, h_in, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()
    vsl.geo_ids = [body]

    for shape in adjacent_shapes:
        vsl.set_interface(shape)

    return vsl


def filling(model, dim, vessel, char_l=0, T_init=273.15, material="", name="filling"):
    return surrounding(
        model,
        dim,
        [0, vessel.params.X0[1] + vessel.params.t],
        vessel.params.r_in,
        vessel.params.h_in,
        char_l,
        T_init,
        material,
        name,
    )


def surrounding(
    model, dim, X0, r, h, char_l=0, T_init=273.15, material="", name="surrounding"
):
    # get all other shapes first
    shapes = model.get_shapes(2)
    # create this shape afterwards
    sur = Shape(model, dim, name)
    sur.params.X0 = X0
    sur.params.r = r
    sur.params.h = h
    sur.params.T_init = T_init
    sur.params.material = material
    if char_l == 0:
        sur.mesh_size = min([r, h]) / 5
    else:
        sur.mesh_size = char_l

    body = cylinder(X0[0], X0[1], 0, r, h, dim)
    dim_tags = []
    for shape in shapes:
        dim_tags += shape.dimtags
    sur.geo_ids = cut([(2, body)], dim_tags, False)

    for shape in shapes:
        print(shape.name)
        sur.set_interface(shape)

    return sur


def resistance_heater():
    pass


def resistance_heating_insulation():
    pass
