from objectgmsh import *
import warnings

occ = gmsh.model.occ  # equivalent to factory


def line_from_points(model, points, name="line"):
    """Create a 1D shape from a list of points.

    Args:
        model (Model): objectgmsh model
        points (list): list of points [x1, x2, ...],
            xi = [x, y, (z optional)]
        name (str): name of the shape

    Returns:
        Shape: objectgmsh Shape object
    """
    line = Shape(model, 1, name)
    line.params.points = points
    if len(points[0]) == 2:
        pts = [factory.addPoint(x[0], x[1], 0) for x in points]
    else:
        pts = [factory.addPoint(x[0], x[1], x[2]) for x in points]
    line.geo_ids = [factory.addSpline(pts)]
    factory.synchronize()
    return line

def shape_from_points(model, points, name, material="", mesh_size=0):
    """Create a 2D shape from a list of points.

    Args:
        model (Model): objectgmsh model
        points (list): list of points [x1, x2, ...],
            xi = [x, y, (z optional)]
        name (str): name of the shape
        material (str, optional): material name. Defaults to "".
        mesh_size (float, optional): mesh size characteristic length.
            Defaults to 0.

    Returns:
        Shape: objectgmsh Shape object
    """
    shape = Shape(model, 2, name)
    shape.params.material = material
    shape.mesh_size = mesh_size
    if len(points[0]) == 2:
        pts = [factory.addPoint(x[0], x[1], 0) for x in points]
    else:
        pts = [factory.addPoint(x[0], x[1], x[2]) for x in points]
    pts.append(pts[0])  # for closed loop
    lines = []
    for i in range(len(pts) - 1):
        lines.append(factory.add_line(pts[i], pts[i+1]))
    shape.geo_ids = [factory.addSurfaceFilling(factory.addCurveLoop(lines))]
    factory.synchronize()
    return shape

def shape_with_hole(model, points_outside, points_inside, name, material="", mesh_size=0):
    """Create a 2D shape with a hole from a list of points.

    Args:
        model (Model): objectgmsh model
        points_outside (list): list of points [x1, x2, ...],
            xi = [x, y, (z optional)]
        points_inside (list): list of points [x1, x2, ...],
            xi = [x, y, (z optional)]
        name (str): name of the shape
        material (str, optional): material name. Defaults to "".
        mesh_size (float, optional): mesh size characteristic length.
            Defaults to 0.

    Returns:
        Shape: objectgmsh Shape object
    """
    shape = shape_from_points(model, points_outside, name, material, mesh_size)
    hole = shape_from_points(model, points_inside, "temp_hole")
    factory.cut(shape.dimtags, hole.dimtags, removeTool=False)
    factory.synchronize()
    model.remove_shape(hole)
    return shape

def compute_current_crystal_surface(
    surface_points,
    current_length,
    seed_length=0,
    dx_min = 1e-4,
    **_,
):
    """Compute the current surface shape of the crystal in a
    quasi-transient simulation.

    Args:
        surface_points (list): List of points defining the crystal shape
        current_length (list): Current length of the crystal
        seed_length (float, optional): Length of the seed (if included
          in points defined above)
        dx_min (float, optional): Minimum distance to last point
          (used to avoid meshing errors). Defaults to 1e-4.

    Returns:
        tuple: list with surface points, volume, crystal surface angle
    """
    # sort out all points that exceed crystal length (just keep the last for a moment)
    surface_points = np.array(surface_points)
    mask = surface_points[:, 1] < round(current_length + seed_length, 10)
    i = np.argmin(mask) + 1  # include one mor point
    surface_points = surface_points[:i, :]

    # adjust coordinates of last point with lin. interpolation to get correct crystal length
    factor = (current_length + seed_length - surface_points[-2, 1]) / (surface_points[-1, 1] - surface_points[-2, 1])
    r = surface_points[-2, 0] + factor * (surface_points[-1, 0] - surface_points[-2, 0])
    z = surface_points[-2, 1] + factor * (surface_points[-1, 1] - surface_points[-2, 1])
    surface_points[-1, 0] = r
    surface_points[-1, 1] = z

    # a too small distance between last point and second last point leads to a bad mesh
    x1 = surface_points[-1, :]
    x2 = surface_points[-2, :]
    dx = ( (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 )**0.5
    if dx < dx_min:  # remove 2nd last point
        surface_points[-2, :] = x1
        surface_points = surface_points[:-1, :]

    # compute volume
    volume = 0
    for i in range(len(surface_points) - 1):
        r1 = surface_points[i, 0]
        r2 = surface_points[i + 1, 0]
        h = surface_points[i + 1, 1] - surface_points[i, 1]
        volume +=  np.pi *  h / 3 * (r1**2 + r1*r2 + r2**2)  # truncated cone

    dx = surface_points[-1, 0] - surface_points[-2, 0]
    dy = surface_points[-1, 1] - surface_points[-2, 1]
    alpha =  np.rad2deg(np.arctan(dx / dy))

    return surface_points, volume, alpha

def compute_meniscus_volume(meniscus_x, meniscus_y ):
    """Compute the meniscus volume for a quasi-transient simulation.

    Args:
        meniscus_x (list): x-coordinates
        meniscus_y (list): y-coordinates

    Returns:
        float: meniscus volume
    """
    volume = 0
    for i in range(len(meniscus_x) - 1):
        r1 = meniscus_x[i]
        r2 = meniscus_x[i + 1]
        h = meniscus_y[i + 1] - meniscus_y[i]
        volume +=  np.pi *  h / 3 * (r1**2 + r1*r2 + r2**2)  # truncated cone
    return volume

def menicus_hurle(
        r_crys,
        alpha_crys,
        beta,
        gamma,
        rho,
        g=9.81,
        alpha_start=.1,
        res=100,
):
    """Compute meniscus shape according to Hurle 1983 

    Args:
        r_crys (float): crystal radius
        alpha_crys (float): crystal angle in deg
        beta (float): wetting angle in deg
        gamma (float): surface tension
        rho (float): density
        g (float, optional): gravity. Defaults to 9.81.
        alpha_start (float, optional): starting angle for evaluation.
            Defaults to 1.
        res (int, optional): resolution / number of points.
            Defaults to 100.

    Returns:
        tuple (np.array, np.array, np.array, float, float):
        (x, y, r_crit, h_meniscus, alpha)
    """
    alpha_crys = np.deg2rad(alpha_crys)
    beta = np.deg2rad(beta)

    a_l = (2 * gamma / (rho * g)) ** 0.5  # Laplace constant
    alpha_gr = np.pi / 2 - (beta + alpha_crys)
    h_meniscus = a_l * ((1 - np.cos(alpha_gr)) / (1 + a_l / (2**0.5 * r_crys)) ** 0.5)
    r_crit = 1.5 * (2 * gamma / (rho * g))**0.5
    if r_crys < r_crit:
        warnings.warn(f"Crystal radius {r_crys*1e3:.2f} mm lower than critical radius {r_crit*1e3:.2f} mm. Meniscus shape approximation may be inaccurate.")
    
    RR = r_crys * 2**0.5/ a_l

    alpha = np.linspace(np.deg2rad(alpha_start), alpha_gr, res)
    x = a_l / 2**0.5 * (
    RR + 1/( 1 + 1 / RR )**0.5 * (
            np.log(np.tan(alpha_gr/4) / np.tan(alpha/4))
            + 2*(np.cos(alpha_gr/2) - np.cos(alpha/2))
        )
    )
    y = a_l / 2**0.5 * ( 2 * (1 - np.cos(alpha)) / (1 + 1/RR) )**0.5
    return (x, y, r_crit, h_meniscus, alpha)

def meniscus_landau(
        r_crys,
        beta,
        gamma,
        rho,
        g=9.81,
        res=100,
):
    # Landau 87, analytical solution of Young-Laplace equation in 2D, valid for infinite crystal radius
    beta = np.deg2rad(beta)  # theta in Landau 87
    a = (2 * gamma / (rho * g)) ** 0.5  # capillary constant
    h = a * (1 - 1 * np.sin(beta)) ** 0.5
    y = np.linspace(0, h, res)[1:]
    x0 = (
        -a / 2**0.5 * np.arccosh(2**0.5 * a / h)
        + a * (2 - h**2 / a**2) ** 0.5
    )  # x(y) in Landau has wrong sign, multiplied by -1 here
    x = (
        a / 2**0.5 * np.arccosh(2**0.5 * a / y)
        - a * (2 - y**2 / a**2) ** 0.5
        + x0
    )
    return x + r_crys, y, h

def cylinder_shape(
    model,
    dim,
    r_in,
    r_out,
    h,
    overlap=0,
    top_shape=None,
    bot_shape=None,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible_support",
):
    cyl = Shape(model, dim, name)
    cyl.params.r_in = r_in
    cyl.params.r_out = r_out
    cyl.params.h = h
    if bot_shape is None:
        cyl.params.X0 = [r_in, top_shape.params.X0[1] - h + overlap]
    else:
        try:
            cyl.params.X0 = [
                r_in,
                bot_shape.params.X0[1] + bot_shape.params.h - overlap,
            ]
        except AttributeError:
            cyl.params.X0 = [
                r_in,
                bot_shape.params.X0[1] + bot_shape.params.l - overlap,
            ]
    cyl.params.T_init = T_init
    cyl.params.material = material
    if char_l == 0:
        cyl.mesh_size = min([h, r_out - r_in]) / 5
    else:
        cyl.mesh_size = char_l

    body = cylinder(0, cyl.params.X0[1], 0, r_out, h, dim)
    if r_in != 0:
        hole = cylinder(0, cyl.params.X0[1], 0, r_in, h, dim)
        factory.cut([(dim, body)], [(dim, hole)])
        factory.synchronize()
    cyl.geo_ids = [body]
    if bot_shape is None:
        cyl.set_interface(top_shape)
    else:
        cyl.set_interface(bot_shape)
    return cyl

def stacked_cylinder_shape(
    model,
    dim,
    cylinders,
    overlap=0,
    top_shape=None,
    bot_shape=None,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible_support",
):  
    h_tot = 0
    for cyl in cylinders:
        h_tot += cyl["h"]
    shape = Shape(model, dim, name)
    shape.params.h = h_tot
    if bot_shape is None:
        shape.params.X0 = [0, top_shape.params.X0[1] - h_tot + overlap]
    else:
        try:
            shape.params.X0 = [
                0,
                bot_shape.params.X0[1] + bot_shape.params.h - overlap,
            ]
        except AttributeError:
            shape.params.X0 = [
                0,
                bot_shape.params.X0[1] + bot_shape.params.l - overlap,
            ]
    shape.params.T_init = T_init
    shape.params.material = material
    if char_l == 0:
        shape.mesh_size = h_tot / 5 / len(cylinders)
    else:
        shape.mesh_size = char_l

    y0 = shape.params.X0[1]
    for cyl in cylinders:
        body = cylinder(0, y0, 0, cyl["r_out"], cyl["h"], dim)
        if cyl["r_in"] != 0:
            hole = cylinder(0, y0, 0, cyl["r_in"], cyl["h"], dim)
            factory.cut([(dim, body)], [(dim, hole)])
        factory.synchronize()
        factory.fragment(shape.dimtags, [(dim, body)])
        shape.geo_ids.append(body)
        y0 += cyl["h"]
    if bot_shape is None:
        shape.set_interface(top_shape)
    else:
        shape.set_interface(bot_shape)
    return shape

def crucible(
    model,
    dim,
    h,
    r_in,
    r_out,
    t_bt,
    y0=0.0,
    char_l=0,
    T_init=273.15,
    material="",
    name="crucible",
):
    """Cylindrical crucible

    Args:
        model (Model): objectgmsh model
        dim (int): dimension
        h (float): height
        r_in (float): inner radius
        r_out (float): outer radius
        t_bt (float): bottom thickness
        char_l (float, optional): mesh size characteristic length.
            Defaults to 0.
        T_init (float, optional): initial temperature. Defaults to
            273.15.
        material (str, optional): material name. Defaults to "".
        name (str, optional): name of the shape. Defaults to "crucible".

    Returns:
        Shape: objectgmsh shape
    """
    crc = Shape(model, dim, name)

    crc.params.h = h
    crc.params.r_in = r_in
    crc.params.r_out = r_out
    crc.params.t_bt = t_bt
    crc.params.T_init = T_init
    crc.params.X0 = [0, -crc.params.t_bt + y0]
    crc.params.material = material

    if char_l == 0:
        crc.mesh_size = min([r_out - r_in, t_bt]) / 5
    else:
        crc.mesh_size = char_l

    body = cylinder(0, crc.params.X0[1], 0, r_out, h, dim)
    hole = cylinder(0, y0, 0, r_in, h - t_bt, dim)
    factory.cut([(dim, body)], [(dim, hole)])
    factory.synchronize()

    crc.geo_ids = [body]

    return crc

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
    alpha_crys=0,
    phase_if=None,
    rho=0,
    gamma=0,
    beta=0,
    g=9.81,
    res=100,
):
    """Melt in cylindrical crucible, with meniscus

    Args:
        model (Model): objectgmsh model
        dim (int): dimension
        crucible (Shape): objectgmsh shape object
        h (float): melt height in crucible
        char_l (float, optional): mesh size characteristic length.
            Defaults to 0.
        T_init (float, optional): initial temperature. Defaults to
            273.15.
        material (str, optional): melt material name. Defaults to "".
        name (str, optional): name of the shape. Defaults to "melt".
        crystal_radius (float, optional): radius of crystal. Defaults to
            0.
        phase_if (Shape, optional): phase boundary. Defaults to None.
        rho (float, optional): density of the melt (for meniscus
            computation). Defaults to 0.
        gamma (float, optional): surface tension of the melt (for
            meniscus computation). Defaults to 0.
        beta (float, optional): contact angle at crystal (for meniscus
            computation). Defaults to 0.
        g (float, optional): gravitational acceleration (for meniscus
            computation). Defaults to 9.81.
        res (int, optional): number of points in meniscus computation.
            Defaults to 100.

    Returns:
        Shape: objectgmsh shape
    """
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
    else:  # with meniscus
        meniscus_x, meniscus_y, r_crit, _, _ = menicus_hurle(
            crystal_radius,
            alpha_crys,
            beta,
            gamma,
            rho,
            g,
            res=res,
        )
        meniscus_y += h
        #####

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
    r_top=-1,
    char_l=0,
    T_init=273.15,
    X0=[0, 0],
    material="",
    melt=None,
    phase_if=None,
    name="crystal",
):
    """Cylindrical / conical crystal.

    Args:
        model (Model): objectgmsh model
        dim (int): dimension
        r (float): crystal radius
        l (float): crystal length
        r_top (float, optional): top radius of conical crystal. Defaults
            to -1 -> cylindrical crystal shape
        char_l (float, optional): mesh size characteristic length.
            Defaults to 0.
        T_init (float, optional): initial temperature. Defaults to
            273.15.
        X0 (list, optional): origin of crystal (bottom of symmetry
            axis), if not in contact with melt. Defaults to [0, 0].
        material (str, optional): material name. Defaults to "".
        melt (Shape, optional): melt object, if the crystal is in
            contact with melt. Defaults to None.
        phase_if (shape, optional): phase boundary shape. Defaults to
            None.
        name (str, optional): name of the shape. Defaults to "crystal".

    Returns:
        Shape: objectgmsh shape
    """
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
        if r_top != -1:
            raise NotImplementedError("Non-cylindrical shapes not supported if crystal is not in contact with melt.")
        crys.geo_ids = [cylinder(X0[0], X0[1], 0, r, l, dim)]
    else:  # in contact with melt
        crys.params.X0 = [0, melt.params.X0[1] + melt.params.h_meniscus]
        if phase_if is None:
            bottom_left = factory.addPoint(crys.params.X0[0], crys.params.X0[1], 0)
            bottom_right = factory.addPoint(crys.params.X0[0] + r, crys.params.X0[1], 0)
        else:
            bottom_left = phase_if.left_boundary
            bottom_right = phase_if.right_boundary
        top_left = factory.addPoint(0, crys.params.X0[1] + l, 0)
        if r_top == -1:  # cylindrical shape
            top_right = factory.addPoint(r, crys.params.X0[1] + l, 0)
        else:
            top_right = factory.addPoint(r_top, crys.params.X0[1] + l, 0)
        left = factory.addLine(bottom_left, top_left)
        top = factory.addLine(top_left, top_right)
        right = factory.addLine(top_right, bottom_right)
        if phase_if is None:
            bottom = factory.addLine(bottom_right, bottom_left)
            loop = factory.addCurveLoop([left, top, right, bottom])
        else:
            loop = factory.addCurveLoop([left, top, right, phase_if.geo_id])
            model.remove_shape(phase_if)
        crys.geo_ids = [factory.addSurfaceFilling(loop)]
        crys.set_interface(melt)
    return crys

def crystal_from_points(model,
    dim,
    surface_points,
    current_length=0,
    seed_length=0,
    char_l=0,
    T_init=273.15,
    X0=[0, 0],
    material="",
    melt=None,
    phase_if=None,
    name="crystal",  
):  
    surface_points, _, alpha = compute_current_crystal_surface(
        surface_points,
        current_length,
        seed_length
    )

    crys = Shape(model, dim, name)
    crys.params.r = surface_points[-1, 0]
    crys.params.l = current_length + seed_length
    crys.params.T_init = T_init
    crys.params.material = material

    if char_l == 0:
        crys.mesh_size = crys.params.r / 10
    else:
        crys.mesh_size = char_l

    if melt is None:  # detached crystal
        crys.params.X0 = X0
    else:
        crys.params.X0 = [0, melt.params.X0[1] + melt.params.h_meniscus]

    if phase_if is None:  # straight line
        bottom_left = factory.addPoint(crys.params.X0[0], crys.params.X0[1], 0)
        bottom_right = factory.addPoint(crys.params.X0[0] + surface_points[-1, 0], crys.params.X0[1], 0)
    else:
        bottom_left = phase_if.left_boundary
        bottom_right = phase_if.right_boundary


    points = [factory.add_point(surface_points[i, 0], crys.params.X0[1] + current_length + seed_length - surface_points[i, 1], 0) for i in range(surface_points.shape[0] - 1)]
    points.append(bottom_right)  # don't use coordinates from surface_point to get proper phase_if 
    lines = [factory.add_line(points[i], points[i+1]) for i in range(len(points) - 1)]

    left = factory.addLine(bottom_left, points[0])

    if phase_if is None:
        bottom = factory.addLine(points[-1], bottom_left)
        loop = factory.addCurveLoop([bottom, left] + lines)
    else:
        dimtags = [(1, tag) for tag in lines]
        # factory.fragment(dimtags, phase_if.dimtags)
        loop = factory.addCurveLoop([phase_if.geo_id, left] + lines)
        model.remove_shape(phase_if)
    body = factory.addSurfaceFilling(loop)
    if dim == 3:
        body = rotate(body)
        # factory.rotate([(3, body)], 0, 0, 0, 0, 1, 0, np.pi/2)
    crys.geo_ids = [body]
    crys.set_interface(melt)

    return crys

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
    """2D inductor, defined as a couple of circles

    Args:
        model (Model): objectgmsh model
        dim (int): dimension
        d (float): diameter of windings
        d_in (flaot): inner diameter of windings (internal cooling)
        X0 (float): origin, center of bottom winding
        g (float, optional): gap between windings. Defaults to 0.
        n (int, optional): number of windings. Defaults to 1.
        char_l (int, optional): mesh size characteristic length. Defaults to 0.
        T_init (float, optional): initial temperature. Defaults to 273.15.
        material (str, optional): material name. Defaults to "".
        name (str, optional): shape name. Defaults to "inductor".

    Returns:
        Shape: objectgmsh shape
    """
    # X0: center of bottom winding
    ind = Shape(model, dim, name)
    ind.params.d = d
    ind.params.d_in = d_in
    ind.params.g = g
    ind.params.n = n
    ind.params.X0 = X0
    ind.params.T_init = T_init
    ind.params.material = material
    ind.params.area = np.pi * (d**2 - d_in**2) / 4
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

def heater(model, dim, r_in, h_over_floor, h, t, vessel, char_l=0, T_init=700, material="", name="heater"):
    heater = Shape(model, dim, name)
    heater.params.r_in = r_in
    heater.params.h_over_floor = h_over_floor
    heater.params.h = h
    heater.params.t = t
    heater.params.X0 = [r_in, vessel.params.X0[1] + vessel.params.t + h_over_floor]
    heater.params.T_init = T_init
    heater.params.material = material
    if char_l == 0:
        heater.mesh_size = t/4
    else:
        heater.mesh_size = char_l
    
    heater.geo_ids = [factory.add_rectangle(heater.params.X0[0], heater.params.X0[1], 0, t, h)]

    return heater

def insulation(model, dim, bottom_r_in, bottom_t, side_h, side_r_in, side_t, top_r_in, top_t, vessel, char_l=0, T_init=293.15, material="", name="insulation"):
    ins = Shape(model, dim, name)
    ins.params.X0 = [bottom_r_in, vessel.params.X0[1] + vessel.params.t]
    ins.params.material = material
    if char_l == 0:
        ins.mesh_size = bottom_t / 10
    else:
        ins.mesh_size = char_l

    bottom = factory.add_rectangle(ins.params.X0[0], ins.params.X0[1], 0, side_r_in + side_t - bottom_r_in, bottom_t)
    ins.geo_ids.append(bottom)
    if side_h != 0:
        side = factory.add_rectangle(side_r_in, ins.params.X0[1] + bottom_t, 0, side_t, side_h)
        factory.fragment([(dim, bottom)], [(dim, side)])
        ins.geo_ids.append(side)
        if top_t != 0:
            top = factory.add_rectangle(top_r_in, ins.params.X0[1] + bottom_t + side_h, 0, side_r_in + side_t - top_r_in, top_t)
            factory.fragment([(dim, side)], [(dim, top)])
            ins.geo_ids.append(top)
    ins.set_interface(vessel)
    return ins

def axis_top(
    model,
    dim,
    bot_shape,
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
    try:
        ax.params.X0 = [0, bot_shape.params.X0[1] + bot_shape.params.l]
    except AttributeError:
        ax.params.X0 = [0, bot_shape.params.X0[1] + bot_shape.params.h]
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
    ax.set_interface(bot_shape)
    if vessel is not None:
        ax.set_interface(vessel)

    return ax

def exhaust_hood(
    model,
    dim,
    points,
    y_bot=0,  # coordinate of vessel bottom surface
    char_l=0,
    T_init=273.15,
    material="",
    name="exhaust_hood",
):
    exh = Shape(model, dim, name)
    exh.params.material = material
    exh.params.T_init = T_init
    exh.mesh_size = 0.01 if char_l == 0 else char_l

    points = np.array(points)
    points[:, 1] += y_bot
    pts = [factory.add_point(x[0], x[1], 0) for x in points.tolist()]
    lines = [factory.add_line(pts[i-1], pts[i]) for i in range(len(pts))]
    loop = factory.add_curve_loop(lines)
    surf = factory.add_surface_filling(loop)
    if dim == 2:
        exh.geo_ids = [surf]
    elif dim == 3:
        exh.geo_ids = [rotate(surf)]
    factory.synchronize()
    return exh

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
    shapes = model.get_shapes(dim)
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
    if len(shapes) > 0:
        sur.geo_ids = cut([(dim, body)], dim_tags, False)
    else:
        sur.geo_ids = [body]

    for shape in shapes:
        sur.set_interface(shape)
    model.synchronize()
    return sur

def adjust_circle_center(x1, x2, r, xc):
    """Compute correct circle arc center coordinate from approximate
    value (in 2D).

    Args:
        x1 (list / np.array): Start point of circle arc
        x2 (list / np.array): End point of circle arc
        r (float): Circle arc radius
        xc (list / np.array): Approximate circle center
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    xc = np.array(xc)

    if np.linalg.norm(x2 - x1) >= 2* r:
        raise ValueError(f"Invalid circle arc with radius {r}.")

    a = 0.5 * (x2[0] - x1[0])
    b = x2[1] - x1[1]
    c = 0.5 * (x2[1] - x1[1])
    d = x2[0] - x1[0]

    p = (2 * c * d - 2 * a * b) / (b**2 + d**2)
    q = (a**2 + c**2 - r**2) / (b**2 + d**2)

    alpha1 = - p / 2 + (p**2 / 4 - q)**0.5
    alpha2 = - p / 2 - (p**2 / 4 - q)**0.5

    center1 = x1 + np.array([a - alpha1 * b, c + alpha1 * d])
    center2 = x1 + np.array([a - alpha2 * b, c + alpha2 * d])

    if np.linalg.norm(center1 - xc) < np.linalg.norm(center2 - xc):
        return center1
    else:
        return center2

def create_lines(points, coordinate_update=[0, 0, 0]):
    if len(points[0]) == 2:
        pts = [occ.addPoint(x[0] + coordinate_update[0], x[1] + coordinate_update[1], 0) for x in points]
    else:
        pts = [occ.addPoint(x[0] + coordinate_update[0], x[1] + coordinate_update[1], x[2] + coordinate_update[2]) for x in points]
    return [occ.add_line(pts[i], pts[i+1]) for i in range(len(pts) - 1)]
        
def create_spline(points, coordinate_update=[0, 0, 0]):
    if len(points[0]) == 2:
        pts = [occ.addPoint(x[0] + coordinate_update[0], x[1] + coordinate_update[1], 0) for x in points]
    else:
        pts = [occ.addPoint(x[0] + coordinate_update[0], x[1] + coordinate_update[2], x[2] + coordinate_update[2]) for x in points]
    return occ.addSpline(pts)

def create_arc(start, center, end, radius=0, coordinate_update=[0, 0, 0]):
    if radius > 0:
        center = adjust_circle_center(start, end, radius, center)
    if len(start) == 2:
        p0 = occ.addPoint(start[0] + coordinate_update[0], start[1] + coordinate_update[1], 0)
        p1 = occ.addPoint(center[0] + coordinate_update[0], center[1] + coordinate_update[1], 0)
        p2 = occ.addPoint(end[0] + coordinate_update[0], end[1] + coordinate_update[1], 0)
    else:
        p0 = occ.addPoint(start[0] + coordinate_update[0], start[1] + coordinate_update[1], start[2] + coordinate_update[2])
        p1 = occ.addPoint(center[0] + coordinate_update[0], center[1] + coordinate_update[1], center[2] + coordinate_update[2])
        p2 = occ.addPoint(end[0] + coordinate_update[0], end[1] + coordinate_update[1], end[2] + coordinate_update[2])
    # occ.synchronize()
    # gmsh.fltk.run()
    return occ.add_circle_arc(p0, p1, p2)

def create_2d_shape(model, name, outline, mesh_size=0, material="", T_init=0, coordinate_update=[0, 0]):
    shape = Shape(model, 2, name)
    shape.params.material = material
    shape.params.T_init = T_init
    if mesh_size > 0:
        shape.mesh_size = mesh_size

    lines = []
    for element in outline:
        type_ = element.pop("type")
        if type_ == "line":
            newlines = create_lines(**element, coordinate_update=coordinate_update)
        elif type_ == "spline":
            newlines = [create_spline(**element, coordinate_update=coordinate_update)]
        elif type_ == "circle-arc":
            newlines = [create_arc(**element, coordinate_update=coordinate_update)]
        else:
            raise TypeError(f"Unknown outline type '{type_}'.")
        if len(lines) > 0:
            occ.fragment([(1, lines[-1])], [(1, newlines[0])])
        lines += newlines
    occ.fragment([(1, lines[0])], [(1, lines[-1])])
    shape.geo_ids = [occ.add_surface_filling(occ.add_curve_loop(lines))]
    occ.synchronize()

    return shape
        
def create_melt_shape(model, name, maximum_outline, bottom_y_coordinate, melt_height, crystal_radius, rho, gamma, beta, g, phase_if=None, mesh_size=0, material="", T_init=0, coordinate_update=[0, 0]):
    # shape with whole crucible volume
    shape = create_2d_shape(model, name, maximum_outline, mesh_size=mesh_size, material=material, coordinate_update=coordinate_update, T_init=T_init)
    
    # cut true shape out of that
    bb = shape.bounding_box
    x_max = bb[3]
    y_max = bb[4]
    y_min = bottom_y_coordinate + melt_height

    # meniscus
    # according to Landau87
    a = (2 * gamma / (rho * g)) ** 0.5  # capillary constant
    h = a * (1 - 1 * np.sin(beta / 360 * 2 * np.pi)) ** 0.5
    z = np.linspace(0, h, 100)[1:]
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
    if meniscus_x.max() >= x_max:  # meniscus longer than melt: cut
        melt_surface_line = False
        for i in range(len(meniscus_x)):
            if meniscus_x[-(i + 1)] > x_max:
                break
        meniscus_x = meniscus_x[-(i + 1) :]
        meniscus_y = meniscus_y[-(i + 1) :]
        meniscus_x[0] = x_max
    else:  # meniscus shorter than melt
        melt_surface_line = True
    meniscus_y += -meniscus_y.min() + y_min
    if phase_if is None:
        y_meniscus = meniscus_y.max()
    else:
        y_meniscus = phase_if.params.points[0][1]
        meniscus_x[-1] = phase_if.params.points[0][0]
        meniscus_y[-1] = phase_if.params.points[0][1]
    # print(list(meniscus_x))
    # print(list(meniscus_y))
    meniscus = create_spline(np.array([meniscus_x, meniscus_y]).T)
    if phase_if == None:
        points = [
            [crystal_radius, y_meniscus],
            [0, y_meniscus],
            [0, y_max],
            [x_max, y_max],
            [x_max, y_min]
        ]
        if melt_surface_line:
            points.append([meniscus_x[0], meniscus_y[0]])
        lines = create_lines(points)  # top, right of cut-surface
        occ.fragment([(1, lines[0])], [(1, meniscus)])
        occ.fragment([(1, lines[-1])], [(1, meniscus)])
        lines.append(meniscus)
    else:
        points = [
            phase_if.params.points[-1],
            [0, y_max],
            [x_max, y_max],
            [x_max, y_min]
        ]
        if melt_surface_line:
            points.append([meniscus_x[0], meniscus_y[0]])
        lines = create_lines(points)  # top, right of cut-surface
        occ.fragment([(1, lines[0])], phase_if.dimtags)
        occ.fragment(phase_if.dimtags, [(1, meniscus)])
        occ.fragment([(1, lines[-1])], [(1, meniscus)])
        lines.append(meniscus)
        lines += phase_if.geo_ids

    cut_surface = occ.add_surface_filling(occ.add_curve_loop(lines))
    occ.cut(shape.dimtags, [(2, cut_surface)])
    occ.synchronize()

    return shape, y_meniscus

def create_axis_shape(model, name, y_bot, y_top, r, y_bot_update=0, y_top_update=0, mesh_size=0, material="", T_init=0):
    y_bot += y_bot_update
    y_top += y_top_update
    
    shape = Shape(model, 2, name)
    shape.params.material = material
    shape.params.T_init = T_init
    if mesh_size > 0:
        shape.mesh_size = mesh_size
    
    points = [
        [0, y_bot],
        [r, y_bot],
        [r, y_top],
        [0, y_top],
        [0, y_bot]
    ]
    lines = create_lines(points)
    shape.geo_ids = [occ.add_surface_filling(occ.add_curve_loop(lines))]
    occ.synchronize()
    return shape

def create_crystal_shape(model, name, outline, mesh_size, crystal_length, base_crystal_length, meniscus_position, phase_if=None, material="", T_init=0):
    dL = base_crystal_length - crystal_length

    shape = create_2d_shape(model, name, outline, mesh_size=mesh_size, material=material, coordinate_update=[0, meniscus_position - dL], T_init=T_init)
    x_max = shape.bounding_box[3]
    # x_max += 1e-3  # add tolerance to avoid error in mesh generation

    if dL > 0:  # cut crystal to required length
        if phase_if is None:
            points = [
                [0, meniscus_position],
                [x_max, meniscus_position],
                [x_max, meniscus_position - dL],
                [0, meniscus_position - dL],
                [0, meniscus_position],
            ]
            lines = create_lines(points)
        else: 
            if x_max > phase_if.params.points[0][0]:
                points = [
                    phase_if.params.points[0],
                    [x_max, meniscus_position],
                    [x_max, meniscus_position - dL],
                    [0, meniscus_position - dL],
                    phase_if.params.points[-1]
                ]
            else:  # should be impossible
                points = [
                    phase_if.params.points[0],
                    [x_max, meniscus_position - dL],
                    [0, meniscus_position - dL],
                    phase_if.params.points[-1]
                ]
            lines = create_lines(points)
            occ.fragment([(1, lines[-1]), (1, lines[0])], phase_if.dimtags)
            lines += phase_if.geo_ids
        cut_surface = occ.add_surface_filling(occ.add_curve_loop(lines))
        occ.cut(shape.dimtags, [(2, cut_surface)])
        occ.synchronize()

    y_crystal_top = meniscus_position + crystal_length
    return shape, y_crystal_top

def create_heater_shape(model, name, dx, dy, bot_left_points, mesh_size=0, material="", T_init=0):
    heater = Shape(model, 2, name)
    heater.params.material = material
    heater.params.T_init = T_init
    if mesh_size > 0:
        heater.mesh_size = mesh_size
    for x0 in bot_left_points:
        heater.geo_ids.append(occ.add_rectangle(x0[0], x0[1], 0, dx, dy))
    return heater

def cylinder_to_cartesian(r, phi, h):  # phi in deg
    x = r * np.sin(phi / 180 * np.pi)
    z = r * np.cos(phi / 180 * np.pi)
    return (x, h, z)

def coil_from_points(
    model,
    dim,
    spline_points,
    radius_pipe,
    dy=0,
    char_l=0,
    T_init=273.15,
    material="",
    name="inductor",
):
    coil = Shape(model, dim, name)
    if char_l == 0:
        coil.mesh_size = radius_pipe / 5
    else:
        coil.mesh_size = char_l
    coil.params.T_init = T_init
    coil.params.material = material
    coil.params.radius_pipe = radius_pipe
    coil.params.supply_bot_h = spline_points[0][2] + dy
    coil.params.supply_top_h = spline_points[-1][2] + dy

    occ_points = []
    for p in spline_points:
        x, y, z = cylinder_to_cartesian(p[0], p[1], p[2])
        occ_points.append(occ.add_point(x, y + dy, z))
    spline = occ.add_spline(occ_points)
    wire = occ.add_wire([spline])

    p0 = spline_points[0]
    x0, y0, z0 = cylinder_to_cartesian(p0[0], p0[1], p0[2])
    disk = occ.add_disk(x0, y0 + dy, z0, radius_pipe, radius_pipe)
    pipe = occ.addPipe([(2, disk)], wire)
    coil.geo_ids = [pipe[0][1]]

    return coil
