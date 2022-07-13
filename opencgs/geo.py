from objectgmsh import *


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
