from pyelmer.gmsh_objects import Shape
from pyelmer.gmsh_utils import *


def line_from_points(model, points, name='line'):
    line = Shape(model, 1, name)
    line.params.points = points
    if len(points[0]) == 2:
        pts = [factory.addPoint(x[0], x[1], 0) for x in points]
    else:
        pts = [factory.addPoint(x[0], x[1], x[2]) for x in points]
    line.params.x_start = pts[0]
    line.params.x_end = pts[-1]
    line.geo_ids = [factory.addSpline(pts)]
    factory.synchronize()
    return line
