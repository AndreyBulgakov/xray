# coding=utf-8
from collections import namedtuple
from pprint import pprint as pp
import sys
import ccross

Pt = namedtuple('Pt', 'x, y')  # Point
Edge = namedtuple('Edge', 'a, b')  # Polygon edge from a to b
Poly = namedtuple('Poly', 'name, edges')  # Polygon

_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min


# Crossing number algorithm realization
def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    '''
    a, b = edge
    if a.y > b.y:
        a, b = b, a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    intersect = False

    if (p.y > b.y or p.y < a.y) or (
                p.x > max(a.x, b.x)):
        return False

    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect


def _odd(x): return x % 2 == 1


def ispointinside(p, poly):
    ln = len(poly)
    return _odd(sum(rayintersectseg(p, edge) for edge in poly.edges))


# def point_inside_polygon(x, y, poly):
#     global xinters
#     n = len(poly)
#     inside = False
#
#     p1x, p1y = poly[0]
#     for i in range(n + 1):
#         p2x, p2y = poly[i % n]
#         if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
#             if p1y != p2y:
#                 xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#             if p1x == p2x or x <= xinters:
#                 inside = not inside
#         p1x, p1y = p2x, p2y
#
#     return inside

def point_inside_polygon(x, y, poly):
    return ccross.point_inside_polygon(x, y, poly)


def polypp(poly):
    print "\n  Polygon(name='%s', edges=(" % poly.name
