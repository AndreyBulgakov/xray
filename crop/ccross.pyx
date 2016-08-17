import os

import datetime
import numpy as np
import pickle
import polyutils

map_dir = 'maps/'
poly_dir = 'polygons/'


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def create_map():
    file_list = os.listdir(poly_dir)
    for file in file_list:
        polys = polyutils.load_polys(poly_dir + file)
        # arr = dicom.read_file(data_dir+file.replace('.polys', '')).pixel_array

        width = 2340
        height = 2340
        leftPoly = polys[0]
        rightPoly = polys[1]

        print leftPoly
        print rightPoly
        minX = leftPoly.edges[0].a.x
        maxX = 0
        minY = leftPoly.edges[0].a.y
        maxY = 0

        for edge in leftPoly.edges:
            point = edge.a
            if point.x < minX:
                minX = point.x
            if point.x > maxX:
                maxX = point.x
            if point.y < minY:
                minY = point.y
            if point.y > maxY:
                maxY = point.y
        for edge in rightPoly.edges:
            point = edge.a
            if point.x < minX:
                minX = point.x
            if point.x > maxX:
                maxX = point.x
            if point.y < minY:
                minY = point.y
            if point.y > maxY:
                maxY = point.y

        print (minX, minY, maxX, maxY)

        # Create polygons for an other algorithm
        lp = []
        for edge in leftPoly.edges:
            lp.append((edge.a.x, edge.a.y))
        rp = []
        for edge in rightPoly.edges:
            rp.append((edge.a.x, edge.a.y))

        # If point in one of polygons
        # pred = lambda (x, y): point_inside_polygon(x, y, lp) or point_inside_polygon(x, y, rp)

        result = np.zeros((width, height), dtype=bool)
        for x in range(minX, maxX):
            for y in range(minY, maxY):
                result[y][x] = point_inside_polygon(x, y, lp) or point_inside_polygon(x, y, rp)
        np.savez_compressed(map_dir + file + '.map', map=result)
        print "Map for {} saved".format(file)
        print datetime.datetime.now()
