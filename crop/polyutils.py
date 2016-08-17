# coding=utf-8
import datetime
import dicom

from CrossingNumber import Pt, Edge, Poly
import pickle
import os
import CrossingNumber
import numpy as np
map_dir = 'maps/'
poly_dir = 'polygons/'
data_dir = '../01-Apr-2015/'

def createPoly(name, xarr, yarr):
    xarr.append(xarr[0])
    yarr.append(yarr[0])
    poly = Poly(name=name, edges=[])
    for i in range(1, len(xarr)):
        previousPoint = Pt(x=xarr[i - 1], y=yarr[i - 1])
        currentPoint = Pt(x=xarr[i], y=yarr[i])
        edge = Edge(a=previousPoint, b=currentPoint)
        poly.edges.append(edge)
    return poly

def save_polys(filename, leftLungPoly, rightLungPoly):
    polys = [leftLungPoly, rightLungPoly]
    pickle.dump(polys, open(filename, "wb"))


def load_polys(filename):
    polys = pickle.load(open(filename, "rb"))
    leftLungPoly = polys[0]
    rightLungPoly = polys[1]
    return (leftLungPoly, rightLungPoly)