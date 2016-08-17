import copy
import os
import dicom
import pylab
import matplotlib.pyplot as plt
import numpy as np
import time
import compress
import time
import math

def check(val,arr):
    res = [False, False, False, False, False, False, False, False]
    for l in range(len(arr)):
        for i in range(len(arr[l][0])):
            for j in range(1, 9):
                if(arr[l][0][i] == val[j-1][0] and arr[l][1][i] == val[j-1][1]):
                    res[j-1] = True
    return res

def pixelsfix(arrx, i0, j0, step):

    arr = arrx[i0:i0+step, j0:j0+step]
    hist = np.histogram(arr, 30)
    colors = hist[0]
    bittenInd = np.where((colors >= 1) & (colors <= 5))[0]

    s = l = 0
    hist[1][-1] += 1
    sum = 0
    indexesFull = []
    for w in range(len(bittenInd)):
        indexesFull.append(np.where((arr >= hist[1][bittenInd[w]]) & (arr < hist[1][bittenInd[w]+1])))
    for w in range(0, len(bittenInd)):
        indexes = indexesFull[w]
        #print indexes
        for t in range(0, len(indexes[0])):
            tt = s = l = 0
            sum += 1
            i = indexes[0][t]
            j = indexes[1][t]
            b = check(((i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i+1, j), (i-1, j+1), (i, j+1), (i+1, j+1)), indexesFull)
            for k in range(i-1, i+2):
                if j-1 >= 0 and k >= 0 and k < step and not b[k-i+1]:
                    s += arr[k,j-1]
                    l += 1
            if i-1 >= 0 and not b[3]:
                s += arr[i-1,j]
                l += 1
            if i+1 < step and not b[4]:
                s += arr[i+1,j]
                l += 1
            for k in range(i-1, i+2):
                if j+1 < step and k >= 0 and k < step and not b[5 + k - i + 1]:
                    s += arr[k,j+1]
                    l += 1
            if(l > 0):
                arrx[i0+i,j0+j] = s // l


def isBroken(arr):
    return np.mean(arr) > 10000


