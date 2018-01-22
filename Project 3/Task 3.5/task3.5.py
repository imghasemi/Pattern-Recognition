#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:29:06 2018

@author: qn
"""

import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


def plot_data_and_fit(h, w, x, y):
    plt.plot(h, w, 'ko', x, y, 'r-')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()

def trsf(x):
    return x / 100.

if __name__ == "__main__":

    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]

    # Task 1.1
    outlierInd = np.where( X[:,0] != -1 )
    X, y = X[outlierInd], y[outlierInd]

    # let's transpose t data matrix
    X = X.T
    hgt = X[1,:]
    wgt = X[0,:]
    xmin = hgt.min()-15
    xmax = hgt.max()+15
    ymin = wgt.min()-15
    ymax = wgt.max()+15
    n = 10
    x = np.linspace(xmin, xmax, 100)

    # method 1:
    # regression using ployfit
    c = poly.polyfit(hgt, wgt, n)
    y = poly.polyval(x, c)
    plot_data_and_fit(hgt, wgt, x, y)

    # method 2:
    # regression using the Vandermonde matrix and pinv
    X = poly.polyvander(hgt, n)
    c = np.dot(la.pinv(X), wgt)
    y = np.dot(poly.polyvander(x,n), c)
    plot_data_and_fit(hgt, wgt, x, y)

    # method 3:
    # regression using the Vandermonde matrix and lstsq
    X = poly.polyvander(hgt, n)
    c = la.lstsq(X, wgt)[0]
    y = np.dot(poly.polyvander(x,n), c)
    plot_data_and_fit(hgt, wgt, x, y)

    # method 4:
    # regression on transformed data using the Vandermonde
    # matrix and either pinv or lstsq
    X = poly.polyvander(trsf(hgt), n)
    c = np.dot(la.pinv(X), wgt)
    y = np.dot(poly.polyvander(trsf(x),n), c)
    plot_data_and_fit(hgt, wgt, x, y)