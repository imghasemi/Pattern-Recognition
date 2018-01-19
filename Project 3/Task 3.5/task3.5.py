#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

train_file = 'whData.dat'
plot_dir = 'plots'


def main():
    dt = np.dtype([('weight', np.int), ('height', np.int), ('sex', np.str_, 1)])
    d = np.loadtxt(fname='whData.dat', comments='#', dtype=dt)
    # heights with missed weight
    h_ = d[np.where(d['weight'] < 0)]['height']
    # remove outliers
    d = d[np.where(d['weight'] >= 0)]
    wgt = d['weight']
    hgt = d['height']

    xmin = hgt.min() - 15
    xmax = hgt.max() + 15
    ymin = wgt.min() - 15
    ymax = wgt.max() + 15

    def plot_data_and_fit(h, w, x, y, descr='plot', output_f='plot.png'):
        fig = plt.figure()
        fig.suptitle(descr)
        plt.plot(h, w, 'ko', x, y, 'r-')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        output_f = '/'.join([plot_dir, output_f])
        plt.savefig(output_f, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def trsf(x):
        return x / 100.

    n = 10
    x = np.linspace(xmin, xmax, 100)
    # method 1:
    # regression using ployfit
    c = poly.polyfit(hgt, wgt, n)
    y = poly.polyval(x, c)
    plot_data_and_fit(hgt, wgt, x, y, 'meth1: polyfit', 'polyfit.png')

    # method 2:
    # regression using the Vandermonde matrix and pinv
    X = poly.polyvander(hgt, n)
    c = np.dot(la.pinv(X), wgt)
    y = np.dot(poly.polyvander(x, n), c)
    plot_data_and_fit(hgt, wgt, x, y, 'meth2: Vandermonde matrix and pinv',
                      'vandermonde_pinv.png')

    # method 3:
    # regression using the Vandermonde matrix and lstsq
    X = poly.polyvander(hgt, n)
    c = la.lstsq(X, wgt)[0]
    y = np.dot(poly.polyvander(x, n), c)
    plot_data_and_fit(hgt, wgt, x, y, 'meth3: Vandermonde matrix and lstsq',
                      'vandermonde_lstsq.png')
    # method 4:
    # regression on transformed data using the Vandermonde
    # matrix and either pinv or lstsq
    X = poly.polyvander(trsf(hgt), n)
    c = np.dot(la.pinv(X), wgt)
    y = np.dot(poly.polyvander(trsf(x), n), c)
    plot_data_and_fit(hgt, wgt, x, y,
                      'meth4: transformed data using Vandermonde',
                      'vandermonde_transformed.png')


if __name__ == '__main__':
    main()