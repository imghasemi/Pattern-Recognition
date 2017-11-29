#!/usr/bin/env python2.7

import numpy as np
import timeit as t

"""
Here we took the part of formula which we tried to optimize.
The no_optimisation function calculates the formula before optimisations.
The optimisation function calculates the formula after optimisations.
"""


def no_optimisation(histogramData):
    a = 1
    k = 1
    return np.sum(((histogramData / a) ** k) * np.log(histogramData / a))


def optimisation(xValues):
    a = 1
    k = 1
    return np.sum(h * ((xValues / a) ** k) * np.log(xValues / a))


if __name__ == '__main__':
    h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

    while h[0] == 0:
        h = np.delete(h, 0)

    xValues = np.arange(1, len(h) + 1)
    histogramOfX = []
    number_of_test = 100
    for i in range(len(h)):
        histogramOfX = np.append(histogramOfX, [xValues[i]] * int(h[i]))


    print('average time for summations (approach without '
          'optimisation):\n%1.20f' %
          (t.timeit('no_optimisation(histogramOfX)',
                   setup='from __main__ import no_optimisation, histogramOfX',
                   number=number_of_test)/float(number_of_test)))

    print('average time for multiplication (optimised approach):\n%1.20f' %
          (t.timeit('optimisation(xValues)',
                    setup='from __main__ import optimisation, xValues',
                    number=number_of_test)/float(number_of_test)))

