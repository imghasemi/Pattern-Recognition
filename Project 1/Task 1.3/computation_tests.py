#!/usr/bin/env python2.7

import numpy as np
import timeit as t

"""
Here we took the part of formula which we tried to optimize.
The dl_dk_init function calculates the formula before optimisations.
The dl_dk_optimised function calculates the formula after optimisations.
"""


def dl_dk_init(N, k, a, sum_log_di, histogramData):
    return N / k - N * np.log(a) + sum_log_di - \
           np.sum(((histogramData / a) ** k) * np.log(histogramData / a))


def dl_dk_optimised(N, k, a, sum_log_di, xValues):
    return N / k - N * np.log(a) + sum_log_di - \
           np.sum(h * ((xValues / a) ** k) * np.log(xValues / a))


if __name__ == '__main__':
    h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

    while h[0] == 0:
        h = np.delete(h, 0)

    xValues = np.arange(1, len(h) + 1)
    histogramOfX = []
    number_of_test = 1000
    for i in range(len(h)):
        histogramOfX = np.append(histogramOfX, [xValues[i]] * int(h[i]))

    k = 1
    a = 1
    N = len(h)

    sum_log_di = np.sum(h * np.log(xValues))

    before = t.timeit('dl_dk_init(N, k, a, sum_log_di, histogramOfX)',
                        setup='from __main__ import dl_dk_init, N, k, a, '
                        'sum_log_di, histogramOfX',
                        number=number_of_test)/float(number_of_test)

    after = t.timeit('dl_dk_optimised(N, k, a, sum_log_di, xValues)',
                        setup='from __main__ import dl_dk_optimised, N, k, a, '
                        'sum_log_di, xValues',
                        number=number_of_test)/float(number_of_test)
    print('Test numbers: %s ' % number_of_test)
    print('average time for summations '
          '(approach without optimisation):\n%1.20f' % before)
    print('average time for multiplication '
          '(optimised approach):\n%1.20f' % after)
    print('ratio before/after : %10.10f' % (before/after))



