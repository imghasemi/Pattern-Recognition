#!/usr/bin/env python2
# Ensures you use only the python2.X on your machine
# Author: Mohammadali Ghasemi

import numpy as np
import math
import matplotlib.pyplot as plt
import time
start_time = time.time()

# read data from file
h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]
while h[0] == 0:
    h = np.delete(h, 0)
N = int(np.sum(h))
# Generate x-Values for the histogram
xValues = np.arange(1, len(h) + 1, dtype='float')


def newtonParametersCalculator(k, a):
    # Constant calculation are done here
    sum_log_di = np.sum(h * np.log(xValues))
    sum_di_a_k = np.sum(h * (xValues / a) ** k)

    # nearly 10 times faster than working with histogram
    dl_dk = N / k - N * np.log(a) + sum_log_di - np.sum(h * ((xValues / a) ** k) * np.log(xValues / a))
    dl_da = (k / a) * (sum_di_a_k - N)
    d2l_dk = -N / (k ** 2) - np.sum(h * ((xValues / a) ** k) * (np.log(xValues / a)) ** 2)
    d2l_da = (k / (a ** 2)) * (N - (k + 1) * sum_di_a_k)
    d2l_dkda = (1 / a) * sum_di_a_k + (k / a) * np.sum(h * ((xValues / a) ** k) * np.log(xValues / a)) - N / a

    return np.array(np.matmul(np.linalg.inv(np.matrix([[d2l_dk, d2l_dkda], [d2l_dkda, d2l_da]])),
                              np.array([-dl_dk, -dl_da])) + np.array([k, a]))[0]

def iterationFunction(k, a, n):
    # 20 iterations, no termination
    oldParameters = np.array([k, a])
    for i in range(n):
        newParameters = newtonParametersCalculator(oldParameters[0], oldParameters[1])
        oldParameters = newParameters
    return newParameters

if __name__ == "__main__":
    # read data from file
    h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

    # Remove leading zeros.
    while h[0] == 0:
        h = np.delete(h, 0)

    # Generate x-Values for the histogram
    xValues = np.arange(1, len(h) + 1)

    # Data generation for Newton's Method.
    histogramOfX = [];
    for i in range(len(h)):
        histogramOfX = np.append(histogramOfX, [xValues[i]] * int(h[i]))

    parameters = iterationFunction(1, 1, 20)
    k = parameters[0]
    a = parameters[1]
    filename = "fit_newton.pdf"

    # plot fitted distribution with observed samples
    plt.plot(xValues, h, 'k')
    plt.axis([xValues[0], xValues[len(xValues) - 1], 0, max(h) + 10])

    plt.plot(sum(h) * (k / a) * ((xValues / a) ** (k - 1)) * np.exp(-1 * ((xValues / a) ** k)), 'r')
    plt.legend(('Google Data', 'Scaled Weibull Fit(Newton)'))

    plt.savefig(filename, facecolor='w', edgecolor='w', papertype=None,
                format='pdf', transparent=False, bbox_inches='tight',
                pad_inches=0.1)
    plt.close()
    print("--- %s seconds ---" % (time.time() - start_time))

