#!/usr/bin/env python2
# Ensures you use only the python2.X on your machine
# Author: Mohammadali Ghasemi

import numpy as np
import math
import matplotlib.pyplot as plt


# from PartialDifferentialEqSolver import PDESolver as PDFS


def newtonParametersCalculator(k, a, histogramData):
    # We Input parameters 'k' and 'a' (alpha) into the function.
    N = len(histogramData)

    # Calculated all the matrix elements of the Newtonian Method.
    B1 = N / k - N * math.log(a) + np.sum(np.log(histogramData)) - np.sum(
        ((histogramData / a) ** k) * np.log(histogramData / a))
    B2 = (k / a) * (np.sum((histogramData / a) ** k) - N)
    M11 = -N / (k ** 2) - np.sum(((histogramData / a) ** k) * (np.log(histogramData / a)) ** 2)
    M22 = (k / ((a) ** 2)) * (N - (k + 1) * np.sum((histogramData / a) ** k))
    M12 = M21 = (1 / a) * np.sum((histogramData / a) ** k) + (k / a) * np.sum(
        ((histogramData / a) ** k) * np.log(histogramData / a)) - N / a
    return \
        np.array(

            np.matmul(np.linalg.inv(np.matrix([[M11, M12], [M21, M22]])), np.array([-B1, -B2])) + np.array([k, a]))[0]


def iterationFunction(k, a, n, histogram):
    # Normally, a termination condition would also be specified, but since we have observed that 'k' and 'a'
    # terminate to a fixed value, when started from 'k'=1 and 'a'=1 in 20 iterations, no termination condition was
    # written

    # Termination condition would be of the form while newPara!=oldPara (If they don't terminate exactly,
    # we can specify a tolerance value)
    oldParameters = np.array([k, a])
    for i in range(n):
        newParameters = newtonParametersCalculator(oldParameters[0], oldParameters[1], histogram)
        oldParameters = newParameters
    return newParameters


if __name__ == "__main__":
    # numpy array was chosen as the calculations of the matrix elements becomes very simple
    h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

    # Delete leading zeros.
    while h[0] == 0:
        h = np.delete(h, 0)

    # Generate x-Values for the histogram
    xValues = np.arange(1, len(h) + 1)

    # Data generation for Newton's Method.
    histogramOfX = [];
    for i in range(len(h)):
        histogramOfX = np.append(histogramOfX, [xValues[i]] * int(h[i]))

    parameters = iterationFunction(1, 1, 20, histogramOfX);
    k = parameters[0];
    a = parameters[1];

    plt.plot(xValues, h)
    plt.axis([xValues[0], xValues[len(xValues) - 1], 0, max(h) + 10])

    # Scaling factor for the Weibull fit was derived by setting: scale_factor*integral[Weibull] = Area Under the
    # Curve for Histogram
    plt.plot(sum(h) * (k / a) * ((xValues / a) ** (k - 1)) * np.exp(-1 * ((xValues / a) ** k)), 'r')
    plt.legend(('Google Data', 'Scaled Weibull Fit'))
    plt.show()
