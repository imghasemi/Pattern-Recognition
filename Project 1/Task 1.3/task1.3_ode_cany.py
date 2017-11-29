#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:58:20 2017

@author: canyuce
"""

import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# read data from file
h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

# Remove leading zeros.
while h[0] == 0:
    h = np.delete(h, 0)

# Generate x-Values for the histogram
xValues = np.arange(1, len(h) + 1)

# Data generation for the ODE Solution
hist = [];
for i in range(len(h)):
    hist = np.append(hist, [xValues[i]] * int(h[i]))

N = len(hist)
sum_log_di = np.sum(np.log(hist))

# function that returns [dL/dK, dL/da]
def model(y, t):
    K, a = y[0], y[1]
    d_K = N / K - N * math.log(a) + sum_log_di - np.sum( ((hist / a) ** K) * 
                               np.log(hist / a))
    d_a = (K / a) * (np.sum((hist / a) ** K) - N)
    return [d_K, d_a]

# initial condition
K, a = 1., 1.

# time points
t = np.linspace(0,100,1000)

# solve ODE
y = odeint(model,[K,a],t)

# plot results
K, a = y[-1];
filename = "fit_ode.pdf"

plt.plot(xValues, h)
plt.axis([xValues[0], xValues[len(xValues) - 1], 0, max(h) + 10])

# Scaling factor for the Weibull fit was derived by setting: 
# scale_factor*integral[Weibull] = Area Under the Curve for Histogram
plt.plot(sum(h) * (K / a) * ((xValues / a) ** (K - 1)) * np.exp(-1 * 
         ((xValues / a) ** K)), 'r')
plt.legend(('Google Data', 'Scaled Weibull Fit(ODE)'))

plt.savefig(filename, facecolor='w', edgecolor='w',papertype=None,
            format='pdf', transparent=False, bbox_inches='tight', 
            pad_inches=0.1)
plt.close()
