#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 00:14:44 2017
"""

import numpy as np
import matplotlib.pyplot as plt

out_file = None;

# read data as 2D array of data type 'object'
data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

# read height and weight data into 2D array (i.e. into a matrix)
X = data[:,0:2].astype(np.float32)
    
# Task 2.1
norm_const = 40
mask = (X[:,0] == -1)
h_est, h = X[mask][:,1]/norm_const, X[~mask][:,1]/norm_const
y = X[~mask][:,0]/norm_const

min_h, max_h = min(h), max(h)
min_y, max_y = min(y), max(y)

ax_x_min, ax_x_max = min_h-(max_h-min_h)/3, max_h+(max_h-min_h)/3
ax_y_min, ax_y_max = min_y-(max_y-min_y)/3, max_y+(max_y-min_y)/3

for d in [10,5,1]:
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    
    X = np.vander(h, d+1)
    # use SVD to calculate the pseudo inverse of the matrix X
    u, s, v_t = np.linalg.svd(X, full_matrices=False)
    
    # trick used in numpy.linalg.pinv
    # discard small singular values
    rcond = 1e-15
    small = s < rcond * max(s)
    s = np.divide(1.,s)
    s[small] = 0
    #w = np.dot(np.dot(np.dot(v_t.T, np.diag(s)), u.T), y)
    w = np.dot(np.linalg.pinv(X), y)
    # draw plot
    x = np.linspace(ax_x_min, ax_x_max, 1000)
    axs.set_xlim(ax_x_min, ax_x_max)
    axs.set_ylim(ax_y_min, ax_y_max)
    
    plt.xlabel('Height')
    plt.ylabel('Weight')
    
    axs.plot(x, np.polyval(w, x), label='deg=%d'%(d),color='green')
    axs.scatter(h, y, s=10, color='orange')
    axs.scatter(h_est, np.polyval(w, h_est), s=20, color='red')

    # scale x/y axis ticks
    labels = [l*norm_const for l in axs.get_xticks()]
    axs.set_xticklabels(labels)
    labels = [l*norm_const for l in axs.get_yticks()]
    axs.set_yticklabels(labels)

    # set properties of the legend of the plot
    leg = axs.legend(loc='lower right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)
    
    plt.savefig('prediction_d=%d.pdf'%(d), facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Predicted Values
#
# d=10: [ 4.19999981  4.30000019  4.17500019] [-180.5 -482.  -300.5]
# d=5:  [ 4.19999981  4.30000019  4.17500019] [ 1.375  1.625  1.5  ]
# d=1   [ 4.19999981  4.30000019  4.17500019] [ 1.56272268  1.72452927  1.52227163]
#

