#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:54:18 2017
Pattern Recognition Assignment 1 Task 1.5
estimating the dimension of fractal objects in an image
@author: qn


step 1, produce binarization image. I directly use professor's code. Any people can help me change it?
step 2, produce box scalering sets S and it's correspondence count X, S obviously from 1/2 to 1/(2^(L-2)), while L =log2(height of image)
step 3, fitting a line, use linear least square, print it's slope D. Fitting line is saved as "plot_task_1_5_"+imgName+".pdf" in the file Dir

"""

import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# here can change the image file name
imgName = 'lightning-3'

"""
Get the binary image. This function is directly copied from Professor's code.
"""


def foreground2BinImg(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - img.filters.gaussian_filter(f, sigma=1.00,
                                                                                                 mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d < 0.1 * m] = 0
    d[d >= 0.1 * m] = 1
    return img.morphology.binary_closing(d)


"""
give specified scalar and then return its number of correspondent boxes.
Image with box can be shown through changing the Param if_box_image = true.
g = image 
L = scala number, scala = 1/(2^L)
if_box_image = switcher of showing image with box
return count = number of correspondent boxes
"""


def scalarboxcounting(g, L, if_box_image=False):
    scala = 2 ** L
    sub_width = g.shape[0] / scala
    count = 0

    if if_box_image:
        # get box image
        fig, ax = plt.subplots(1)
        f = g.astype(np.float)
        f[f > 0] = 180
        f[f == 0] = 0
        ax.imshow(f)

    for x in range(0, scala):
        for y in range(0, scala):
            sub_image = g[x * sub_width:(x + 1) * sub_width, y * sub_width:(y + 1) * sub_width]
            if not (sub_image == False).all():
                count += 1
                if if_box_image:
                    # draw a red square in that subimage
                    rect = patches.Rectangle((y * sub_width, x * sub_width), sub_width, sub_width, linewidth=1,
                                             edgecolor='r', facecolor='none');
                    ax.add_patch(rect)

    if if_box_image:
        plt.show()

    return count


"""
boxcounting, given Bin image g, return two list, S and count
g = bin image
s = list of scaling factor 
count = list of correspondent boxes number
"""


def boxcounting(g):
    s = []
    count = []
    image_width = g.shape[0]
    for i in range(1, int(np.log2(image_width) - 1)):
        count.append(scalarboxcounting(g, i, True))
        s.append(1.0 / (2 ** i))
    return s, count


def plotlogline(D, b, s, count):
    s_invese = np.divide(1, s)
    line_x = np.linspace(1, s_invese.max(), num=100)
    line_y = np.power(10 * np.ones(len(line_x)), D * np.log10(line_x) + b)
    # create a fig
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(s_invese, count, 'bo', label='data')
    axs.plot(line_x, line_y, 'r', label='fitting line')

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # set x,y scale as log
    plt.yscale('log')
    plt.xscale('log')

    # set x,y axis labels
    axs.set_xlabel("log(1/s)")
    axs.set_ylabel("logn")
    # plt.show()
    plt.savefig("plot_task_1_5_" + imgName + ".pdf", facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


"""
through s and count to fit line:
    D*log(1/s)+b=log(count) 
Let x = log(1/s)
    y = log(count)
So it becomes Dx + b = y
use least square method to calculate D and b
"""


def linefitting(s, count):
    x = np.log10(np.divide(1, s))
    y = np.log10(count)
    # to solve the linear least square, we should rewrite equation Dx+b=y => XA = y
    # where X = [x,1] (x should be a col vector), and A = [[D],[b]], and y = y.
    # then through Linear least square Derivation, we know
    # A = ((X.T*X).inv)*X.t*y
    X = np.vstack([x, np.ones(len(x))]).T
    D, b = np.linalg.lstsq(X, y)[0]
    plotlogline(D, b, s, count)
    return D, b


if __name__ == "__main__":
    f = msc.imread(imgName + '.png', flatten=True).astype(np.float)
    g = foreground2BinImg(f)
    s, count = boxcounting(g)
    D, b = linefitting(s, count)
    print "slope D = " + str(D)

