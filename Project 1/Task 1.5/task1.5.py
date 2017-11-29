#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:54:18 2017
Pattern Recognition Assignment 1 Task 1.5
estimating the dimension of fractal objects in an image
@author: qn

step 1, Apply binarization procedure.
step 2, Produce box scaling sets S and compute corresponding box count X, 
        S obviously from 1/2 to 1/(2^(L-2)), while L =log2(height of image)
step 3, Fit a line, use linear least square, estimate it's slope D. 
        Line plot is saved as "plot_task_1_5_"+imgName+".pdf" into current
        directory.
"""

# lightning slope D = 1.57768502708
# tree slope D = 1.84639005655

import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# here one can change the image file name
imgName = 'tree-2'

"""
Get the binary image. This function is directly copied from Professor's code.
"""
def foreground2BinImg(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - \
        img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d < 0.1 * m] = 0
    d[d >= 0.1 * m] = 1
    return img.morphology.binary_closing(d)

"""
Returns the number of boxes corresponding to each binarization level.
im = binary image 
L  = scala number, scala = 1/(2^L)

return count = number of boxes corresponding to given scale.
"""
def scalarboxcounting(im, L):
    scala = 2 ** L
    sub_width = im.shape[0] / scala
    count = 0

    # get box image
    fig, ax = plt.subplots(1)
    f = im.astype(np.float)
    f[f > 0] = 180
    f[f == 0] = 0
    ax.imshow(f)

    for x in range(0, scala):
        for y in range(0, scala):
            sub_image = im[x * sub_width:(x + 1) * sub_width, y * 
                           sub_width:(y + 1) * sub_width]
            if (sub_image == True).any():
                count += 1
                # draw a red square in that subimage
                rect = patches.Rectangle((y * sub_width, x * sub_width), 
                                         sub_width, sub_width, 
                                         linewidth=1,edgecolor='r', 
                                         facecolor='none');
                ax.add_patch(rect)

    filename = "image_boxes_" + imgName + "_" + str(L) + ".png"
    plt.savefig(filename, facecolor='w', edgecolor='w',papertype=None,
        format='png', transparent=False, bbox_inches='tight', 
        pad_inches=0.1)

    return count

"""
boxcounting, given Bin image g, return two list, S and count
im    = binary image
s     = scaling factor list

return l_count = list of box count corresponding to each scale.
"""
def boxcounting(im):
    s = []
    l_count = []
    image_width = im.shape[0]
    for i in range(1, int(np.log2(image_width) - 1)):
        l_count.append(scalarboxcounting(im, i))
        s.append(1.0 / (2 ** i))
    return s, l_count


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
    leg = axs.legend(loc='upper right', shadow=True, fancybox=True, 
                     numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # set x,y scale as log
    plt.yscale('log')
    plt.xscale('log')

    # set x,y axis labels
    axs.set_xlabel("log(1/s)")
    axs.set_ylabel("logn")
    # plt.show()
    plt.savefig("plot_task_1_5_" + imgName + ".pdf", facecolor='w', 
                edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

"""
Uses s and box counts to fit line using least sqaures.
    
D*log(1/s)+b=log(count) 
Let x = log(1/s)
    y = log(l_count)
    So it becomes Dx + b = y
    Then uses least square method to calculate D and b
"""
def linefitting(s, l_count):
    x = np.log10(np.divide(1, s))
    y = np.log10(l_count)
    # to solve the linear least square, we should rewrite equation 
    # Dx+b=y => XA = y
    # where X = [x,1] (x should be a col vector), and A = [[D],[b]], and y = y.
    # then through Linear least square Derivation, we know
    # A = ((X.T*X).inv)*X.t*y
    X = np.vstack([x, np.ones(len(x))]).T
    D, b = np.linalg.lstsq(X, y)[0]
    plotlogline(D, b, s, l_count)
    return D, b


if __name__ == "__main__":
    f = msc.imread(imgName + '.png', flatten=True).astype(np.float)
    g = foreground2BinImg(f)
    s, l_count = boxcounting(g)
    D, b = linefitting(s, l_count)
    print "slope D = " + str(D)
