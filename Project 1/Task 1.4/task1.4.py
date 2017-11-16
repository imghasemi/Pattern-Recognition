from __future__ import division

from sympy.solvers import solve
from sympy import Symbol
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

x = Symbol('x', real = True)
y = Symbol('y', real = True)

y_ = solve(np.abs(x)**(0.5) + np.abs(y)**(0.5) - 1.0, y)
x_ = np.linspace(-1,1,1000)

# create a figure and its axes
fig = plt.figure()
axs = fig.add_subplot(111)

for y__ in y_:
    f = lambdify(x, y__)
    f_x = f(x_)
    axs.plot(x_, f_x, 'b', )

# plot the data 
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.2, 1.2
filename = "norm_figure.pdf"

axs.set_xlim(x_min, x_max)
axs.set_ylim(y_min, y_max)
axs.set_aspect('equal')
axs.grid(True, which='both')

# set properties of the legend of the plot
blue_patch = mpatches.Patch(color='blue', label='1/2 Norm')
plt.legend(handles=[blue_patch])
leg = axs.legend(loc='upper right', shadow=True, fancybox=True, numpoints=1)

# either show figure on screen or write it to disk
if filename == None:
    plt.show()
else:
    plt.savefig(filename, facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()