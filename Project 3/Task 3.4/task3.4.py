import numpy as np
import matplotlib.pyplot as plt
from math import e
import matplotlib.cm as mpl_cm 


def f(x1, x2, w1, w2, theta): # f: activation function
    return 2*e**(-0.5 * ((w1*x1 + w2*x2 - theta)**2))-1
    
def dE_dw1(x1, x2, y, w1, w2, theta):
    return -2.0*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*x1*(-theta + w1*x1 + w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))

def dE_dw2(x1, x2, y, w1, w2, theta):
    return -2.0*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*x2*(-theta + w1*x1 + w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))
    
def dE_dtheta(x1, x2, y, w1, w2, theta):
    return 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*(-1.0*theta + 1.0*w1*x1 + 1.0*w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))
    
def gradientDescend(x1_list, x2_list, y_list, w1, w2, theta):
    # initialize learning rates
    n_theta = 0.001
    n_w = 0.005
    
    l = len(x1_list)
    s_w1 = s_w2 = s_theta = 0
    
    # compute partial derivative of loss function E
    for i in xrange(l):
        s_w1 += dE_dw1(x1_list[i], x2_list[i], y_list[i], w1, w2, theta)
        s_w2 += dE_dw2(x1_list[i], x2_list[i], y_list[i], w1, w2, theta)
        s_theta += dE_dtheta(x1_list[i], x2_list[i], y_list[i], w1, w2, theta) 
        
    # update parameters theta, w1 and w2
    theta_new = theta - n_theta * s_theta
    w1_new = w1 - n_w * s_w1
    w2_new = w2 - n_w * s_w2
    
    return w1_new, w2_new, theta_new
    
def m_plotClassifierAndData(n_loop, x1_list, x2_list, y_list):
    plt.figure()
    # step 1. plot the Data 
    # step 1.1: move spines to the center, passing through (0,0)
    ax = plt.gca()  
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_position('center')
    
    # step 1.2: plot data
    plt.plot(x1_list[y_list==1], x2_list[y_list==1], 'bo', label="1")
    plt.plot(x1_list[y_list==-1], x2_list[y_list==-1], 'ro', label="-1")
    plt.legend(loc="best", numpoints=1)
    
    # step 2. plot the classifier  
    vmin = -2
    vmax =  2
    cpal = "RdBu"
    cmap_cont = mpl_cm.get_cmap(cpal)
    x = np.arange(-2.0, 2.0, 0.01)
    y = np.arange(-2.0, 2.0, 0.01)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy, w1, w2, theta)
    ax.pcolormesh(x, y, z,
               cmap=cmap_cont,
               vmin=vmin,vmax=vmax )
    plt.title('classifier obtained after ' + str(n_loop) + ' loops')
    plt.show()
    
if __name__ == "__main__":
    # reading data
    data = np.genfromtxt("xor-X.csv", delimiter=',')
    labels = np.genfromtxt("xor-y.csv", delimiter=',')
    x1_list = data[0] 
    x2_list = data[1]
    y_list = labels
  
    # initialize parameters 
    w1 = 1
    w2 = -0.5 
    theta = 0
    print "after 0 loop:"
    print "w = [" + str(w1) + ", " + str(w2) + "]" 
    print "theta = " + str(theta)
    
    m_plotClassifierAndData(0, x1_list, x2_list, y_list)

    n_loop = 50
    
    for i in xrange(n_loop):
        [w1, w2, theta] = gradientDescend(x1_list, x2_list, y_list, w1, w2, theta)
    
    y_new = f(x1_list, x2_list, w1, w2, theta)
    
    print "after " + str(n_loop) + " loops: "
    print "w = [" + str(w1) + ", " + str(w2) + "]" 
    print "theta = " + str(theta)
    
    m_plotClassifierAndData(n_loop, x1_list, x2_list, y_list)
