import numpy as np
import matplotlib.pyplot as plt
from math import e
    
def f(x1, x2, w1, w2, theta):
    return 2*e**(-0.5 * ((w1*x1 + w2*x2 - theta)**2))-1
    
def dE_dw1(x1, x2, y, w1, w2, theta):
    return -2.0*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*x1*(-theta + w1*x1 + w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))

def dE_dw2(x1, x2, y, w1, w2, theta):
    return -2.0*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*x2*(-theta + w1*x1 + w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))
    
def dE_dtheta(x1, x2, y, w1, w2, theta):
    return 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2)*(-1.0*theta + 1.0*w1*x1 + 1.0*w2*x2)*(-y - 1 + 2*e**(-0.5*(-theta + w1*x1 + w2*x2)**2))
    
def gradientDescend(x1_list, x2_list, y_list, w1, w2, theta):
    n_theta = 0.001
    n_w = 0.005
    
    l = len(x1_list)
    s_w1 = s_w2 = s_theta = 0
    
    for i in xrange(l):
        s_w1 += dE_dw1(x1_list[i], x2_list[i], y_list[i], w1, w2, theta)
        s_w2 += dE_dw2(x1_list[i], x2_list[i], y_list[i], w1, w2, theta)
        s_theta += dE_dtheta(x1_list[i], x2_list[i], y_list[i], w1, w2, theta) 
        
    theta_new = theta - n_theta * s_theta
    w1_new = w1 - n_w * s_w1
    w2_new = w2 - n_w * s_w2
    
    return w1_new, w2_new, theta_new
    
def m_plot(x1_list, x2_list, y_list):
    # plotting: moving spines to the center, passing through (0,0)
    ax = plt.gca()  
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_position('center')
    
    # plotting
    plt.plot(x1_list[y_list==1], x2_list[y_list==1], 'ro', label="1")
    plt.plot(x1_list[y_list==-1], x2_list[y_list==-1], 'co', label="-1")
    plt.legend(loc="best", numpoints=1)
    
    plt.show()
    
if __name__ == "__main__":
    # reading data
    data = np.genfromtxt("xor-X.csv", delimiter=',')
    labels = np.genfromtxt("xor-y.csv", delimiter=',')
    x1_list = data[0] 
    x2_list = data[1]
    y_list = labels
    
    m_plot(x1_list, x2_list, y_list)
    
    w1 = w2 = theta = 1
    n_loop = 50
    
    for i in xrange(n_loop):
        [w1, w2, theta] = gradientDescend(x1_list, x2_list, y_list, w1, w2, theta)
    
    y_new = []
    for i in xrange(200):
        y_new.append(f(x1_list[i], x2_list[i], w1, w2, theta))

    plt.figure()
    plt.axis([0, 200, -1.1, 1.1])
    plt.plot(y_new, 'r')
    plt.plot(y_list, 'b')
    plt.show()
