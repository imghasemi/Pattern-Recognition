#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:44:07 2017

Bayesian regression for missing value prediction

@author: qn

The regression model:
        Y = w0 + w1*x + w2*x^2 + w3*x^3 + w4*x^4 + w5*x^5 + e
        Y = X*W + e
        e is noise, x = height, y = weight
        
        X = [x^5 x^4 x^3 x^2 x 1] Y=[y] w=[[w5],[w4],[w3],[w2],[w1],[w0]]
We want to find the best W from given Data, which can be describled by max posteriori:
        W_hat = argmaxP(D|W)
We assume a gaussian prior:
        p(W) ~ N(W|U0,sigma0^2*I)
        U0 = 0
        sigma0^2 = 3 
For simplicity, we assume the likelihood funciton is also a gaussian distrubiton:
        P(D|W) ~ N(y|y - WT*x , sigma^2)
        here, y and x are from data
Conjuagte! Then we know the posteriori distrubition is also gaussian:
        P(W|D) ~ N(W|u,lambda^(-1))
        u = (1/sigma^2)*lambda^-1*XT*y
        lambda = (1/sigma^2)XT*X + (1/sigma0^2)*I
For guassian distribution, we directly know that the max probability point is its expectation point, so:
        W_hat = u = (XT * X + (sigma^2/sigma0^2) * I)^-1 * XT * y

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from numpy.linalg import pinv
#from numpy.linalg import cond
#from scipy.sparse.linalg import lsqr

if __name__ == "__main__":
     # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X_ori = data[:,0:2].astype(np.float)
    
#    # remove data we want estimate, but should we remove these outliar???
    mask = (X_ori[:,0] == -1)
#    X_est, X = X_ori[mask][:,1], X_ori[~mask][:,1]
#    Y = X_ori[~mask][:,0]
    X_est = X_ori[mask][:,1]
    Y = X_ori[:,0]
    
    #generate design matrix
    X = X_ori[:,1]
    X0 = np.ones(X.shape)
    X1 = X
    X2 = X * X
    X3 = X * X2
    X4 = X * X3
    X5 = X * X4
    X = np.vstack((X5,X4,X3,X2,X1,X0)).transpose()

    
    #generate sigma0 and sigma
    sigma0Square = 3.0
    sigmaSquare = 1000.0   # this number can altnatively change
    
    #estimate W with Least Square Method
    #note: regradless of XTX is invertible or not, we can always use its pesudo inverse
    #to calculate this least square method
    #then it becomes
    # W = pinv(X)*Y
    #W_hat_lse = pinv(X).dot(Y)
    #W_hat_lse = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    #however, this lstsq method is best, it is robust and fast
    W_hat_lse = np.linalg.lstsq(X,Y)[0]
    
    #estimate W with Baysian regression
    #for this formula, we don't have numrical instability problems, so just using the inv method
    n,m = X.shape
    I = np.identity(m)
    W_hat = np.dot(np.dot(inv(np.dot(X.T,X) + (sigmaSquare/sigma0Square)*I),X.T),Y)

    
    #plot
    #set up and low bound
    min_x, max_x = min(X[:,4]), max(X[:,4])
    min_y, max_y = min(Y), max(Y)
    ax_x_min, ax_x_max = min_x-(max_x-min_x)/3, max_x+(max_x-min_x)/3
    ax_y_min, ax_y_max = min_y-(max_y-min_y)/3, max_y+(max_y-min_y)/3
    
    #create a fig
    fig = plt.figure()
    axs = fig.add_subplot(111)   
    axs.set_xlim(ax_x_min, ax_x_max)
    axs.set_ylim(ax_y_min, ax_y_max)
    
    plt.xlabel('Height\nSigma^2 = '+str(sigmaSquare))
    plt.ylabel('Weight')
    
    #get regression lines and points
    line_x = np.linspace(ax_x_min, ax_x_max, 1000)
    axs.plot(line_x, np.polyval(W_hat, line_x), label='Bayesian',color='green')
    axs.plot(line_x, np.polyval(W_hat_lse, line_x), label='LSE',color='blue')
    axs.scatter(X[:,4], Y, s=10, color='orange')
    axs.scatter(X_est, np.polyval(W_hat, X_est), label="est from Bayesian", s=20, color='red')
    axs.scatter(X_est, np.polyval(W_hat_lse, X_est), label = "est from LSE", s=20, color='pink')
    
    # set properties of the legend of the plot
    leg = axs.legend(loc='lower right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)
    
    plt.savefig('prediction_sigma=%d.pdf'%(sigmaSquare), facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
   

    