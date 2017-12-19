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
from numpy.linalg import matrix_rank
from numpy.linalg import pinv


def plotline(Data, W):
   
    #get regression line
    line_x = np.linspace(120, 200, num=300)
    X1 = line_x.reshape((line_x.shape[0],1))
    X0 = np.ones(X1.shape)
    X2 = X1 * X1
    X3 = X1 * X2
    X4 = X1 * X3
    X5 = X1 * X4
    X = np.concatenate((X5,X4,X3,X2,X1,X0),axis = 1)
    #X = np.concatenate((X0,X1,X2,X3,X4,X5),axis = 1)
    line_y = (X.dot(W))
    
    #create a fig
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_xlim(120, 200)
    axs.set_ylim(-2 , 200)
    axs.plot(Data[:,1], Data[:,0],'bo',label = ' data')
    axs.plot(line_x,line_y,'r',label = 'fitting line')
    
    # set properties of the legend of the plot
    leg = axs.legend(loc='upper right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)
    
    
    # set x,y axis labels
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    #plt.show()
    plt.savefig("plot_task_2.3.pdf", facecolor='w', edgecolor='w',
        papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
     # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)
    
    #remove outliar, not in this task.
    #outlierInd = np.where( X[:,0] != -1 ) 
    #X = X[outlierInd]
    
    
    #read height data to Y
    Y = X[:,0:1].copy()
    
    #generate design matrix
    X[:,0:1] = np.ones(X[:,0:1].shape)
    X0 = X[:,0:1]
    X1 = X[:,1:2]
    X2 = X[:,1:2] * X[:,1:2]
    X3 = X[:,1:2] * X2
    X4 = X[:,1:2] * X3
    X5 = X[:,1:2] * X4
    X = np.concatenate((X5,X4,X3,X2,X1,X0),axis = 1)

    
    #generate sigma0 and sigma
    sigma0Square = 3.0
    sigmaSquare = 1000.0   # this number should test
    
    #test with LSE
    W_hat_lse, res, rank, s = np.linalg.lstsq(X, Y)
    # the rank of X = 5, then XTX rank is also 5, which means it is not full rank....
    #the inves has some problems.
    #but the least square function from numpy is very good.  how it realize?
    
    #estimate W
    XTX = X.transpose().dot(X)
    #rank_xtx = matrix_rank(XTX)
    #W_hat = inv(XTX)
    #W_hat = pinv((XTX+(sigmaSquare/sigma0Square)*np.identity(XTX.shape[0])))
    W_hat = inv((XTX+(sigmaSquare/sigma0Square)*np.identity(XTX.shape[0])))
    W_hat = W_hat.dot(X.transpose())
    W_hat = W_hat.dot(Y)
    
    #plot
    plotline(data[:,0:2].astype(np.float), W_hat)
    
