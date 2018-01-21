
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#PCA
"""
Created on Sat Jan 20 19:23:43 2018
task 3.3: Using PCA to do dimensionality reduction
@author: qn

data shape(500,150) like this:
X1 X2 X3 ... X150
.  .  .  ... .


1. read data and do normlize? why we need normlize?
2. transfer data set to 0-mean by reducing the bias( bias = 1/n * sum(data))
3. get the covariance matrix
4. calculate eigen_vector and eigen_value(complex number)
5. choose two maxium eigen_value's corresponding eigenvector.
6. using this two eigenvector construct projection matrix
7. do projection
     
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt






if __name__ == "__main__":
    # reading data
    data = np.genfromtxt("data-dimred-X.csv", delimiter=',')
    labels = np.genfromtxt("data-dimred-y.csv", delimiter=',')
    
    # normlize and 0 mean data
    data_norm = np.divide(data, LA.norm(data, axis = 0))
    bias = 1* np.sum(data_norm,axis = 1, dtype = 'float')/data_norm.shape[1]
    data_zero_mean = (np.subtract(data_norm.T, bias)).T
    
    # compute data covariance matrix
    C = data_zero_mean.dot(data_zero_mean.T)*1/data_zero_mean.shape[1]
    
    # compute the eigen_value and eigen_vector
    eigen_value, eigen_vector = LA.eig(C)
    
    # complex number, determine wether all imag part is small enough then we can remove them.
    if (np.all(eigen_value.imag < 1e-09)):
        eigen_value = eigen_value.real
        
    # take the maxium 2 eigenvector
    maxEigenvalueindice = np.argsort(eigen_value)[498:500]
    maxEigenvector = eigen_vector[:,maxEigenvalueindice]
    if (np.all(maxEigenvector.imag < 1e-09)):
        maxEigenvector = maxEigenvector.real
    
    # reduce data dimension, but should we do this at the raw data ? or at the zero_mean data?
    new_data = maxEigenvector.T.dot(data)
    
    # plot data
    class1 = labels == 1
    class2 = labels == 2
    class3 = labels == 3
    plt.plot(new_data[0,class1], new_data[1,class1] ,'ro')
    plt.plot(new_data[0,class2], new_data[1,class2] ,'go')
    plt.plot(new_data[0,class3], new_data[1,class3] ,'bo')
    plt.show()
    
    # calculate error and ratio of spectrum
    sumPricinplevalue = np.sum(eigen_value[maxEigenvalueindice])
    sumOthervalue = np.sum(eigen_value) - sumPricinplevalue
    error = sumOthervalue
    spectrum = sumPricinplevalue/sumOthervalue
