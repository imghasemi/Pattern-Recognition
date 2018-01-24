#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: MohammadaliGhasemi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# --------------------------------------------------------------------------
# Projection function
# --------------------------------------------------------------------------
def projectionFunction(X, eigenset):
    return np.dot(X, eigenset.T)


# --------------------------------------------------------------------------
# PCA Algorithm
# --------------------------------------------------------------------------
# Compute the PCA algorithm with two args: Data, integer length of eigenVector
# (Biggest ones)
def PCA(dataX, nBiggestVector):
    # calculate the overall mean of data
    overalDataMean = np.mean(dataX, axis=0)
    # normalize to zero mean
    zeroMeanData = dataX - overalDataMean
    # transpose data
    covarianceMatrix = np.cov(
        np.transpose(zeroMeanData))
    # derive eig values and vectors
    eigenValues, eigenVectors = np.linalg.eigh(covarianceMatrix)
    # return n-biggest eigen-vectors
    return eigenVectors.T[np.argpartition(eigenValues, -nBiggestVector)
                          [-nBiggestVector:]]


# ---------------------------------------------------------------------------
# LDA Algorithm
# ---------------------------------------------------------------------------
# Compute the Linear Discriminant Analysis (LDA) for reducing the dimension
# from 500 to 2s
def LDA(X, classLength, nBiggestVector):
    # calculating the overal mean of data
    overalDataMean = np.mean(X, axis=0)
    # classMeans contains the means related for each class
    classMeans = np.array([np.mean(X[0:50], axis=0), np.mean(X[50:100], 
                           axis=0), np.mean(X[100:150], axis=0)])
    # meanList: we use this array (150 elements) for sake of simplicity during
    # the calculation
    meanList = np.concatenate([[classMeans[i]] * classLength[i] 
        for i in range(len(classLength))])
    # here we calculate S_B which is the within class scatter matrix
    S_W = np.zeros((len(X[0]), len(X[0])))
    for i in range(len(X)):
        S_W = S_W + np.outer((X - meanList)[i], (X - meanList)[i])
    # Between class scatter matrix
    S_B = np.zeros((len(X[0]), len(X[0])))
    for i in range(len(classMeans)):
        S_B = S_B + classLength[i] * np.outer((classMeans[i] - overalDataMean),
                               (classMeans[i] - overalDataMean))
    # deriving the eig vals and vectors
    eigenValues, eigenVectors = np.linalg.eigh(np.dot(np.linalg.pinv(S_W), S_B))
    # return n-biggest eigen-vectors
    return eigenVectors.T[np.argpartition(eigenValues, -nBiggestVector)
                          [-nBiggestVector:]]


# ---------------------------------------------------------------------------
# Main application:
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    dataX = np.transpose(np.genfromtxt("data-dimred-X.csv", delimiter=','))
    Y_Label = np.genfromtxt("data-dimred-y.csv")

    classLengthArray = np.array(
        [len(Y_Label[Y_Label == 1]), len(Y_Label[Y_Label == 2]), 
         len(Y_Label[Y_Label == 3])])

    finalPCA = projectionFunction(dataX, PCA(dataX, 2))
    finalLDA = projectionFunction(dataX, LDA(dataX, classLengthArray, 2))
    # -------------------------------------------------------------------------------------------------------------
    # 2D visualisation
    # -------------------------------------------------------------------------------------------------------------

    # 2D LDA
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(finalLDA.T[0][np.where(Y_Label == i)],
                    finalLDA.T[1][np.where(Y_Label == i)], color=j)
        plt.title('LDA Algorithm 2D')
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.savefig('Figure_LDA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # 2D PCA
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(finalPCA.T[0][np.where(Y_Label == i)], 
                    finalPCA.T[1][np.where(Y_Label == i)], color=j)
        plt.title('PCA Algorithm 2D')
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.savefig('Figure_PCA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # --------------------------------------------------------------------
    # 3D visualisation
    # --------------------------------------------------------------------
    finalPCA_3D = projectionFunction(dataX, PCA(dataX, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D((finalPCA_3D.T)[0][np.where(Y_Label == i)], 
                      (finalPCA_3D.T)[1][np.where(Y_Label == i)],
                      (finalPCA_3D.T)[2][np.where(Y_Label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('PCA Algorithm 3D')
    plt.savefig('Figure_PCA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    finalLDA_3D = projectionFunction(dataX, LDA(dataX, classLengthArray, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D((finalLDA_3D.T)[0][np.where(Y_Label == i)], 
                      (finalLDA_3D.T)[1][np.where(Y_Label == i)],
                      (finalLDA_3D.T)[2][np.where(Y_Label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('LDA Algorithm 3D')
    plt.savefig('Figure_LDA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
