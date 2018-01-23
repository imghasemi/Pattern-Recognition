import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def projectionFunction(X, eigenset):
    return np.dot(X, eigenset.T)


# PCA Algorithm
def PCA(X, nBiggestVector):  # Gives the n "largest" eigenvectors
    dataMean = np.mean(X, axis=0)
    zeroMeanData = X - dataMean
    covMat = np.cov(
        np.transpose(zeroMeanData))  # Transpose as X is defined as each row being one data point. Not column.
    eigenVals, eigenVecs = np.linalg.eigh(covMat)

    # argpartition(eigenVals,-n)[-n:] gives the position values of n largest eigenVals.
    return eigenVecs.T[np.argpartition(eigenVals, -nBiggestVector)[-nBiggestVector:]]


# PCA Algorithm
def LDA(X, clsLength, n):
    # Overal Mean
    overalMean = np.mean(X, axis=0)

    classMeans = np.array([np.mean(X[0:50], axis=0), np.mean(X[50:100], axis=0), np.mean(X[100:150], axis=0)])
    # classMeans contains the means of each class. len(clsMean)=len(clsLength)=#classes

    meanList = np.concatenate([[classMeans[i]] * clsLength[i] for i in range(len(clsLength))])
    # meanList contains elements which are equal to the number of data points. Here, len(meanList)=150

    S_W = np.zeros((len(X[0]), len(X[0])))
    for i in range(len(X)):
        S_W = S_W + np.outer((X - meanList)[i], (X - meanList)[i])

    S_B = np.zeros((len(X[0]), len(X[0])))
    for i in range(len(classMeans)):
        S_B = S_B + clsLength[i] * np.outer((classMeans[i] - overalMean), (classMeans[i] - overalMean))

    eigenVals, eigenVecs = np.linalg.eigh(np.dot(np.linalg.inv(S_W), S_B))

    return eigenVecs.T[np.argpartition(eigenVals, -n)[-n:]]


if __name__ == "__main__":

    X = np.transpose(np.genfromtxt("data-dimred-X.csv", delimiter=','))
    Y_Label = np.genfromtxt("data-dimred-y.csv")

    classLengthArray = np.array(
        [len(Y_Label[Y_Label == 1]), len(Y_Label[Y_Label == 2]), len(Y_Label[Y_Label == 3])])

    finalPCA = projectionFunction(X, PCA(X, 2))
    finalLDA = projectionFunction(X, LDA(X, classLengthArray, 2))
    # These are the projected 2D points for LDA and PCA

    # 2D LDA
    for m, n in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(finalLDA.T[0][np.where(Y_Label == m)], finalLDA.T[1][np.where(Y_Label == m)], color=n)
        plt.title('LDA Algorithm 2D')
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.savefig('Figure_LDA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # 2D PCA
    for m, n in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(finalPCA.T[0][np.where(Y_Label == m)], finalPCA.T[1][np.where(Y_Label == m)], color=n)
        plt.title('PCA Algorithm 2D')
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.savefig('Figure_PCA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # # 2D - For Comparison
    # plt.scatter(finalLDA.T[0], finalLDA.T[1], color='red')
    # plt.scatter(finalPCA.T[0], finalPCA.T[1])
    # plt.legend(('LDA', 'PCA'), loc=3)
    # plt.show()

    # -------------------------------------------------------------------------------------------------------------
    # 3D
    finalPCA_3D = projectionFunction(X, PCA(X, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for m, n in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D((finalPCA_3D.T)[0][np.where(Y_Label == m)], (finalPCA_3D.T)[1][np.where(Y_Label == m)],
                      (finalPCA_3D.T)[2][np.where(Y_Label == m)], color=n)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('PCA Algorithm 3D')
    plt.savefig('Figure_PCA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    finalLDA_3D = projectionFunction(X, LDA(X, classLengthArray, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for m, n in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D((finalLDA_3D.T)[0][np.where(Y_Label == m)], (finalLDA_3D.T)[1][np.where(Y_Label == m)],
                      (finalLDA_3D.T)[2][np.where(Y_Label == m)], color=n)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('LDA Algorithm 3D')
    plt.savefig('Figure_LDA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
