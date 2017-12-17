# k-NN using brute force search
import numpy as np
import time

# kNN function gets the predicted label from findNN function and computes the accuracy of prediction.
# Parameters:
#   _testData: test data
#   _trainingData: training data
#   _k: use k nearest neighors to vote for the label
# Return:
#   the accuracy of our label prediction
def kNN(testData, trainingData, k):
    numOfRightPredict = 0.
    nd = testData[:,0].size
    for i in xrange(nd):
        label = findNN(testData[i], trainingData, k)
        if np.isclose(label, testData[i][2]):
            numOfRightPredict += 1
    return numOfRightPredict/nd

# findNN function computes distance^2 between every training data and the given single test data.
# Then we use linear search to find out k nearest neighbors and vote for the label.
# Parameters: 
#   _testPoint: the given single test data
#   _trainingData: training data
#   _k: use k nearest neighors to vote for the label
# Return:
#   the predicted label of the given single test data
def findNN(testPoint, trainingData, k):
    a = np.array([1., 1.])
    distSquares = np.dot((trainingData[:, 0:2]-testPoint[0:2]) * (trainingData[:, 0:2]-testPoint[0:2]), a)
    
    predictLabels = np.zeros(k)
    for i in range(k):
        j = np.argmin(distSquares)
        predictLabels[i] = trainingData[j][2]
        distSquares[j] = float('inf')
    c1 = sum(predictLabels == -1.)
    c2 = sum(predictLabels == 1.)
    return -1. if c1 > c2 else 1.
    
if __name__ == "__main__":
    trainingData = np.loadtxt('data2-train.dat')
    testData = np.loadtxt('data2-test.dat')

    for k in xrange(1,7,2):
        startTime = time.clock()
        accuracy = kNN(testData, trainingData, k)
        print "k =", k, ", accuracy =", accuracy, ", runtime =", time.clock()-startTime, "s"
