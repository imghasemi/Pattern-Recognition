# k-NN using brute force search
"""
@author: Chu-I Chao
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# kNN function gets the predicted label from findNN function and computes the accuracy of prediction.
# Parameters:
#   testData: test data
#   trainingData: training data
#   k: use k nearest neighors to vote for the label
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
#   testPoint: the given single test data
#   trainingData: training data
#   k: use k nearest neighors to vote for the label
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
    
def m_plot(xs, ys):
    plt.plot(xs, ys, 'ro')
    for i in xrange(len(xs)):
        round_y = "%.4f" % ys[i]
        plt.annotate(round_y, xy=(xs[i], ys[i]), xytext=(xs[i], ys[i]-0.003))
    line, = plt.plot(xs, ys, lw=2)
    plt.axis([0, 6, 0.88, 0.94])
    plt.xlabel("k")
    plt.ylabel("accuracy(%)")
    plt.savefig("accuracy.pdf", format="pdf")
    plt.show()    
    return

if __name__ == "__main__":
    trainingData = np.loadtxt('data2-train.dat')
    testData = np.loadtxt('data2-test.dat')
    
    plot_result = True
    
    ks = np.zeros(3)
    accs = np.zeros(3)
    
    for k in xrange(1,7,2):
        startTime = time.clock()
        accuracy = kNN(testData, trainingData, k)
        print "k =", k, ", accuracy =", accuracy, ", runtime =", time.clock()-startTime, "s"
        
        i = (k-1)/2
        ks[i] = k
        accs[i] = accuracy
    
    if (plot_result):
        m_plot(ks, accs)
