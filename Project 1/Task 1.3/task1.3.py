from __future__ import division

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import sys


def estimateOfWeibullDistribution(k, a, D):  # k: kappa, a: alpha, D: 1-D data sample
    N = len(ds)
    s_log_di = 0
    s_di_a_k_log_di_a = 0
    s_di_a_k = 0
    s_di_a_k_log_di_a_2 = 0

    for i in range(N):
        s_log_di += math.log(D[i])
        s_di_a_k_log_di_a += pow(D[i] / a, k) * math.log(D[i] / a)
        s_di_a_k += pow(D[i] / a, k)
        s_di_a_k_log_di_a_2 += pow(D[i] / a, k) * pow(math.log(D[i] / a), 2)

    p_k = N / k - N * math.log(a) + s_log_di - s_di_a_k_log_di_a
    p_a = k / a * (s_di_a_k - N)
    p_kk = -N / pow(k, 2) - s_di_a_k_log_di_a_2
    p_aa = k / pow(a, 2) * (N - (k + 1) * s_di_a_k)
    p_ka = 1 / a * (s_di_a_k) + k / a * (s_di_a_k_log_di_a) - N / a

    mat = np.matrix([[p_kk, p_ka], [p_ka, p_aa]])
    inv_mat = np.linalg.inv(mat)

    new_ka = np.matrix([[k], [a]]) + np.dot(inv_mat, np.matrix([[-p_k], [-p_a]]))
    return new_ka


def testEstimation(p):  # check the sum of probability distribution p is 1 or not
    sum_prob = 0
    for x in p:
        sum_prob += x
    if abs(sum_prob - 1) < 0.01:
        return True
    else:
        print "Estimation test failed"
        return False


if __name__ == "__main__":
    # read data from a .csv file
    hist = []
    with open('myspace.csv', 'rb') as csvfile:
        csv_reader = csv.reader(csvfile)  # Note: csv.reader read data line by line

        # for each row, store non-zero data in the 2nd column into the histogram array
        for row in csv_reader:
            n = (int)(row[1])
            if n > 0:
                hist.append(n)

    # compute data sample from the histogram
    X = len(hist)
    ds = []
    for x in range(X):
        tmp_array = [x + 1 for i in range(hist[x])]
        ds.extend(tmp_array)

    k = 1.
    a = 1.
    for i in range(20):
        ka = estimateOfWeibullDistribution(k, a, ds)
        k = np.matrix.item(ka, 0)
        a = np.matrix.item(ka, 1)

    # compute probability function using k(kappa) and a(alpha)
    prob = [0 for x in range(X)]
    for x in range(X):
        prob[x] = k / a * pow(x / a, k - 1) * math.exp(-pow(x / a, k))

    if testEstimation(prob) == False:
        exit()

    # find the scale which makes the area between histogram and a correspondingly scaled version
    # of the fitted distribution smallest
    hist_max = max(hist)
    weibull_max = max(prob)
    min_sum_value = sys.maxsize
    min_scale = -1
    go_less = False;
    for i in xrange(int(hist_max / weibull_max), 0, -1):
        sum_value = 0
        for j in range(X):
            sum_value += abs(hist[j] - i * prob[j])
        if (min_sum_value > sum_value):
            min_scale = i
            min_sum_value = sum_value
            go_less = True
        if (go_less == True and min_sum_value < sum_value):
            break

    plt.plot(hist)
    plt.plot(min_scale * np.array(prob))
    plt.show()