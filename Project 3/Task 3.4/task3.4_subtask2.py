#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
x_file = 'xor-X.csv'
y_file = 'xor-y.csv'
plot_dir = 'plots'


def make_mashgrid(train_d, step=0.01):
    x1_min = train_d[:, 0].min()
    x1_max = train_d[:, 0].max()
    x2_min = train_d[:, 1].min()
    x2_max = train_d[:, 1].max()
    x1_span = (x1_max-x1_min)*0.3
    x2_span = (x2_max-x2_min)*0.3

    return np.meshgrid(np.arange(x1_min-x1_span, x1_max+x1_span, step),
                       np.arange(x2_min-x2_span, x2_max+x2_span, step))


def predict_and_plot(X, y, clf_tuple):
    """
    Fits the svm to train data, makes a prediction for the
    test data. Plots the results in img folder.

    :param train_d: array with train data
    :param test_d:  array with test data
    :param clf_tuple: clf object with description string
    :return: none
    """
    description, clf = clf_tuple
    fig = plt.figure()
    axs = fig.add_subplot(111)
    fig.suptitle(description)

    x1x1, x2x2 = make_mashgrid(X)
    clf.fit(X, y)
    # classify test data
    y_ev = clf.predict(X)
    # set new labels for the test data after classification
    XY = np.vstack((X.T, y_ev.T)).T
    # cover the plot by colored grid
    Z = clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
    print 'this is support vector = ', clf.support_vectors_, ' for ', description
    Z = Z.reshape(x1x1.shape)
    # axs.contourf(x1x1, x2x2, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    axs.contourf(x1x1, x2x2, Z, colors=('y', 'b'), alpha=0.5)

    X_pos = XY[XY[:, 2] > 0][:, 0:2]
    X_neg = XY[XY[:, 2] < 0][:, 0:2]

    # plot data
    plt.plot(X_pos[:, 0], X_pos[:, 1], 'bo', label='+1 (train data)', alpha=0.5)
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'yo', label='-1 (train data)', alpha=0.5)

    global plot_dir
    output_file = '.'.join([description.replace(' ', '_').
                            replace(':', ''), 'png'])
    output_file = '/'.join([plot_dir, output_file])
    plt.savefig(output_file, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def svm_prediction(X, y):
    clfs = {'linear: c=1.0': svm.SVC(kernel='linear', C=1.0),
            # 'rbf: gamma=0.7, C=5.0': svm.SVC(kernel='rbf', gamma=0.7, C=0.5),
            # 'rbf: gamma=0.7, C=1.0': svm.SVC(kernel='rbf', gamma=0.7, C=1.0),
            # 'rbf: gamma=0.7, C=2.0': svm.SVC(kernel='rbf', gamma=0.7, C=2.0),
            # 'rbf: gamma=0.7, C=8.0': svm.SVC(kernel='rbf', gamma=0.7, C=4.0),
            # 'rbf: gamma=0.5, C=5.0': svm.SVC(kernel='rbf', gamma=0.5, C=0.5),
            # 'rbf: gamma=0.5, C=1.0': svm.SVC(kernel='rbf', gamma=0.5, C=1.0),
            # 'rbf: gamma=0.5, C=2.0': svm.SVC(kernel='rbf', gamma=0.5, C=2.0),
            # 'rbf: gamma=0.5, C=8.0': svm.SVC(kernel='rbf', gamma=0.5, C=4.0),
            # 'poly: degree=3, C=1.0': svm.SVC(kernel='poly', degree=3, C=1.0),
            'poly: degree=4, C=1.0': svm.SVC(kernel='poly', degree=4, C=1.0),
            }
    for k, v in clfs.items():
        print('make prediction using svm (%s)' % k)
        predict_and_plot(X, y, clf_tuple=(k, v))


def main():
    # PxN
    x = np.loadtxt(x_file, delimiter=',', dtype=float).T
    # PxM
    y = np.loadtxt(y_file).reshape(x.shape[0], 1)
    svm_prediction(x, y)


if __name__ == '__main__':
    main()