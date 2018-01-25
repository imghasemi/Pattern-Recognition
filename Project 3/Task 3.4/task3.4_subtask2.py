#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
x_file = 'xor-X.csv'
y_file = 'xor-y.csv'
plot_dir = 'plots'


def make_mashgrid(train_d, step=0.01):
    """
    Creates a mashgrid of points
    :param train_d: train data of x1 and x2
    :param step: the step for grid
    :return: the gird of points
    """
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
    generates SVM and plots the prediction with train data
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
    # uncomment to add title with description on the plot
    # fig.suptitle(description)

    x1x1, x2x2 = make_mashgrid(X)
    clf.fit(X, y.ravel())
    # cover the plot by colored grid
    Z = clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
    print 'support vector for', description, ' = ', clf.support_vectors_
    Z = Z.reshape(x1x1.shape)
    axs.contourf(x1x1, x2x2, Z, colors=('y', 'b'), alpha=0.5)

    XY = np.vstack((X.T, y.T)).T
    X_pos = XY[XY[:, 2] > 0][:, 0:2]
    X_neg = XY[XY[:, 2] < 0][:, 0:2]

    # uncomment to plot svm points
    svm_vectors = clf.support_vectors_
    plt.plot(svm_vectors[:, 0], svm_vectors[:, 1], 'go', label='svm', alpha=0.5)

    # plot data
    plt.plot(X_pos[:, 0], X_pos[:, 1], 'bo', label='+1', alpha=0.5)
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'yo', label='-1', alpha=0.5)
    plt.legend(loc="lower right", numpoints=1)

    global plot_dir
    output_file = '.'.join([description.replace(' ', '_').replace(':', '').
                           replace('.', '_').replace(',', '_and'), 'png'])
    output_file = '/'.join([plot_dir, output_file])
    plt.savefig(output_file, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def svm_prediction(X, y):
    """
    defines different types of SVMs and runs a prediction for each one
    :param X: matrix of x1 and x2
    :param y: array of labels
    :return:
    """
    clfs = {'linear: c=1.0': svm.SVC(kernel='linear', C=1.0),
            'linear: c=0.1': svm.SVC(kernel='linear', C=0.1),
            'linear: c=10.0': svm.SVC(kernel='linear', C=10.0),
            'rbf: gamma=0.7, C=0.5': svm.SVC(kernel='rbf', gamma=0.7, C=0.5),
            'rbf: gamma=0.7, C=1.0': svm.SVC(kernel='rbf', gamma=0.7, C=1.0),
            'rbf: gamma=0.7, C=2.0': svm.SVC(kernel='rbf', gamma=0.7, C=2.0),
            'rbf: gamma=0.7, C=8.0': svm.SVC(kernel='rbf', gamma=0.7, C=8.0),
            'rbf: gamma=0.5, C=0.1': svm.SVC(kernel='rbf', gamma=0.5, C=0.1),
            'rbf: gamma=0.5, C=1.0': svm.SVC(kernel='rbf', gamma=0.5, C=1.0),
            'rbf: gamma=0.5, C=2.0': svm.SVC(kernel='rbf', gamma=0.5, C=2.0),
            'rbf: gamma=0.5, C=8.0': svm.SVC(kernel='rbf', gamma=0.5, C=8.0),
            'rbf: gamma=0.5, C=10.0': svm.SVC(kernel='rbf', gamma=0.5, C=10.0),
            'poly: degree=2, C=1.0': svm.SVC(kernel='poly', degree=2, C=1.0),
            'poly: degree=3, C=1.0': svm.SVC(kernel='poly', degree=3, C=1.0),
            'poly: degree=3, C=2.0': svm.SVC(kernel='poly', degree=3, C=2.0),
            'poly: degree=3, C=3.0': svm.SVC(kernel='poly', degree=3, C=3.0),
            'poly: degree=3, C=0.1': svm.SVC(kernel='poly', degree=3, C=0.1),
            'poly: degree=4, C=1.0': svm.SVC(kernel='poly', degree=4, C=1.0),
            'poly: degree=4, C=0.1': svm.SVC(kernel='poly', degree=4, C=0.1),
            'poly: degree=4, C=100.0': svm.SVC(kernel='poly', degree=4, C=100),
            'poly: degree=7, C=1.0': svm.SVC(kernel='poly', degree=7, C=1.0),
            'poly: degree=6, C=1.0': svm.SVC(kernel='poly', degree=6, C=1.0),
            }
    for k, v in clfs.items():
        print('make prediction using svm (%s)' % k)
        predict_and_plot(X, y, clf_tuple=(k, v))


def main():
    # PxN
    X = np.loadtxt(x_file, delimiter=',', dtype=float).T
    # PxM
    y = np.loadtxt(y_file).reshape(X.shape[0], 1)
    svm_prediction(X, y)


if __name__ == '__main__':
    main()