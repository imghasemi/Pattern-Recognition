#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
x_file = 'xor-X.csv'
y_file = 'xor-y.csv'
# x_file = 'trainx.csv'
# y_file = 'trainy.csv'
plot_dir = 'plots'


def transf_fn(x, derivative=False, nonlinear=False):
    if nonlinear:
        return -2*x*np.exp(-x**2/2) if derivative else 2*np.exp(-x**2/2)-1
    else:
        return np.exp(-x)/(1+np.exp(-x))**2 if derivative else 1/(1+np.exp(-x))
    # return x*(1-x) if derivative else 1/(1+np.exp(-x))


def make_mashgrid(train_d, step=0.01):
    x1_min = train_d[:, 0].min()
    x1_max = train_d[:, 0].max()
    x2_min = train_d[:, 1].min()
    x2_max = train_d[:, 1].max()
    return np.meshgrid(np.arange(x1_min, x1_max, step),
                       np.arange(x2_min, x2_max, step))


# def plot_data(X, y, X_ev, y_ev, x1x1_, x2x2_, filename='file1.png'):
#     XY = np.hstack((X, y))
#     XY_ev = np.hstack((X_ev, y_ev))
#     x1_min = XY[:, 0].min()
#     x1_max = XY[:, 0].max()
#     x2_min = XY[:, 1].min()
#     x2_max = XY[:, 1].max()
#
#     def distinguish(XY):
#         return XY[XY[:, 2] > 0][:, 0:2], XY[XY[:, 2] < 0][:, 0:2]
#
#     # matrix of x1 and x2 where y is positive, and negative
#     X_pos, X_neg = distinguish(XY)
#     X_ev_pos, X_ev_neg = distinguish(XY_ev)
#
#     fig = plt.figure()
#     axs = plt.subplot(111)
#     fig.suptitle('init data')
#     x1_span = (x1_max-x1_min)*0.4
#     x2_span = (x2_max-x2_min)*0.4
#     plt.xlim(x1_min-x1_span, x1_max+x1_span)
#     plt.ylim(x2_min-x2_span, x2_max+x2_span)
#     plt.plot(X_pos [:, 0], X_pos [:, 1], 'bo', label='+1 (train data)')
#     plt.plot(X_neg[:, 0], X_neg[:, 1], 'yo', label='-1 (train data)')
#     # plt.plot(X_ev_pos [:, 0], X_ev_pos [:, 1], 'go', label='+1 (test data)')
#     # plt.plot(X_ev_neg[:, 0], X_ev_neg[:, 1], 'mo', label='-1 (test data)')
#     Z = XY_ev[:, -1].reshape(x1x1_.shape)
#     # axs.contourf(x1x1_, x2x2_, Z, cmap=plt.cm.paired, alpha=1)
#     axs.contourf(x1x1_, x2x2_, Z, colors=('k',), alpha=1)
#     leg = axs.legend(loc='lower right', shadow=True, fancybox=True, numpoints=1)
#     leg.get_frame().set_alpha(0.5)
#
#     # xy_pos = np.where(xy[:, 3] >= 0)
#     filename = '/'.join([plot_dir, filename])
#     plt.savefig(filename, facecolor='w', edgecolor='w',
#                 papertype=None, format='png', transparent=False,
#                 bbox_inches='tight', pad_inches=0.1)
#     plt.close()


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
    plt.plot(X_pos[:, 0], X_pos[:, 1], 'bo', label='+1 (train data)', alpha=0.6)
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'yo', label='-1 (train data)', alpha=0.6)

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


def evaluate(W, x):
    x1x1, x2x2 = make_mashgrid(x)
    xx = np.stack((np.ones(len(x1x1.ravel())), x1x1.ravel(), x2x2.ravel())).T
    y = np.dot(xx, W)
    return np.hstack((xx[:, 1:], y)), x1x1, x2x2


def main():
    eta = 0.001
    # PxN
    x = np.loadtxt(x_file, delimiter=',', dtype=float).T
    # Px(N+1) adding bias
    print np.ones((x.shape[0], 1)).shape
    print x.shape
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    # PxM
    y = np.loadtxt(y_file).reshape(x.shape[0], 1)
    print 'x.shape = ', x.shape, 'y.shape = ', y.shape
    print x, y
    # (N+1)xM
    w = np.random.uniform(-0.5, 0.5, (x.shape[1], y.shape[1]))
    n = 1
    for i in xrange(n):
        # PxM = Px(N+1) x (N+1)xM
        out = transf_fn(np.dot(x, w), nonlinear=True)
        # PxM
        err = y - out
        # PxM
        delta = err*transf_fn(out, derivative=True, nonlinear=True)
        # (N+1)xM = (N+1)xP x PxM
        w += eta*np.dot(x.T, delta)
        print(err)
    print(w)
    svm_prediction(x[:, 1:], y)
    # xy, x1x1, x2x2 = evaluate(w, x[:, 1:])
    # plot_data(x[:, 1:], y, xy[:, :2], xy[:, 2:], x1x1, x2x2)


if __name__ == '__main__':
    main()