import numpy as np
import matplotlib.pyplot as plt
from math import pi

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plots_dir = 'plots'


def pdf(w, h, s_w, s_h, m_w, m_h, ro):
    Q = ((w - m_w) / s_w) ** 2 \
        - 2 * ro * (((w - m_w) / s_w) * ((h - m_h) / s_h)) \
        + ((h - m_h) / s_h) ** 2
    return np.exp(-Q / (2 * (1-ro**2))) / (2 * pi * s_w * s_h * np.sqrt(1-ro**2))


def plot_w_h(w, h, filename='w_h.png', w_=None, h_=None):
    plt.figure()
    plt.plot(h, w, 'ro', alpha=0.4)
    plt.xlabel('height')
    plt.ylabel('weight')
    if w_ is not None and h_ is not None:
        plt.plot(h_, w_, 'bo', alpha=0.4)
    plot_path = '/'.join([plots_dir, filename])
    plt.savefig(plot_path, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


def plot_bi_variate_distribution(w, h, s_w, s_h, m_w, m_h, ro, filename):
    x_span = 0.1*(np.max(w)-np.min(w))
    y_span = 0.1*(np.max(h)-np.min(h))
    x = np.arange(np.min(w)-x_span, np.max(w)+x_span, 0.1)
    y = np.arange(np.min(h)-y_span, np.max(h)+y_span, 0.1)
    x, y = np.meshgrid(x, y)
    z = pdf(x, y, s_w, s_h, m_w, m_h, ro)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
    plt.xlabel('weight')
    plt.ylabel('height')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plot_path = '/'.join([plots_dir, filename])
    plt.savefig(plot_path, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


def main():
    dt = np.dtype([('weight', np.int), ('height', np.int), ('sex', np.str_, 1)])
    d = np.loadtxt(fname='whData.dat', comments='#', dtype=dt)
    # heights with missed weight
    h_ = d[np.where(d['weight'] < 0)]['height']
    # remove outliers
    d = d[np.where(d['weight'] >= 0)]
    w = d['weight']
    h = d['height']

    # mean of weight
    w_mean = np.mean(w)

    # mean of height
    h_mean = np.mean(h)
    print(w_mean, h_mean)

    # deviation of weight
    s_w = np.sqrt(np.sum((w - w_mean) ** 2) / float(len(w)))
    # deviation of height
    s_h = np.sqrt(np.sum((h - h_mean) ** 2) / float(len(h)))
    print('sigma weight: %s, sigma height: %s' % (s_w, s_h))

    # covariance matrix 2x2
    cov = np.cov(np.array([w, h]), bias=True)
    # covariance just for w and h
    # cov = np.sum((w-w_mean)*(h-h_mean))/len(w)

    # covariance between array x and array y
    cov_w_h = cov[0][1]
    # correlation coefficient
    ro = cov_w_h / (s_w * s_h)
    print('ro = %s' % ro)
    plot_w_h(w, h, 'filtered weight and height (no outliers)', 'w_h.png')
    # plot_bi_variate_distribution(w, h, s_w, s_h, w_mean, h_mean, ro,
    #                              'bi_variate_distr.png')
    # predict missed weights
    w_ = w_mean + (ro * (h_-h_mean))/(s_w/s_h)
    print('predicted weights: %s ' % w_)
    plot_w_h(w, h, 'filled_missings.png', w_, h_)


if __name__ == '__main__':
    main()
