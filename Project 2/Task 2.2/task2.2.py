import numpy as np
import matplotlib.pyplot as plt
from math import pi

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def pdf(w, h, s_w, s_h, m_w, m_h, ro):
    Q = ((w - m_w) / s_w) ** 2 \
        - 2 * ro * (((w - m_w) / s_w) * ((h - m_h) / s_h)) \
        + ((h - m_h) / s_h) ** 2
    return np.exp(-Q / (2 * (1 - ro ** 2))) / (
    2 * pi * s_w * s_h * np.sqrt(1 - ro ** 2))


def plot_distribution(w, h, s_w, s_h, m_w, m_h, ro):
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
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def main():
    dt = np.dtype([('weight', np.int), ('height', np.int), ('sex', np.str_, 1)])
    d = np.loadtxt(fname='whData.dat', comments='#', dtype=dt)
    # remove outlayers
    d = d[np.where(d['weight'] >= 0)]
    w = np.array(d['weight'])
    h = np.array(d['height'])

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
    cov = np.cov(np.array([w, h]))
    # covariance between array x and array y
    cov_w_h = cov[0][1]
    # correlation coefficient
    ro = cov_w_h / (s_w * s_h)
    plot_distribution(w, h, s_w, s_h, w_mean, h_mean, ro)


if __name__ == '__main__':
    main()
