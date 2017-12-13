import numpy as np
import matplotlib.pyplot as plt
from math import pi


def pdf(w, h, s_w, s_h, m_w, m_h, ro):
    Q = ((w - m_w) / s_w) ** 2 \
        - 2 * ro * (((w - m_w) / s_w) * ((h - m_h) / s_h)) \
        + ((h - m_h) / s_h) ** 2
    return np.exp(-Q / (2 * (1 - ro ** 2))) / (
    2 * pi * s_w * s_h * np.sqrt(1 - ro ** 2))


def main():
    dt = np.dtype([('weight', np.int), ('height', np.int), ('sex', np.str_, 1)])
    d = np.loadtxt(fname='whData.dat', comments='#', dtype=dt)
    # remove outlayers
    d = d[np.where(d['weight'] >= 0)]
    # TODO: remove this
    for i in d:
        print(i)
    plt.figure(1)
    plt.plot(np.arange(6), np.ones(6))
    plt.show()
    # mean of weight
    mw = np.mean(d['weight'])

    # mean of height
    mh = np.mean(d['height'])
    print(mw, mh)

    weight = np.array(d['weight'])
    height = np.array(d['height'])

    # deviation of weight
    sw = np.sqrt(np.sum((weight - mw) ** 2) / float(len(weight)))
    # deviation of height
    sh = np.sqrt(np.sum((height - mh) ** 2) / float(len(height)))
    print('sigma weight: %s, sigma height: %s' % (sw, sh))

    w_h = np.array([weight, height])
    cov = np.cov(w_h)
    # print(cov[0][0]**2+cov[0][1]**2, np.sqrt(cov[0][0]**2+cov[0][1]**2))
    # print(cov[1][0]**2+cov[1][1]**2, np.sqrt(cov[1][0]**2+cov[1][1]**2))
    # print(sw**2)
    # print(sh**2)

    print(cov)
    ro = (cov[0][0] * cov[1][0] + cov[0][1] * cov[1][1]) / (sw * sh)


if __name__ == '__main__':
    main()