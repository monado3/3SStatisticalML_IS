import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen


class ApproximateDistro:
    def __init__(self, mu, sigma):
        self.m1 = 10 * np.random.rand()
        self.m2 = 10 * np.random.rand()
        self.mu = mu
        self.sigma = sigma

    def update_m1(self):
        self.m1 = self.mu[0] - (self.sigma[0][1] / self.sigma[0][0]) * (self.m2 - self.mu[1])

    def update_m2(self):
        self.m2 = self.mu[1] - (self.sigma[1][0] / self.sigma[1][1]) * (self.m1 - self.mu[0])

    def pdf(self, x, y):
        q1 = norm(loc=self.m1, scale=np.sqrt(1 / self.sigma[0][0]))
        q2 = norm(loc=self.m2, scale=np.sqrt(1 / self.sigma[1][1]))
        z1 = q1.pdf(x)
        z2 = q2.pdf(y)
        return z1 * np.c_[z2]


def plot_2D_contour(distro, loop=None):
    x = y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    if isinstance(distro, multivariate_normal_frozen):
        pos = np.dstack((X, Y))
        ax.set_title('True distro (2D-gaussian)')
        ax.contourf(X, Y, distro.pdf(pos))
        ax.text(0.1, 0.1, f'mu =\n{distro.mean}\nsigma =\n{distro.cov}',
                bbox=dict(facecolor='white'), transform=ax.transAxes)
    else:
        ax.set_title(f'Approximate distro (q(z1)q(z2))  loops = {loop}')
        ax.contourf(X, Y, distro.pdf(x, y))
        ax.text(0.1, 0.1, f'q(z1) ~ N({distro.m1}, {1/distro.sigma[0][0]})\n'
                          f'q(z2) ~ N({distro.m2}, {1/distro.sigma[1][1]})\n',
                bbox=dict(facecolor='white'), transform=ax.transAxes)

    ax.grid()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')


def main():
    graphdir = op.join(op.dirname(op.abspath('__file__')), 'graphs')

    # settings
    np.random.seed(0)
    mu = np.array([1, 2])
    sigma = np.array([[1, 0.5], [0.5, 2]])
    loops = 100

    # initialize
    appro_distro = ApproximateDistro(mu, sigma)

    # True pdf
    rv = multivariate_normal(mu, sigma)
    plot_2D_contour(rv)
    plt.savefig(op.join(graphdir, 'true_distro.png'))
    plt.show()

    # main algorithm
    for loop in range(loops + 1):
        if loop in (0, 1, 2, loops):
            plot_2D_contour(appro_distro, loop)
            plt.savefig(op.join(graphdir, f'appro_distro_loop{loop:03}.png'))
            plt.show()

        appro_distro.update_m1()
        appro_distro.update_m2()


if __name__ == '__main__':
    main()