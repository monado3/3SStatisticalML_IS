import matplotlib.pyplot as plt
import numpy as np


# In[]
def make_samples(n=1000):
    x1_arr = np.random.randn(n - 1)
    x2_arr = 4 * np.random.rand(n - 1) - 2
    return np.concatenate([np.array([x1_arr, x2_arr]), np.c_[np.array([6, 2])]], axis=1)


def plot_samples(samples, projection=False, g=None):
    fig, ax = plt.subplots()
    ax.scatter(samples[0], samples[1], marker='x', s=10, label='samples')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid()
    if np.any(projection):
        x = projection[0]
        y = projection[1]
        lim = max(ax.get_ylim()) / y
        ax.set_xlim((-7.5, 7.5))
        ax.text(0.8, 0.22, f'n = {samples.shape[1]}\n\nvector b =\n[{x:.4f},\n{y:.4f}]', transform=ax.transAxes)
        ax.plot([-lim * x, lim * x], [-lim * y, lim * y], 'k-', label='vector b')
        ax.set_title(f'samples and vector b (g(s) = {g})')
        ax.legend(loc='lower right')
    else:
        ax.set_title(f'samples')
    plt.savefig(f'PP_newton_{g}.png')
    plt.show()


def normalize(vec):
    return vec / np.linalg.norm(vec)


def mat_power(mat, pow):
    D_ = np.diag(np.power(l, -0.5))
    l, P = np.linalg.eig(mat)
    return np.matrix(P) * np.matrix(D_) * np.matrix(P).I



def centering_whitening(samples):
    n = samples.shape[1]
    mat_X = np.matrix(samples)
    mat_H = np.matrix(np.identity(n) - (1 / n) * np.ones([n, n]))
    return mat_power((1 / n) * mat_X * (mat_H ** 2) * mat_X.T, -0.5) * mat_X * mat_H


def PP_ap_newtons(mat_X_tilde, g, diff_g, loops=100):
    vec_b = np.array([1, 1])
    vec_b = normalize(vec_b)
    for loop in range(loops):
        projection = np.array(np.dot(vec_b, mat_X_tilde)).ravel()
        g_ = np.array([g(projection).ravel(), g(projection).ravel()])
        vec_b = vec_b * np.mean(diff_g(projection)) - np.mean(np.array(mat_X_tilde) * g_, axis=1)
        vec_b = normalize(vec_b)
    return vec_b


# In[]
np.random.seed(0)
samples = make_samples()
mat_X_tilde = centering_whitening(samples)
vec_b_1 = PP_ap_newtons(mat_X_tilde, lambda s: np.power(s, 3), lambda s: 3 * np.power(s, 2))
plot_samples(samples, vec_b_1, 's^3')
vec_b_2 = PP_ap_newtons(mat_X_tilde, lambda s: np.tanh(s), lambda s: 1 - np.power(np.tanh(s), 2))
plot_samples(samples, vec_b_2, 'tanh(s)')
