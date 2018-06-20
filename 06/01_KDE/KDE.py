import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import norm


# In[]
def make_samples(n=10000):
    x = np.empty(n)
    u = np.random.rand(n)

    flag = np.logical_and(0 <= u, u < 1 / 8)
    x[flag] = np.sqrt(8 * u[flag])
    flag = np.logical_and(1 / 8 <= u, u < 1 / 4)
    x[flag] = 2 - np.sqrt(2 - 8 * u[flag])
    flag = np.logical_and(1 / 4 <= u, u < 1 / 2)
    x[flag] = 1 + 4 * u[flag]
    flag = np.logical_and(1 / 2 <= u, u < 3 / 4)
    x[flag] = 3 + np.sqrt(4 * u[flag] - 2)
    flag = np.logical_and(3 / 4 <= u, u <= 1)
    x[flag] = 5 - np.sqrt(4 - 4 * u[flag])

    return x


def kde(arr, h, min_max=False):
    n = len(arr)
    if not min_max:
        X = np.linspace(np.min(arr), np.max(arr), num=1000).reshape(-1, 1)
    else:
        X = np.linspace(min_max[0], min_max[1], num=1000).reshape(-1, 1)
    inner_sigma = X - arr
    inner_sigma /= h
    inner_sigma = norm.pdf(inner_sigma)
    sigma = np.sum(inner_sigma, axis=1)
    return X.ravel(), sigma / (n * h)


def train_test_split_by_idx(arr, idx, n_split):
    split_arr = np.split(arr, n_split)
    test_arr = split_arr[idx]
    split_arr.pop(idx)
    train_arr = np.concatenate(split_arr)
    return train_arr, test_arr


def lcv(samples_arr, estimator, h, n_split=4):
    LCV = 0
    min_x = np.min(samples_arr)
    max_x = np.max(samples_arr)
    randomized_arr = np.random.permutation(samples_arr)
    for idx in range(n_split):
        train_arr, test_arr = train_test_split_by_idx(randomized_arr, idx, n_split)
        points_by_pdf = estimator(train_arr, h, (min_x, max_x))
        interpolator = interpolate.interp1d(points_by_pdf[0], points_by_pdf[1], kind='quadratic')
        p_Tj_arr = interpolator(np.sort(test_arr))
        inner_sigma = np.log(p_Tj_arr)
        LCV += np.sum(inner_sigma) / len(test_arr)
    return LCV / n_split


def grid_search(samples_arr, estimator, h_tuple=(0.005, 0.05, 0.2, 0.5)):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.hist(samples_arr, bins=30, color='skyblue', density=True, label='normed histogram')
    LCV_lis = []
    points_by_pdf_lis = []
    for h in h_tuple:
        LCV_lis.append(lcv(samples_arr, estimator, h))
        points_by_pdf_lis.append(kde(samples_arr, h))
    best_idx = np.argmax(LCV_lis)
    for idx, points_by_pdf in enumerate(points_by_pdf_lis):
        if idx == best_idx:
            plot_pdf(points_by_pdf, h_tuple[idx], LCV_lis[idx], flag=True)
        else:
            plot_pdf(points_by_pdf, h_tuple[idx], LCV_lis[idx])
    plt.savefig(os.path.join(current_dir, 'pdfs.png'))
    plt.show()
    plt.hist(samples_arr, bins=30, density=True, label='normed histogram')
    plot_pdf(points_by_pdf_lis[best_idx], h_tuple[best_idx], LCV_lis[best_idx],
             title='the estimated pdf which has the best LCV score and the histogram', flag=True)
    plt.savefig(os.path.join(current_dir, 'best_pdf.png'))
    plt.show()
    plot_lcv(h_tuple, LCV_lis)
    plt.savefig(os.path.join(current_dir, 'LCVs.png'))
    plt.show()


def plot_pdf(points_by_pdf, h, LCV, title='the distribution of samples and the estimated pdf', flag=False):
    if flag:
        plt.plot(points_by_pdf[0], points_by_pdf[1], label=f'estimated pdf (h = {h}, LCV = {LCV:.3f}) (best LCV)')
    else:
        plt.plot(points_by_pdf[0], points_by_pdf[1], label=f'estimated pdf (h = {h}, LCV = {LCV:.3f})')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.title(title)
    plt.legend()


def plot_lcv(x, y, n_split=4):
    plt.plot(x, y)
    plt.title(f'LCV(the number of split = {n_split}) by h')
    plt.xlabel('h')
    plt.ylabel('LCV')
    plt.grid()


# In[]
np.random.seed(0)
x = make_samples()
grid_search(x, kde)
