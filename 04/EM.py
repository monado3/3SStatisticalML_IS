import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as gauss

# In[]
def make_samples(n=5000):
    return np.random.randn(n) + (np.random.rand(n) > 0.3)*4 - 2

def gmm_pdf(x_arr, w_arr, mu_arr, sigma_arr):
    y_arr = np.zeros(len(x_arr))
    for j,w in enumerate(w_arr):
        y_arr += w*gauss.pdf(x_arr, loc=mu_arr[j], scale=sigma_arr[j])
    return y_arr

def step_E(x_arr, w_arr, mu_arr, sigma_arr):
    m = len(w_arr)
    sum_arr = np.array([w_arr[j]*gauss.pdf(x_arr, loc=mu_arr[j], scale=sigma_arr[j]) for j in range(m)])
    return sum_arr / np.sum(sum_arr, axis=0)

def calculate_w(eta_arr):
    return np.sum(eta_arr, axis=1) / eta_arr.shape[1]

def calculate_mu(eta_arr, x_arr):
    return np.sum(eta_arr*x_arr, axis=1) / np.sum(eta_arr, axis=1)

def calculate_sigma(eta_arr, x_arr, mu_arr):
    return np.sqrt(np.sum(eta_arr*np.power(x_arr - np.c_[mu_arr], 2), axis=1) / np.sum(eta_arr, axis=1))

def step_M(eta_arr, x_arr, mu_arr):
    return calculate_w(eta_arr), calculate_mu(eta_arr, x_arr), calculate_sigma(eta_arr, x_arr, mu_arr)

def EM_algorithm(x_arr, m=4):
    w_arr = np.ones(m)
    mu_arr = np.linspace(-3,3,m)
    sigma_arr = np.linspace(1,3,m)
    for step in range(500):
        eta_arr = step_E(x_arr, w_arr, mu_arr, sigma_arr)
        w_arr, mu_arr, sigma_arr = step_M(eta_arr, x_arr, mu_arr)
    return w_arr, mu_arr, sigma_arr

def plot_hist(x_arr, bins=30):
    plt.hist(x_arr, label='samples',bins=bins, density=True)

def plot_gmm(x_arr, w_arr, mu_arr, sigma_arr):
    min_x = np.min(x_arr)
    max_x = np.max(x_arr)
    X = np.linspace(min_x, max_x, 1000)
    Y = gmm_pdf(X, w_arr, mu_arr, sigma_arr)
    plt.plot(X, Y, label='estimated pdf')

def show_result(x_arr, w_arr, mu_arr, sigma_arr):
    plot_hist(x_arr)
    plot_gmm(x_arr, w_arr, mu_arr, sigma_arr)
    plt.title('the normed histogram of samples and the estimated pdf of gmm')
    plt.text(-5.3,0.21,f'n = {len(x_arr)}\nm = {len(w_arr)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.savefig('HistandEM.png')
    plt.show()

# In[]
np.random.seed(0)
x_arr = make_samples()
w_arr, mu_arr, sigma_arr = EM_algorithm(x_arr)
show_result(x_arr, w_arr, mu_arr, sigma_arr)
