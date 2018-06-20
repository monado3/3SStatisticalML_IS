import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# In[]
def calculate_mle(n , true_theta=0.3):
    m = np.random.binomial(n, true_theta)
    return m/n

def plot_distro_of_mle(mle_func, n, true_theta, trial_num, gauss_fit=False, cnt=[0]):
    def gauss_plot():
        def gauss(x, coef, std_dev):
            return coef*norm.pdf(x, loc=true_theta, scale=std_dev)

        param, _ = curve_fit(gauss, X, Y)
        Y_ = gauss(X, param[0], param[1])
        ax.plot(X, Y_, label='fitting by gaussian')
        ax.legend()

    mle_arr = np.random.binomial(n, true_theta, trial_num)
    X, Y = np.unique(mle_arr/n, return_counts=True)
    fig, ax = plt.subplots()
    ax.plot(X, Y, label='result')
    if gauss_fit: gauss_plot() # N(true_theta, 1/(n*F^(true_theta)) )
    ax.set_xlim([0,1])
    ax.set_xticks([X/10 for X in range(10)])
    ax.set_title('the distribution of the maximum likelihood thetas')
    ax.set_xlabel('the value of maximum likelihood theta')
    ax.set_ylabel('appearance time')
    ax.text(0.51,0.6,
    f'true theta = {true_theta}\nn = {n}\nthe number of trials = {trial_num}',
    transform=ax.transAxes)
    ax.grid()
    cnt[0]+=1
    plt.savefig(f'distro_mle{cnt[0]}.png')
    plt.show()

def plot_mle_change_by_n(trial_num, true_theta):
    def generate_mle():
        cnt = 0
        for trial, x in enumerate(bernoulli_arr, 1):
            if x:
                cnt+=1
            yield cnt/trial

    bernoulli_arr = np.random.binomial(1, true_theta, trial_num)
    X = np.arange(1,trial_num+1)
    Y = np.array(list(generate_mle()))
    fig, ax = plt.subplots()
    ax.plot(X, Y)
    ax.set_title('the transition of the maximum likelihood theta with an increase in trials')
    ax.set_xlabel('the number of trials')
    ax.set_ylabel('the value of maximum likelihood theta')
    ax.grid()
    ax.text(0.7,0.8,f'true theta = {true_theta}',transform=ax.transAxes)
    plt.savefig('mle_change_by_n.png')
    plt.show()

# In[]
# parameter
np.random.seed(0)
true_theta = 0.3

# In[]
# question 1
n = 70
print(f'the maximum likelihood theta : {calculate_mle(n, true_theta):.8f}\
 (true theta = {true_theta}, n = {n})')

# In[]
# question 2
n = 100
trial_num = 500
plot_distro_of_mle(calculate_mle, n, true_theta, trial_num)

# In[]
# question 3
trial_num = int(1e6)
plot_distro_of_mle(calculate_mle, n, true_theta, trial_num, gauss_fit=True)


# In[]
trial_num = int(1e4)
plot_mle_change_by_n(trial_num, true_theta)
