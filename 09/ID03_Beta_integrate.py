import numpy as np
from scipy import integrate
from scipy.stats import beta

# In[]
n = int(1e4)
threshold = 0.1
for a, b in (1, 1), (0.1, 0.1), (5, 5):
    rv = beta(5 + a, 15 + b)
    print(f'prior distribution     : Beta({a},{b})')
    print(f'posterior distribution : Beta({5+a},{15+b})')


    # E[x|data]
    def E_integrand(pi):
        return 2 * ((1 - pi) / pi) * rv.pdf(pi)

    pi_arr = np.linspace(1e-10, 1, n)
    print(f'E[x|data]              : {integrate.simps(E_integrand(pi_arr), pi_arr):.5f}')


    # Pr[pi <=  threshold|data]
    pi_arr = np.linspace(0, threshold, n)
    print(f'Pr[pi <= {threshold}|data]     : {integrate.simps(rv.pdf(pi_arr), pi_arr):.5f}', end='\n\n')