import numpy as np
from scipy import integrate
from scipy.stats import beta

# In[]
n = int(1e4)
for a, b in (1, 1), (0.1, 0.1), (5, 5):
    rv = beta(4 + a, 1 + b)
    for start in 0.5, 0.8:
        x = np.linspace(start, 1, n)
    print(f'prior distribution     : Beta({a},{b})',
          f'posterior distribution : Beta({4+a},{1+b})',
          f'prob(pi >= {start}|data)   : {integrate.simps(rv.pdf(x), x):.5f}',
          sep='\n', end='\n\n')
