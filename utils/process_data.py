import numpy as np
from scipy import optimize
from scipy.stats import truncnorm
import torch

def FindMSELowerBound(d, l, h, K):
    start = 10**-15
    while f(start, d, l, h, K) < 0:
        start /= 10
    end = 10**2
    while f(end, d, l, h, K) > 0:
        end *= 10
    assert f(start, d, l, h, K) > 0.0
    assert f(end, d, l, h, K) < 0.0
    
    # Use lambda to create a new function that only varies x
    f_p_optimized = lambda x: f_p(x, d, l, h, K)
    
    root = optimize.bisect(f_p_optimized, start, end)
    assert root > 0
    maximum = f(root, d, l, h, K)
    return maximum
    
    
def f(x, d, l, h, K):
    power = (d+l-1)/d
    term1 = (d/ ((2*np.pi*np.e)**power)) * np.exp(2*h/d)
    term2 = (0.25 + x)**(-l/d)
    term3 = x**((1-d)/d)

    return x * (term1*term2*term3 - K*l)


def f_p(x, d, l, h, K):
    power = (d+l-1)/d
    A = (d/ ((2*np.pi*np.e)**power)) * np.exp(2*h/d)
    B = K*l
    num = ((x+0.25)**(-1 - l/d)) * (-A*l*x**(1+l/d) + A*x**(1/d)*(x + 0.25) - B*d*x*(x+0.25)**(1+l/d))
    return num/(d*x)


def generate_truncated_normal_data(train_size, input_dim, lower_bound=0, upper_bound=1, mu=0, sigma=1):
    # Standardizing the bounds for truncation
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

    # Generating truncated normal data
    data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(train_size, input_dim))

    # Converting to PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)