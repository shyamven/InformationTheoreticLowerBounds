import numpy as np
from scipy import optimize


def FindMSELowerBound(d, l, h, K):
    # Initialization
    x = 1e-60
    original_step = 1e-60
    
    # Initializing loop
    step = original_step
    add = True
    num_steps_between = 0
    # directional_changes = 0
    
    # Iterative process
    while True:
        current_f = f(x, d, l, h, K)
        if add == True: x += step
        else: x -= step
        
        next_f = f(x, d, l, h, K)
        step *= 1.01
    
        num_steps_between += 1
    
        if (next_f - current_f) < 0:
            break
        
            # print(f"Direction change #{directional_changes} at x = {x}, f(x) = {current_f}")
            # print(num_steps_between)
            # if num_steps_between <= 30: break
            # else: num_steps_between = 0
    
            # step = original_step
            # directional_changes += 1
            # if directional_changes % 2 == 1: add = False
            # else: add = True
            
    return next_f


def f(x, d, l, h, K):
    power = (d+l-1)/d
    term1 = (d/ ((2*np.pi*np.e)**power)) * np.exp(2*h/d)
    term2 = (0.25 + x)**(-l/d)
    term3 = x**((1-d)/d)

    return x * (term1*term2*term3 - K*l)


# def f_p(x, d, l, h, K):
#     power = (d+l-1)/d
#     A = (d/ ((2*np.pi*np.e)**power)) * np.exp(2*h/d)
#     B = K*l
#     num = ((x+0.25)**(-1 - l/d)) * (-A*l*x**(1+l/d) + A*x**(1/d)*(x + 0.25) - B*d*x*(x+0.25)**(1+l/d))
#     return num/(d*x)


# def FindMSELowerBound(d, l, h, K):
#     start = 10**-15
#     while f(start, d, l, h, K) < 0:
#         start /= 10
#     end = 10**2
#     while f(end, d, l, h, K) > 0:
#         end *= 10
#     assert f(start, d, l, h, K) > 0.0
#     assert f(end, d, l, h, K) < 0.0
    
#     # Use lambda to create a new function that only varies x
#     f_p_optimized = lambda x: f_p(x, d, l, h, K)
    
#     root = optimize.bisect(f_p_optimized, start, end)
#     assert root > 0
#     maximum = f(root, d, l, h, K)
#     return maximum