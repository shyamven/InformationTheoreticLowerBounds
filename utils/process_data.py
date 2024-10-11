import numpy as np
# from scipy import optimize

def ApproxMSELowerBound(d, l, h_x, K):
    # Calculating approximate s*
    num = 0.25 * 2**(4*l/d) * l * np.exp(-2 + 4*h_x/d)
    den = np.pi**2 * K**2 * d
    expon = d / (d - 2*l)
    s_hat = (num / den) ** expon
    
    # Substitute s* approximation into MSE LB Function
    LB = (d/(2*np.pi*np.e)) * np.exp(2*h_x/d) * (s_hat / (0.25 + s_hat))**(l/d)  - 2*K*np.sqrt(d*l*s_hat)
    
    return LB

#####################################
## Deprecated
#####################################
# def FindMSELowerBound(d, l, h, K):
#     # Initialization
#     x = 1e-100
#     original_step = 1e-100
    
#     # Initializing loop
#     step = original_step
#     add = True
#     num_steps_between = 0
#     # directional_changes = 0
    
#     # Iterative process
#     while True:
#         current_f = f(x, d, l, h, K)
#         if add == True: x += step
#         else: x -= step
        
#         next_f = f(x, d, l, h, K)
#         step *= 1.01
    
#         num_steps_between += 1
    
#         if (next_f - current_f) < 0:
#             break
        
#             # print(f"Direction change #{directional_changes} at x = {x}, f(x) = {current_f}")
#             # print(num_steps_between)
#             # if num_steps_between <= 30: break
#             # else: num_steps_between = 0
    
#             # step = original_step
#             # directional_changes += 1
#             # if directional_changes % 2 == 1: add = False
#             # else: add = True
            
#     return next_f