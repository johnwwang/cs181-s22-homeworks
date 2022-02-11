import numpy as np
import matplotlib.pyplot as plt
from T1_P1 import compute_loss

data = np.array(
        [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)])

x_coord = data[:, 0]

n = len(data)

tau1 = 0.1
tau2 = 2
tau3 = 100

tau_arr = [tau1, tau2, tau3]

def f(x_star, tau):
    def K(x_i, x_j):
        return np.exp(-np.power((x_i-x_j), 2)/tau)
    
    y = 0
    
    for i in range(n):
        x_i, y_i = data[i]
        y += K(x_i, x_star)*y_i
    
    return y



if name == "__main__":
    