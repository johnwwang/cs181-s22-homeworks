#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import math

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]


n = len(data)
    
def compute_loss(tau):
    
    def K(x_i, x_j):
        return np.exp(-np.power((x_i-x_j), 2)/tau)
    
    loss = 0    
    
    for i in range(n):
        x_i, y_i = data[i]
        loss_i = y_i
        
        for j in range(n):
            if i == j:
                continue
            
            x_j, y_j = data[j]
            loss_i -= K(x_i, x_j)*y_j
        
        loss += np.power(loss_i, 2)
        
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))