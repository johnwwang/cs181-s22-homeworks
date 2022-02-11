#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import matplotlib.pyplot as plt
import random
from T1_P1 import compute_loss


# In[39]:


data = np.array(
        [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)])

x_coord = data[:, 0]
y_coord = data[:, 1]

n = len(data)

tau1 = 0.01
tau2 = 2
tau3 = 100

tau_arr = np.array([tau1, tau2, tau3])
f_star_arr = []


# In[29]:


def f(x_star, tau):
    def K(x_i, x_j):
        return np.exp(-np.power((x_i-x_j), 2)/tau)
    
    y = 0
    
    for i in range(n):
        x_i, y_i = data[i]
        y += K(x_i, x_star)*y_i
    
    return y


# In[31]:


for tau in tau_arr:
    f_star_arr.append(np.vectorize(lambda x : f(x, tau))(x_coord))


# In[34]:


plt.figure()


# In[50]:


x = np.linspace(0, 12)
for tau in tau_arr:
    c = (random.random(), random.random(), random.random())
    plt.plot(x, f(x, tau), color=c, label=str(tau))
    plt.scatter(x_coord, y_coord, color = "red")
plt.legend(loc='upper right')


# In[ ]:




