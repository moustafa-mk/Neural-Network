#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


def sig(x):
    z = 1 / (1 + np.exp(-x))
    return z


def d_sig(z):
    return z * (1 - z)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2
