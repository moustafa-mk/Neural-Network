#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


def sig(x):
    """
    sigmoid function
    :param x: input
    :return: sigmoid(input)
    """
    z = 1 / (1 + np.exp(-x))
    return z


def d_sig(z):
    """
    derivative of sigmoid function
    :param z: sig
    :return: d(sig)
    """
    return z * (1 - z)


def tanh(x):
    """
    tanh function
    :param x: input
    :return: tanh(input)
    """
    return np.tanh(x)


def d_tanh(x):
    """
    derivative of tanh function
    :param x: tanh
    :return: d(tanh)
    """
    return 1 - np.tanh(x) ** 2
