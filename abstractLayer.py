#!/usr/bin/env python
# coding: utf-8

# ## Abstract Class Layer

# In[1]:


import numpy as np


# In[ ]:


class AbstractLayer:
    """
    abstract class for neural network layers
    """
    def __init__(self):
        self.input = None
        self.output = None

    def feed_forward(self, input):
        """
        abstract function for forward propagation
        :param input: training data input sample
        :return: exception
        """
        raise NotImplementedError

    def update_weights(self, prev_output, lr):
        """
        update weights of layer
        :param prev_output: output of preceding layer
        :param lr: learning rate
        :return: exception
        """
        raise NotImplementedError
