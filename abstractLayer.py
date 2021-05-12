#!/usr/bin/env python
# coding: utf-8

# ## Abstract Class Layer

# In[1]:


import numpy as np


# In[ ]:


class AbstractLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def feed_forward(self, input):
        raise NotImplementedError

    def update_weights(self, prev_output, lr):
        raise NotImplementedError
