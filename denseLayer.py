#!/usr/bin/env python
# coding: utf-8

# ## Fully Connected Layer

# In[13]:


from abstractLayer import AbstractLayer
import numpy as np


# In[14]:


class Dense(AbstractLayer):
    def __init__(self, n_nodes, act_fn, d_act_fn, weights=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.output_size = n_nodes
        self.bias = np.random.rand(1, self.output_size) - 0.5
        self.w = weights
        self.act_fn = np.vectorize(act_fn)
        self.d_act_fn = np.vectorize(d_act_fn)
        self.input_size = None
        self.error = None
        self.input = None
        self.output = None

    def connect(self, n_prev_nodes):
        self.input_size = n_prev_nodes

    def compile(self):
        if self.w is None:
            self.init_weights()

    def init_weights(self):
        self.w = np.random.rand(self.output_size, self.input_size) - 0.5

    def feed_forward(self, input):
        self.input = input
        self.output = self.act_fn(np.dot(self.input, self.w.T) + self.bias)
        return self.output

    def calc_error(self, output_error):
        # dc = self.d_act_fn(self.output)
        # dc = dc.reshape(output_error.shape)
        self.error = np.multiply(output_error, self.d_act_fn(self.output))
        return self.error

    def calc_prev_output_error(self):
        return np.matmul(self.error, self.w)

    def update_weights(self, prev_output, lr):
        self.w = self.w + lr * np.matmul(self.error.T, prev_output)
        self.bias = self.bias + lr * self.error

    def get_output(self):
        return self.output

    def get_output_size(self):
        return self.output_size
