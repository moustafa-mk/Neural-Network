#!/usr/bin/env python
# coding: utf-8

# ## Fully Connected Layer

# In[13]:


from abstractLayer import AbstractLayer
import numpy as np


# In[14]:


class Dense(AbstractLayer):
    """
    Class for fully connected dense layer
    """

    def __init__(self, n_nodes, act_fn, d_act_fn, weights=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.output_size = n_nodes
        self.bias = np.random.rand(1, self.output_size) - 0.5
        self.w = weights
        self.act_fn = np.vectorize(act_fn)  # vectorized form of the activation function for parallel computations
        self.d_act_fn = np.vectorize(d_act_fn)
        self.input_size = None
        self.error = None
        self.input = None
        self.output = None

    def connect(self, n_prev_nodes):
        """
        set input size using number of nodes of preceding layer
        :param n_prev_nodes: number of nodes of preceding layer
        :return:
        """
        self.input_size = n_prev_nodes

    def compile(self):
        """
        setup the layer
        :return:
        """
        if self.w is None:
            self.init_weights()

    def init_weights(self):
        """
        initialize weights with random values between -0.5 and 0.5
        :return:
        """
        self.w = np.random.rand(self.output_size, self.input_size) - 0.5

    def feed_forward(self, input):
        """
        applies forward propagation by multiplying inputs by their corresponding weights and apply activation function
        :param input:
        :return:
        """
        self.input = input
        self.output = self.act_fn(np.dot(self.input, self.w.T) + self.bias)
        return self.output

    def calc_error(self, output_error):
        """
        calculate error for each node in the layer
        :param output_error: error in the output of the layer
        :return:
        """
        self.error = np.multiply(output_error, self.d_act_fn(self.output))
        return self.error

    def calc_prev_output_error(self):
        """
        calculate output error for the preceding layer
        :return:
        """
        return np.matmul(self.error, self.w)

    def update_weights(self, prev_output, lr):
        """
        update layer weights w.r.t the layer error
        :param prev_output: output of preceding layer
        :param lr: learning rate
        :return:
        """
        self.w = self.w + lr * np.matmul(self.error.T, prev_output)
        self.bias = self.bias + lr * self.error

    def get_output(self):
        """
        getter function for layer output
        :return: output of layer
        """
        return self.output

    def get_output_size(self):
        """
        getter functions for number of nodes of layer
        :return: number of nodes of current layer
        """
        return self.output_size
