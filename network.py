#!/usr/bin/env python
# coding: utf-8

# In[2]:
import random

import abstractLayer
import numpy as np


# In[3]:


def mse(target, output):
    """
    calculate mean squared error
    :param target: intended output
    :param output: actual output
    :return: mean squared error between target and output
    """
    se = np.subtract(target, output) ** 2
    return se.mean()


class NN:
    """
    Class for Neural Network
    """

    def __init__(self):
        self.layers = []
        self.round = np.vectorize(round)

    def add(self, layer):
        """
        add a layer to the neural network
        :param layer: layer to be added
        :return:
        """
        self.layers.append(layer)

    def compile(self, input_size):
        """
        called when fit is first called.
        connects layers by setting the input size of each layer by the output size of the previous layer.
        :param input_size: number of nodes of the input layer
        :return:
        """
        self.layers[0].connect(input_size)
        self.layers[0].compile()
        for i in range(1, len(self.layers)):
            prev_output_size = self.layers[i - 1].get_output_size()
            self.layers[i].connect(prev_output_size)
            self.layers[i].compile()

    def feed_forward(self, X):
        """
        apply forward propagation and calculate output of neural network
        :param X: input sample
        :return: output of the output layer
        """
        output = self.layers[0].feed_forward(X)
        for i in range(1, len(self.layers)):
            output = self.layers[i].feed_forward(output)
        return output

    def back_propagate(self, input, output_error, lr):
        """
        apply backward propagation for learning weights
        :param input: training sample
        :param output_error: error in the output layer
        :param lr: learning rate
        :return:
        """
        self.layers[-1].calc_error(output_error)
        output_error = self.layers[-1].calc_prev_output_error()

        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].calc_error(output_error)
            output_error = self.layers[i].calc_prev_output_error()

        prev_output = input
        for i in range(0, len(self.layers)):
            self.layers[i].update_weights(prev_output, lr)
            prev_output = self.layers[i].get_output()

    def fit(self, X_train, y_train, epochs=1000, lr=0.1):
        """
        apply feedforward and backpropagation for each epoch
        :param X_train: training data features
        :param y_train: training data output
        :param epochs: number of required iterations
        :param lr: learning rate
        :return:
        """
        self.compile(len(X_train[0].flatten()))
        samples = len(X_train)
        for epoch in range(epochs):
            for j in range(samples):
                sample = X_train[j]
                output = self.feed_forward(sample)

                error = mse(y_train[j], output)

                output_error = np.subtract(y_train[j], output)
                self.back_propagate(sample, output_error, lr)

                print('epoch %d/%d   error=%f' % (epoch + 1, epochs, error / samples))

    def predict_list(self, X):
        """
        predicts output for a list of inputs
        :param X: list of inputs
        :return: list of predictions
        """
        output = []
        for i in range(len(X)):
            output.append(self.feed_forward(X[i]))
        return output

    def predict(self, x):
        """
        predicts output for input x
        :param x: input sample
        :return: prediction for output of x
        """
        return self.feed_forward(x)

    def accuracy(self, X_test, y_test):
        """
        calculates accuracy of the network
        :param X_test: test data features
        :param y_test: test data output
        :return: accuracy of neural network
        """
        pred = self.round(self.predict_list(X_test))
        correct = np.sum(pred == y_test)
        return correct * 100 / len(y_test)
