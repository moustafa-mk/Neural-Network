#!/usr/bin/env python
# coding: utf-8

# In[15]:
import random

from network import NN
from denseLayer import Dense
import numpy as np
from activations import sig
from activations import d_sig
from activations import tanh
from activations import d_tanh

# In[16]:

if __name__ == '__main__':
    random.seed(1)

    x_train = np.array(
        [[[0, 0]], [[0, 5]], [[1, 1]], [[1, 4]], [[2, 1]], [[2, 2]], [[2, 3]], [[2, 5]], [[3, 0]], [[3, 2]], [[3, 3]],
         [[3, 4]]])
    y_train = np.array([[[0]], [[1]], [[0]], [[1]], [[0]], [[0]], [[1]], [[1]], [[0]], [[1]], [[1]], [[0]]])

    net = NN()
    net.add(Dense(n_nodes=4, act_fn=sig, d_act_fn=d_sig))
    net.add(Dense(n_nodes=1, act_fn=sig, d_act_fn=d_sig))

    # train
    net.fit(x_train, y_train, epochs=100, lr=0.05)

    # test
    out = net.predict_list(x_train)
    print(out)
    out = np.round(out)
    print(out)
    accuracy = net.accuracy(x_train, y_train)
    print(accuracy)
