#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np

class SimpleNN(object):
    """
    Simple Backpropagation Neural Network
    """
    def __init__(self, _shape):
        self._shape = tuple(_shape)
        self._weights = [2*np.random.random((dim_out, dim_in)) - 1
                         for dim_in, dim_out in zip(self._shape[:-1], self._shape[1:])]
        self._biases = [2*np.random.random(dim)-1 for dim in self._shape[1:]]

    @property
    def shape(self):
        """
        the shape of the neural network
        """
        return self._shape

    @property
    def weights(self):
        """
        weights of each layers

        note
        ====
        - `self.weights[i]` is the weight from layer i to layer i+1
        """
        return deepcopy(self._weights)


    def train(self, X, Y, nu=0.1, num_iter=1000):
        """
        params
        ======
        - X <numpy.ndarray>: N by k array, where k is the
          number of neurons at input layer
        - Y <numpy.ndarrat>: N by m array, where m is the
          number of neurons at output layer
        - num_iter <int>: number of iteration of training
          (default: 1000)
        - nu <float>: learning rate (default: 0.1)
        """
        N = X.shape[0]
        for _ in range(num_iter):
            # forward propagation
            activations = self._forward_prob(X)
            # back propagation
            ## find deltas
            deltas = [(Y.T-activations[-1])*grad_sigmoid(activations[-1])]
            for i, (weight, act) in enumerate(zip(self._weights[::-1], activations[:-1][::-1])):
                delta_next = deltas[i]
                if i < len(self.shape)-1:
                    # non-input layer
                    delta = weight.T.dot(delta_next) * grad_sigmoid(act)
                else:
                    # input layer
                    delta = weight.T.dot(delta_next) * act
                deltas.append(delta)
            deltas = deltas[::-1]

            for i in range(len(self._weights)):
                act = activations[i+1]
                delta_w = deltas[i]
                delta_b = deltas[i+1]
                self._weights[i] += nu * (act.dot(delta_w.T))/N
                self._biases[i] += nu * delta_b.mean(axis=1)

    def predict(self, X):
        """
        make prediction on given data

        params
        ======
        - X <numpy.ndarray>: Nxk array. N is the number of data, k is the
          number of neurons at input layer.
        """
        assert X.shape[1] == self._weights[0].shape[1], \
            "incompatible input data shape with input layer's wieght: {}".format(X.shape[1])
        return self._forward_prob(X)[-1].T

    def _forward_prob(self, X):
        """
        params
        ======
        - X: N by k array, k is the number of input neurons
        """
        activations = [X.T] # input layer
        for i, (weight, bias) in enumerate(zip(self._weights, self._biases)):
            act = sigmoid(weight.dot(activations[i])+bias[:,None])
            activations.append(act)
        return activations

    def __str__(self):
        _str = "SimpleNN: " + " x ".join(map(str, self._shape))
        return _str

def sigmoid(X, eposilon=0.01):
    """
    sigmoid function
    """
    return 1/(1+eposilon+np.exp(-X))

def grad_sigmoid(X):
    return sigmoid(X) * (1-sigmoid(X))

def mse(Y1, Y2):
    """
    mean square error
    """
    if len(Y1.shape) == 1:
        Y1 = Y1[None, :]
    if len(Y2.shape) == 1:
        Y2 = Y2[None, :]
    diff = Y1 - Y2
    return np.sqrt((diff*diff).sum(axis=1)).mean()
