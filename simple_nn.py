#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np

class SimpleNN(object):

    def __init__(self, _shape):

        self._shape = tuple(_shape)
        self._weights = [np.random.random((dim_out, dim_in))
                         for dim_in, dim_out in zip(self._shape[:-1], self._shape[1:])]
        self._biases = [np.random.random(dim) for dim in self._shape[1:]]

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
        for _ in range(num_iter):
            # forward propagation
            activations = self._forward_prob(X)
            # back propagation
            ## find deltas
            ## deltas[i] = delta at layer L-i, where L is the
            ## total number of layers
            deltas = [(Y.T-activations[-1])*grad_sigmoid(activations[-1])]
            for i, (weight, act) in enumerate(zip(self._weights[::-1], activations[:-1][::-1])):
                delta_next = deltas[i]
                delta = weight.T.dot(delta_next) * grad_sigmoid(act)
                deltas.append(delta)

            for i in range(len(self._weights)):
                weight = self._weights[i]
                act = activations[i]
                self._weights[i] -= nu * act * deltas[-(i+1)]
                self._biases[i] -= nu * deltas[-(i+1)].mean(axis=1)

    def predict(self, X):
        """
        make prediction on given data

        params
        ======
        - X <numpy.ndarray>: Nxk array. N is the number of data, k is the
          number of neurons at input layer.
        """
        assert X.shape[1] == self._weights[0].shape[0], \
            "incompatible input data shape with input layer's wieght: {}".format(X.shape[1])
        return self._forward_prob(X)[-1].T

    def _forward_prob(self, X):
        """
        params
        ======
        - X: N by k array, k is the number of input neurons
        """
        activations = [sigmoid(self._weights[0].dot(X.T)+self._biases[0][:,None])]
        for i, (weight, bias) in enumerate(zip(self._weights[1:], self._biases[1:])):
            act = sigmoid(weight.dot(activations[i])+bias[:,None])
            activations.append(act)

        return activations

def sigmoid(X):
    """
    sigmoid function
    """
    return 1/(1+np.exp(-X))

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
