#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from random import random
import sys
import argparse
import numpy as np
from simple_nn import SimpleNN, mse

def main(shape):
    print("Training NN with shape: {}".format(shape))
    num_input = shape[0]
    num_output = shape[-1]

    X_test = np.random.random((10, num_input))
    Y_test = np.random.random((10, num_output))
    print("X shape: {}".format(X_test.shape))
    print("Y shape: {}".format(Y_test.shape))

    nn = SimpleNN(shape)
    for _ in range(10):
        nn.train(X_test, Y_test, num_iter=100)
        Y_predict = nn.predict(X_test)
        print("mse: {}".format(mse(Y_test, Y_predict)))

def _tuple_type(arg_str):
    return tuple([int(s.strip()) for s in arg_str.split(",")])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", dest="shape",
                        type=_tuple_type, default=(11, 20, 3),
                        metavar="INPUT,HIDDEN,....,OUTPUT",
                        help="shape of the neural network (default: (10,20,3))")
    args = parser.parse_args()
    main(args.shape)
