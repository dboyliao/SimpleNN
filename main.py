#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from random import random
import sys
import argparse
import numpy as np
from simple_nn import SimpleNN, mse

def main():
    X_test = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1]])
    Y_test = np.array([[0],
                       [0],
                       [0],
                       [1]])

    print("X shape: {}".format(X_test.shape))
    print("Y shape: {}".format(Y_test.shape))

    np.random.seed(3690)
    nn = SimpleNN((2, 2, 2, 1))
    # nn = SimpleNN((3, 4, 1))
    print(nn)
    for _ in range(10):
        nn.train(X_test, Y_test, num_iter=1000)
        Y_predict = nn.predict(X_test)
        print("mse: {}".format(mse(Y_test, Y_predict)))
    print("target:\n{}".format(Y_test))
    print("prediction:\n{}".format(Y_predict))

def _tuple_type(arg_str):
    return tuple([int(s.strip()) for s in arg_str.split(",")])

if __name__ == "__main__":
    main()
