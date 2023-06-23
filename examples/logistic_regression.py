#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

import numpy   as np
import tadpole as td
import scipy.optimize


"""
Logistic regression using gradients calculated by tadpole.

"""


i = td.IndexLit("i", 4)
j = td.IndexLit("j", 3)

x = td.TensorGen([[0.52, 1.12,  0.77],
                  [0.88, -1.08, 0.15],
                  [0.52, 0.06, -1.30],
                  [0.74, -2.49, 1.39]], (i, j))

targets = td.TensorGen([1,1,0,1], (i, ))


def lossfun(w):

    y = (td.tanh(x @ w) + 1) / 2
    z = y * targets + (1 - y) * (1 - targets)

    return -td.sumover(td.log(z))


def gradfun(w):

    return td.gradient(lossfun)(w)


def main():

    w = td.zeros((j, ), dtype="float64")

    print("\n\nInitial loss: ", td.asdata(lossfun(w)))

    for i in range(100):
        w = w - gradfun(w) * 0.01

    print("Optimized loss: ", td.asdata(lossfun(w)))


main()










