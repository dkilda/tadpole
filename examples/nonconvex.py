#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

import numpy   as np
import tadpole as td
import scipy.optimize


"""
An introductory example of nonconvex optimization (using the Rosenbrock 
function https://en.wikipedia.org/wiki/Rosenbrock_function). 

Due to its straightforward nature, the use of tensors and indexing is 
unnecessary in this example, but it demonstrates how to perform 
gradient-based optimization using tadpole.

"""


x = np.array([0.0, 0.0])


def fun(x):

    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2  


def gradfun(x):

    inp = td.astensor(x, (td.IndexLit("i", 2),))
    out = td.evaluate_with_gradient(fun)(inp)

    return td.asdata(out[0]), td.asdata(out[1])


def main():
    
    out = scipy.optimize.minimize(gradfun, x, jac=True, method='CG')
    print(f"The minimum fun(x) = {fun(out.x)} is located at x = {out.x}. ")
    print(f"The exact minimum: 0. The exact location: [1, 1].")    


main()






