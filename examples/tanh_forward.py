#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

import numpy   as np
import tadpole as td


"""
This is a simple demo of derivative calculation using 
the forward-mode AD. It reproduces tanh differentiation, the signature 
example of autograd (https://github.com/HIPS/autograd). 

"""


i = td.IndexLit("i", 201)
w = td.ones((i,)) * 0.07  
z = td.astensor(np.arange(-100,101), (i,))
 

def tanh(x):
    return (1.0 - td.exp(-x)) / (1.0 + td.exp(-x)) 


def grad(fun):
    return td.derivative(fun)


def plot(*args):
    plt.plot(*map(td.asdata, args))


def main():

    x = w * z

    plot(
         x, tanh(x), 
         x, grad(tanh)(x),
         x, grad(grad(tanh))(x), 
         x, grad(grad(grad(tanh)))(x), 
         x, grad(grad(grad(grad(tanh))))(x), 
         x, grad(grad(grad(grad(grad(tanh)))))(x), 
         x, grad(grad(grad(grad(grad(grad(tanh))))))(x), 
        )
    

main()
plt.savefig("tanh_forward.png")







