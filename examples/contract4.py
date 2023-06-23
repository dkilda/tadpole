#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
   IndexLit,
)


"""
Tensor contraction, example-4: 
a contraction pattern that occurs when evaluating the energy and its gradient 
with iPEPS tensor networks, using 1-site unit cell and Corner Transfer Matrix 
(CTM) environment. 

The example demonstrates the calculation of both the first- and 
the second-order gradients.

"""

D   = 4
chi = 10


ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for n in "udlr":
    make_index(f"D{n}", D)
    for i in "12":
        make_index(f"x{n}{i}", chi)


M = td.randn((ind["Dl"], ind["Du"], ind["Dr"], ind["Dd"]))


T = {}
for n in "udlr":
    T[n] = td.randn((ind[f"x{n}1"], ind[f"x{n}2"], ind[f"D{n}"])) 


C = {}
for i, nv in enumerate("du", 1):
  for j, nh in enumerate("lr", 1):
      C[f"{nv}{nh}"] = td.randn((ind[f"x{nv}{j}"], ind[f"x{nh}{i}"]))



def _contract(x):

    out = td.contract(C["ul"], T["u"], T["l"])
    out = td.contract(out, x)
    out = td.contract(out, C["ur"], T["r"])
    out = td.contract(out, C["dl"], T["d"])
    out = td.contract(out, C["dr"])

    return out



def contract(x):

    return _contract(x) / _contract(td.space(x).ones())



def grad(fun):

    return lambda x: td.gradient(contract)(x) @ td.space(x).ones()



def comp(msg, expr):

    print(msg, td.asdata(expr))



def main():

    comp("contraction:            ", contract(M))
    comp("contraction gradient-1: ", grad(contract)(M))   
    comp("contraction gradient-2: ", grad(grad(contract))(M)) 



main()



























