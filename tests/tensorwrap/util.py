#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.tensor.elemwise_unary as tnu
import tadpole.tensor.reindexing     as tnreidx

import tadpole.autodiff.nary as nary
import tadpole.autodiff.grad as agrad

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Testing tools for gradients (VJPs/JVPs)                                ###
###                                                                         ###
###############################################################################


"""
def flatten(x, ind="i"):

    op = tnreidx.tensor_reindex(x)
    return op.flatten(ind)



def conj(x):

    op = tnu.tensor_elemwise_unary(x)
    return op.conj()
"""


# --- Numerical gradient ---------------------------------------------------- # 

def numerical_grad(fun, x, eps=1e-6):

    def grad(g):

        return (fun(x + g * eps/2) - fun(x - g * eps/2)) / eps    

    return grad 




# --- Assert VJP ------------------------------------------------------------ # 

def assert_vjp(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    if tn.iscomplex(y):
       dy = tn.conj(dy)

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)

    i = IndexGen("i", dx.size)
    j = IndexGen("j", dy.size)    

    vjv_out = tn.flatten(vj, i) @ tn.flatten(dx, i)
    vjv_ans = tn.flatten(dy, j) @ tn.flatten(jv, j)

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert JVP ------------------------------------------------------------ #

def assert_jvp(fun, x):

    op = agrad.DifferentialOpForward(fun, x)
    dx = tn.space(x).randn()

    jv_out = op.grad(dx)
    jv_ans = numerical_grad(fun, x)(dx)

    i = IndexGen("i", dx.size)
    
    vjv_out = tn.flatten(dx, i) @ tn.flatten(jv_out, i)
    vjv_ans = tn.flatten(dx, i) @ tn.flatten(jv_ans, i)

    assert tn.space(jv_out) == tn.space(jv_ans)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert gradients of a given mode and order ---------------------------- #

@nary.nary_op
def assert_grad(fun, x, modes=("vjp", "jvp"), order=1): # TODO CHANGE BACK TO order=2 once we implement NodeContainer! 

    if isinstance(modes, str):
       modes = (modes,)

    for mode in modes:

        {
         "vjp": assert_vjp, 
         "jvp": assert_jvp,
        }[mode](fun, x)   

        if order > 1:

           def gradfun(x, g):

               op = {
                     "vjp": agrad.DifferentialOpReverse, 
                     "jvp": agrad.DifferentialOpForward,
                    }[mode](fun, x)

               return op.grad(g)

           g = tn.space(fun(x)).randn()

           assert_grad(gradfun, (0, 1), modes=modes, order=order-1)(x, g) 











