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

import tadpole.autodiff.nary  as nary
import tadpole.autodiff.grad  as agrad

import tests.tensor.fakes as fake
import tests.tensor.data  as data




###############################################################################
###                                                                         ###
###  Testing tools for gradients (VJPs/JVPs)                                ###
###                                                                         ###
###############################################################################


# --- Numerical gradient ---------------------------------------------------- # 

def numerical_grad(fun, x, eps=1e-6):

    def grad(g):

        return (fun(x + g * eps/2) - fun(x - g * eps/2)) / eps    

    return grad 




# --- Assert VJP ------------------------------------------------------------ # 

def assert_vjp(fun, x):

    op = agrad.ReverseDifferentialOp(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    if tn.iscomplex(y):
       dy = tn.conj(dy)

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)
    
    vjv_out = tn.flatten(vj) @ tn.flatten(dx)
    vjv_ans = tn.flatten(dy) @ tn.flatten(jv)

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert JVP ------------------------------------------------------------ #

def assert_jvp(fun, x):

    op = agrad.ForwardDifferentialOp(fun, x)
    dx = tn.space(x).randn()

    jv_out = op.grad(dx)
    jv_ans = numerical_grad(fun, x)(dx)
    
    vjv_out = tn.flatten(dx) @ tn.flatten(jv_out)
    vjv_ans = tn.flatten(dy) @ tn.flatten(jv_ans)

    assert tn.space(jv_out) == tn.space(jv_ans)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert gradients of a given mode and order ---------------------------- #

@nary.nary_op
def assert_grad(fun, x, modes=("vjp", "jvp"), order=2):

    if isinstance(modes, str):
       modes = (modes,)

    for mode in modes:

        {
         "vjp": assert_vjp, 
         "jvp": assert_jvp,
        }[mode](fun, x)   

        if order > 1:

           def fun(x, g):

               op = {
                     "vjp": agrad.ReverseDifferentialOp, 
                     "jvp": agrad.ForwardDifferentialOp,
                    }[mode](fun, x)

               return op.grad(g)

           g = tn.space(fun(x)).randn()

           assert_grad(fun, (0, 1), modes=modes, order=order-1)(x, g)




