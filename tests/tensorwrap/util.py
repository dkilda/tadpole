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


# --- Numerical gradient ---------------------------------------------------- # 

def numerical_grad(fun, x, eps=1e-6):

    def grad(g):         
        return (fun(x + g * eps/2) - fun(x - g * eps/2)) / eps    

    return grad 




# --- Assert VJP ------------------------------------------------------------ # 

def assert_vjp(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).ones() #randn()
    dy = tn.space(y).ones() #randn()

    #print("\n\n\nGRAD START")
    vj = op.grad(dy)
    #print("GRAD END\n\n\n")
    jv = numerical_grad(fun, x)(dx)

    i = IndexGen("i", dx.size)
    j = IndexGen("j", dy.size)  

    try:
       print("ASSERT-VJP-1: ", vj._inds, x._inds, dx._inds) 
    except AttributeError:
       pass

    try:
       print("ASSERT-VJP-2: ", jv._inds, y._inds, dy._inds) 
    except AttributeError:
       pass

    vjv_out = tn.flatten(vj, i) @ tn.flatten(dx, i)
    vjv_ans = tn.flatten(dy, j) @ tn.flatten(jv, j)


    try:
       print("ASSERT-VJP-3.0: ", vj._data._data, dx._data._data, " | ", jv._data._data, dy._data._data) 
    except AttributeError:
       pass

    try:
       print("ASSERT-VJP-3.1: ", vjv_out._data._data, vjv_ans._data._data) 
    except AttributeError:
       pass
    """
    """

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert JVP ------------------------------------------------------------ #

def assert_jvp(fun, x):

    op = agrad.DifferentialOpForward(fun, x)
    dx = tn.space(x).randn()

    jv_out = op.grad(dx)
    jv_ans = numerical_grad(fun, x)(dx)

    dy = tn.space(jv_ans).randn()
    i  = IndexGen("i", dy.size)
    
    vjv_out = tn.flatten(dy, i) @ tn.flatten(jv_out, i)
    vjv_ans = tn.flatten(dy, i) @ tn.flatten(jv_ans, i)

    assert tn.space(jv_out) == tn.space(jv_ans)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert real VJP ------------------------------------------------------- # 

def assert_vjp_real(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)

    i = IndexGen("i", dx.size)
    j = IndexGen("j", dy.size)  

    vjv_out = tn.real(tn.flatten(vj, i) @ tn.flatten(dx, i))
    vjv_ans = tn.real(tn.flatten(dy, j) @ tn.flatten(jv, j)) 

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert null VJP ------------------------------------------------------- # 

def assert_vjp_null(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    if tn.iscomplex(y):
       dy = tn.conj(dy)

    i = IndexGen("i", dx.size)

    vj  = op.grad(dy)
    vjv = tn.flatten(vj, i) @ tn.flatten(dx, i)

    assert tn.space(vj) == tn.space(x)
    assert isinstance(vj, tn.NullGrad) or tn.allclose(vjv, 0)




# --- Assert null JVP ------------------------------------------------------- #

def assert_jvp_null(fun, x):

    op = agrad.DifferentialOpForward(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    i  = IndexGen("i", dx.size)

    jv  = op.grad(dx)
    vjv = tn.flatten(dx, i) @ tn.flatten(jv, i)

    assert tn.space(jv) == tn.space(y)
    assert isinstance(jv, tn.NullGrad) or tn.allclose(vjv, 0) 




# --- Assert type VJP ------------------------------------------------------- # 

def assert_vjp_type(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    vj = op.grad(dy)
    assert tn.space(vj) == tn.space(x)




# --- Assert type JVP ------------------------------------------------------- #

def assert_jvp_type(fun, x):

    op = agrad.DifferentialOpForward(fun, x)
    y  = op.evaluate()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    jv = op.grad(dx)
    assert tn.space(jv) == tn.space(y)




# --- Assert gradients of a given mode and order ---------------------------- #

@nary.nary_op
def assert_grad(fun, x, modes=("vjp","jvp"), submode=None, order=2):

    if  isinstance(modes, str):
        modes = (modes,)
     
    for mode in modes:

        {
         ("vjp", None):   assert_vjp, 
         ("jvp", None):   assert_jvp,
         ("vjp", "real"): assert_vjp_real, 
         ("jvp", "real"): assert_jvp,
         ("vjp", "null"): assert_vjp_null, 
         ("jvp", "null"): assert_jvp_null,
         ("vjp", "type"): assert_vjp_type, 
         ("jvp", "type"): assert_jvp_type,
        }[mode, submode](fun, x)  

        if order > 1:

           g = {
                "vjp": tn.space(fun(x)).ones(), #.randn(),
                "jvp": tn.space(x).randn(),
               }[mode]
                  
           def gradfun(x):

               op = {
                     "vjp": agrad.DifferentialOpReverse, 
                     "jvp": agrad.DifferentialOpForward,
                    }[mode](fun, x)

               return op.grad(g)

           assert_grad(
              gradfun, modes=mode, submode=submode, order=order-1
           )(x)




