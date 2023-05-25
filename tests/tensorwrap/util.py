#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import itertools
import numpy as np

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.array     as ar
import tadpole.container as tc
import tadpole.tensor    as tn
import tadpole.index     as tid

import tadpole.autodiff.nary as nary
import tadpole.autodiff.grad as agrad

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.index import (
   Index,
   IndexGen, 
   IndexLit,
   Indices,
)




###############################################################################
###                                                                         ###
###  Testing tools for gradients (VJPs/JVPs)                                ###
###                                                                         ###
###############################################################################


# --- Map operator for both tensors and containers -------------------------- #

def tmap(fun, *args):

    if all(isinstance(arg, tc.Container) for arg in args):
       return tc.cmap(fun, *args)

    return fun(*args)




# --- Sum operator for both tensors and containers -------------------------- #

def tsum(fun, *args):

    if all(isinstance(arg, tc.Container) for arg in args):
       return tc.csum(fun, *args)

    return fun(*args)




# --- Dot product of flattened tensors -------------------------------------- #

def dot(x, y):

    def _dot(x, y):
        i = IndexGen("i", x.size)
        return tn.flatten(x, i) @ tn.flatten(y, i)

    return tsum(_dot, x, y)




# --- Numerical gradient ---------------------------------------------------- # 

def numerical_grad(fun, x, eps=1e-6):

    def grad(g):

        out1 = fun(tmap(lambda a,b: a + b * eps/2, x, g))
        out2 = fun(tmap(lambda a,b: a - b * eps/2, x, g))

        return tmap(lambda a,b: (a - b) / eps, out1, out2)

    return grad 




# --- Assert VJP ------------------------------------------------------------ # 

def assert_vjp(fun, x):

    op = agrad.diffop_reverse(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)

    vjv_out = dot(vj, dx) 
    vjv_ans = dot(jv, dy) 

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert JVP ------------------------------------------------------------ #

def assert_jvp(fun, x):

    op = agrad.diffop_forward(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()

    jv_out = op.grad(dx)
    jv_ans = numerical_grad(fun, x)(dx)

    dy = tn.space(jv_ans).randn()
    
    vjv_out = dot(dy, jv_out) 
    vjv_ans = dot(dy, jv_ans)  

    assert tn.space(jv_out) == tn.space(y)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert real VJP ------------------------------------------------------- # 

def assert_vjp_real(fun, x):

    op = agrad.diffop_reverse(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)

    vjv_out = tn.real(dot(vj, dx)) 
    vjv_ans = tn.real(dot(jv, dy)) 

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert null VJP ------------------------------------------------------- # 

def assert_vjp_null(fun, x):

    op = agrad.diffop_reverse(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    if tn.iscomplex(y):
       dy = tn.conj(dy)

    vj  = op.grad(dy)
    vjv = dot(vj, dx)

    assert tn.space(vj) == tn.space(x)
    assert isinstance(vj, tn.NullGrad) or tn.allclose(vjv, 0)




# --- Assert null JVP ------------------------------------------------------- #

def assert_jvp_null(fun, x):

    op = agrad.diffop_forward(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    jv  = op.grad(dx)
    vjv = dot(jv, dy) 

    assert tn.space(jv) == tn.space(y)
    assert isinstance(jv, tn.NullGrad) or tn.allclose(vjv, 0) 




# --- Assert type VJP ------------------------------------------------------- # 

def assert_vjp_type(fun, x):

    op = agrad.diffop_reverse(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    vj = op.grad(dy)
    assert tn.space(vj) == tn.space(x)




# --- Assert type JVP ------------------------------------------------------- #

def assert_jvp_type(fun, x):

    op = agrad.diffop_forward(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

    jv = op.grad(dx)
    assert tn.space(jv) == tn.space(y)




# --- Assert VJP with customized gradients ---------------------------------- #

def assert_vjp_custom(fun, x, g):

    if isinstance(x, tuple):
       x = tc.container(x)

    if isinstance(g, tuple):
       g = tc.container(g)

    op = agrad.diffop_reverse(fun, x)
    y  = op.value()

    dx = tn.space(x).randn()
    dy = g

    vj = op.grad(dy)
    jv = numerical_grad(fun, x)(dx)

    vjv_out = tn.real(dot(vj, dx))  
    vjv_ans = tn.real(dot(jv, dy)) 

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert JVP with customized gradients ---------------------------------- #

def assert_jvp_custom(fun, x, g):

    op = agrad.diffop_forward(fun, x)
    y  = op.value()

    dx = g

    jv_out = op.grad(dx)
    jv_ans = numerical_grad(fun, x)(dx)

    dy = tn.space(jv_ans).randn()
    
    vjv_out = dot(dy, jv_out) 
    vjv_ans = dot(dy, jv_ans)  

    assert tn.space(jv_out) == tn.space(y)
    assert tn.allclose(vjv_out, vjv_ans) 




# --- Assert gradients of a given mode and order ---------------------------- #

@nary.nary_op
def assert_grad(fun, x, modes=("vjp","jvp"), submode=None, order=2):

    if isinstance(x, tuple):
       x = tc.container(x)

    if isinstance(modes, str):
       modes = (modes,)
     
    for mode in modes:

        {
         ("vjp", None       ): assert_vjp, 
         ("jvp", None       ): assert_jvp,
         ("vjp", "real"     ): assert_vjp_real, 
         ("jvp", "real"     ): assert_jvp,
         ("vjp", "null"     ): assert_vjp_null, 
         ("jvp", "null"     ): assert_jvp_null,
         ("vjp", "type"     ): assert_vjp_type, 
         ("jvp", "type"     ): assert_jvp_type,
        }[mode, submode](fun, x)  

        if order > 1:

           g = {
                "vjp": lambda: tn.space(fun(x)).randn(),
                "jvp": lambda: tn.space(x).randn(),
               }[mode]()
     
           def gradfun(x, g):

               op = {
                     "vjp": agrad.diffop_reverse, 
                     "jvp": agrad.diffop_forward,
                    }[mode](fun, x)

               return op.grad(g)

           assert_grad(
              gradfun, (0, 1), modes=mode, submode=submode, order=order-1
           )(x, g)










"""

@nary.nary_op
def assert_grad(fun, x, modes=("vjp","jvp"), submode=None, order=2):

    if  isinstance(modes, str):
        modes = (modes,)
     
    for mode in modes:

        {
         ("vjp", None       ): assert_vjp, 
         ("jvp", None       ): assert_jvp,
         ("vjp", "real"     ): assert_vjp_real, 
         ("jvp", "real"     ): assert_jvp,
         ("vjp", "null"     ): assert_vjp_null, 
         ("jvp", "null"     ): assert_jvp_null,
         ("vjp", "type"     ): assert_vjp_type, 
         ("jvp", "type"     ): assert_jvp_type,
        }[mode, submode](fun, x)  

        if order > 1:

           g = {
                "vjp": lambda: tn.space(fun(x)).randn(),
                "jvp": lambda: tn.space(x).randn(),
               }[mode]()
     
           def gradfun(x):

               op = {
                     "vjp": agrad.diffop_reverse, 
                     "jvp": agrad.diffop_forward,
                    }[mode](fun, x)

               return op.grad(g)

           assert_grad(
              gradfun, modes=mode, submode=submode, order=order-1
           )(x)

"""


