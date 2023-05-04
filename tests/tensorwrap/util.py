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

import tadpole.tensorwrap.container as tc

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

    dx = tn.space(x).randn()
    dy = tn.space(y).randn()

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




# --- Assert container VJP -------------------------------------------------- # 

"""
def execute(fun, x):

    if isinstance(x, tn.Tensor):
       return fun(x)

    return tc.ContainerGen([fun(xi) for xi in x])



def _istensor(x):

    return isinstance(x, tn.Tensor)



def _dot(x, y):

    if isinstance(x, tn.NullGrad) or isinstance(y, tn.NullGrad):
       return 0


    print("_DOT: ", x, y, x.shape, y.shape)

    i = IndexGen("i", y.size)

    return tn.flatten(x, i) @ tn.flatten(y, i)



def dot(x, y):

    if _istensor(x) and _istensor(y): 

       return _dot(x, y) 


    if not _istensor(x) and not _istensor(y):

       return sum(_dot(xe, ye) for xe, ye in zip(x, y)) 


    raise ValueError(
       f"dot: invalid combination of input types: {type(x)}, {type(y)}"
    )
"""






def assert_vjp_container(fun, x):

    op = agrad.DifferentialOpReverse(fun, x)
    y  = op.evaluate()

    print("\n\nASSERT-VJP-1: ", x, y)

    dx = tn.space(x).randn() 
    dy = tn.space(y).randn() 

    print("ASSERT-VJP-2: ", dx, dy)

    vj = op.grad(dy)
    jv = numerical_grad_container(fun, x)(dx)  # fun(dx) 

    print("ASSERT-VJP-3-1: ", vj)
    print("ASSERT-VJP-3-2: ", jv)

    vjv_out = dot(vj, dx)
    vjv_ans = dot(dy, jv)

    print("ASSERT-VJP-3-1: ", vjv_out._data._data)
    print("ASSERT-VJP-3-2: ", vjv_ans._data._data)

    assert tn.space(vj) == tn.space(x)
    assert tn.allclose(vjv_out, vjv_ans)




# --- Assert container JVP -------------------------------------------------- #

def assert_jvp_container(fun, x):

    op = agrad.DifferentialOpForward(fun, x)
    dx = tn.space(x).randn()

    print("\n\nASSERT-JVP-1: ", x, fun(x))

    jv_out = op.grad(dx)
    jv_ans = numerical_grad_container(fun, x)(dx) # fun(dx) # numerical_grad_container(fun, x)(dx)

    dy = tn.space(jv_ans).randn()

    print("ASSERT-JVP-2: ", dx, dy)

    print("ASSERT-JVP-3-1: ", jv_out)
    print("ASSERT-JVP-3-2: ", jv_ans)

    vjv_out = dot(dy, jv_out)
    vjv_ans = dot(dy, jv_ans)

    print("ASSERT-JVP-3-1: ", vjv_out._data._data)
    print("ASSERT-JVP-3-2: ", vjv_ans._data._data)

    assert tn.space(jv_out) == tn.space(jv_ans)
    assert tn.allclose(vjv_out, vjv_ans) 





# --- Helpers for container VJP/JVP ----------------------------------------- #

def dot(x, y):

    def _dot(u, v):
        i = IndexGen("i", u.size)
        return tn.flatten(u, i) @ tn.flatten(v, i)

    if isinstance(x, tn.Tensor):
       return _dot(x, y)

    print("DOT: ", x, y)

    return sum(_dot(xe, ye) for xe, ye in zip(x, y))  



def numerical_grad_container(fun, x, eps=1e-6):

    if isinstance(x, tn.Tensor):
       x = tc.ContainerGen([x])

    def grad(g):

        #if isinstance(g, tn.Tensor):
        #   print("NUMGRAD: ", x, g, fun(x + g * eps/2), fun(x - g * eps/2))
        #   return (fun(x + g * eps/2) - fun(x - g * eps/2)) / eps 




        if isinstance(g, tn.Tensor):
           g = tc.ContainerGen([g])
   
        x1 = []  
        x2 = [] 

        for xe, ge in zip(x, g): 

            xe1 = xe + ge * eps/2
            xe2 = xe - ge * eps/2

            x1.append(xe1)
            x2.append(xe2)

        x1 = tc.ContainerGen(x1)
        x2 = tc.ContainerGen(x2)

        out1 = fun(x1) 
        out2 = fun(x2)

        #print("INP: ", x1[0]._data._data,             x2[0]._data._data)
        #print("OUT: ", out1[0].todense()._data._data, out2[0].todense()._data._data)

        if   isinstance(out1, tn.Tensor):
             out = (out1 - out2) / eps
        else:
             out = []
             for o1, o2 in zip(out1, out2):
                 #print("o1, o2: ", (o1._data._data - o2._data._data) / eps)
                 out.append((o1 - o2) / eps)
             out = tc.ContainerGen(out)
             
        return out

    return grad 






# --- Assert gradients of a given mode and order ---------------------------- #

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
         ("vjp", "container"): assert_vjp_container, 
         ("jvp", "container"): assert_jvp_container,
        }[mode, submode](fun, x)  

        if order > 1:

           g = {
                "vjp": lambda: tn.space(fun(x)).randn(),
                "jvp": lambda: tn.space(x).randn(),
               }[mode]()
                  
           def gradfun(x):

               op = {
                     "vjp": agrad.DifferentialOpReverse, 
                     "jvp": agrad.DifferentialOpForward,
                    }[mode](fun, x)

               return op.grad(g)

           assert_grad(
              gradfun, modes=mode, submode=submode, order=order-1
           )(x)




