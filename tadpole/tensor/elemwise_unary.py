#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core as core


from tadpole.tensor.types import (
   Pluggable,
   Engine,
)


from tadpole.tensor.engine import (
   EngineUnary,
   EngineElemwise,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor unary elementwise engine and operator                           ###
###                                                                         ###
###############################################################################


# --- Tensor unary elementwise factory -------------------------------------- #

def tensor_elemwise_unary(x):

    engine = x.pluginto(EngineElemwiseUnary())
    return engine.operator()




# --- Tensor unary elementwise engine --------------------------------------- #

class EngineElemwiseUnary(Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineUnary(TensorElemwiseUnary)

       self._source = source


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._source, other._source)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._source.attach(data, inds))


   def operator(self):

       return self._source.operator()




# --- Tensor unary elementwise operator ------------------------------------- #  

class TensorElemwiseUnary:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _apply(self, fun, *args, **kwargs):

       data = fun(self._data, *args, **kwargs)

       return core.TensorGen(data, self._inds)


   # --- Value methods --- #

   def put(self, pos, vals, accumulate=False):

       return self._apply(ar.put, pos, vals, accumulate=accumulate)


   def clip(self, minval, maxval, **opts):

       return self._apply(ar.clip, minval, maxval, **opts)


   def flip(self, inds=None):

       if inds is None:
          return self._apply(ar.flip)

       return self._apply(ar.flip, self._inds.axes(*self._inds.map(*inds)))


   def cumsum(self, ind=None, dtype=None, **opts):

       if ind is None:

          data = ar.cumsum(self._data, dtype=dtype, **opts)
          data = ar.reshape(data, self._inds.shape)

          return core.TensorGen(data, self._inds)

       return self._apply(
          ar.cumsum, self._inds.axes(ind)[0], dtype=dtype, **opts
       )


   # --- Element access --- #

   def __getitem__(self, pos):

       return core.TensorGen(self._data[pos], Indices())


   # --- Extracting info --- #

   def iscomplex(self):

       return ar.iscomplex(self._data)


   # --- Data type methods --- #

   def astype(self, dtype):

       return self._apply(ar.astype, dtype=dtype)


   # --- Standard math --- #

   def floor(self):

       return self._apply(ar.floor)


   def neg(self):

       return self._apply(ar.neg)

   
   def sign(self):

       return self._apply(ar.sign)

 
   def conj(self):

       return self._apply(ar.conj)


   def real(self):

       return self._apply(ar.real)


   def imag(self):

       return self._apply(ar.imag)


   def absolute(self):

       return self._apply(ar.absolute)


   def sqrt(self):

       return self._apply(ar.sqrt)


   def log(self):

       return self._apply(ar.log)


   def exp(self):

       return self._apply(ar.exp)


   def sin(self):

       return self._apply(ar.sin)


   def cos(self):

       return self._apply(ar.cos)


   def tan(self):

       return self._apply(ar.tan)


   def arcsin(self):

       return self._apply(ar.arcsin)


   def arccos(self):

       return self._apply(ar.arccos)


   def arctan(self):

       return self._apply(ar.arctan)


   def sinh(self):

       return self._apply(ar.sinh)


   def cosh(self):

       return self._apply(ar.cosh)


   def tanh(self):

       return self._apply(ar.tanh)


   def arcsinh(self):

       return self._apply(ar.arcsinh)


   def arccosh(self):

       return self._apply(ar.arccosh)


   def arctanh(self):

       return self._apply(ar.arctanh)


   # --- Linear algebra --- #

   def expm(self):

       return self._apply(ar.expm)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorElemwiseUnary methods      ###
###                                                                         ###
###############################################################################


# --- Helper: unary typecast unary ------------------------------------------ #

def typecast_unary(fun):

    @functools.wraps(fun)
    def wrap(x, *args, **kwargs):

        try:
            return fun(x, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            return fun(core.astensor(x), *args, **kwargs)
         
    return wrap




# --- Value methods --------------------------------------------------------- #

@ad.nondifferentiable
def put(x, pos, vals, accumulate=False):

    op = tensor_elemwise_unary(x)

    return op.put(pos, vals, accumulate=accumulate)


@ad.differentiable
def clip(x, minval, maxval, **opts):

    op = tensor_elemwise_unary(x)

    return op.clip(minval, maxval, **opts)  


@ad.differentiable
def flip(x, inds=None):

    op = tensor_elemwise_unary(x)

    return op.flip(inds)


@ad.differentiable
def cumsum(x, ind=None, dtype=None, **opts):

    op = tensor_elemwise_unary(x)

    return op.cumsum(ind, dtype=dtype, **opts)




# --- Element access -------------------------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    op = tensor_elemwise_unary(x)

    return op[pos]




# --- Extracting info ------------------------------------------------------- #

@ad.nondifferentiable
@typecast_unary
def iscomplex(x):

    op = tensor_elemwise_unary(x)

    return op.iscomplex()




# --- Data type methods ----------------------------------------------------- #

@ad.differentiable
@typecast_unary
def astype(x, dtype):

    op = tensor_elemwise_unary(x)

    return op.astype(dtype)




# --- Standard math --------------------------------------------------------- #

@ad.nondifferentiable
@typecast_unary
def floor(x):

    op = tensor_elemwise_unary(x)
    return op.floor()


@ad.differentiable
@typecast_unary
def neg(x):

    op = tensor_elemwise_unary(x)
    return op.neg()

   
@ad.nondifferentiable
@typecast_unary
def sign(x):

    op = tensor_elemwise_unary(x)
    return op.sign()


@ad.differentiable 
@typecast_unary
def conj(x):

    op = tensor_elemwise_unary(x)
    return op.conj()


@ad.differentiable
@typecast_unary
def real(x):

    op = tensor_elemwise_unary(x)
    return op.real()


@ad.differentiable
def imag(x):

    op = tensor_elemwise_unary(x)
    return op.imag()


@ad.differentiable
@typecast_unary
def absolute(x):

    op = tensor_elemwise_unary(x)
    return op.absolute()


@ad.differentiable
def sqrt(x):

    op = tensor_elemwise_unary(x)
    return op.sqrt()


@ad.differentiable
@typecast_unary
def log(x):

    op = tensor_elemwise_unary(x)
    return op.log()


@ad.differentiable
def exp(x):

    op = tensor_elemwise_unary(x)
    return op.exp()


@ad.differentiable
@typecast_unary
def sin(x):

    op = tensor_elemwise_unary(x)
    return op.sin()


@ad.differentiable
@typecast_unary
def cos(x):

    op = tensor_elemwise_unary(x)
    return op.cos()


@ad.differentiable
@typecast_unary
def tan(x):

    op = tensor_elemwise_unary(x)
    return op.tan()


@ad.differentiable
@typecast_unary
def arcsin(x):

    op = tensor_elemwise_unary(x)
    return op.arcsin()


@ad.differentiable
@typecast_unary
def arccos(x):

    op = tensor_elemwise_unary(x)
    return op.arccos()


@ad.differentiable
@typecast_unary
def arctan(x):

    op = tensor_elemwise_unary(x)
    return op.arctan()


@ad.differentiable
@typecast_unary
def sinh(x):

    op = tensor_elemwise_unary(x)
    return op.sinh()


@ad.differentiable
@typecast_unary
def cosh(x):

    op = tensor_elemwise_unary(x)
    return op.cosh()


@ad.differentiable
@typecast_unary
def tanh(x):

    op = tensor_elemwise_unary(x)
    return op.tanh()


@ad.differentiable
@typecast_unary
def arcsinh(x):

    op = tensor_elemwise_unary(x)
    return op.arcsinh()


@ad.differentiable
@typecast_unary
def arccosh(x):

    op = tensor_elemwise_unary(x)
    return op.arccosh()


@ad.differentiable
@typecast_unary
def arctanh(x):

    op = tensor_elemwise_unary(x)
    return op.arctanh()




# --- Linear algebra -------------------------------------------------------- #

@ad.differentiable
@typecast_unary
def expm(x): # TODO: not part of the API because vjp/jvp are not available yet! 

    op = tensor_elemwise_unary(x)
    return op.expm()




