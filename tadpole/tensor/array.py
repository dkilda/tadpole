#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import operator
from functools import reduce

import tadpole.util     as util
import tadpole.autodiff as ad



###############################################################################
###                                                                         ###
###  General ArrayLike interface for OneArray/TwoArray/NArray/etc objects   ###
###                                                                         ###
###############################################################################


# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def new(self, data):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass




###############################################################################
###                                                                         ###
###  Definition of Void Array (supports array creation)                     ###
###                                                                         ###
###############################################################################


# --- Void-Array ------------------------------------------------------------ #

class VoidArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"VoidArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend


   # --- Arraylike methods --- #

   def new(self, data):

       return OneArray(self._backend, data)


   def __or__(self, other):

       return other 


   # --- Array creation methods --- #

   def asarray(self, array, **opts):

       if isinstance(array, ArrayLike):
          return array

       data = self._backend.asarray(array, **opts)

       return self.new(data)


   def zeros(self, shape, **opts):

       data = self._backend.zeros(shape, **opts)

       return self.new(data)


   def ones(self, shape, **opts):

       data = self._backend.ones(shape, **opts)     

       return self.new(data)     


   def unit(self, shape, idx, **opts):

       data = self._backend.unit(shape, idx, **opts)

       return self.new(data)
       

   def eye(self, N, M=None, **opts):

       data = self._backend.eye(N, M, **opts)

       return self.new(data)
       

   def rand(self, shape, **opts):

       data = self._backend.rand(shape, **opts)

       return self.new(data)
       

   def randn(self, shape, **opts):

       data = self._backend.randn(shape, **opts)

       return self.new(data)
       

   def randuniform(self, shape, boundaries, **opts):

       data = self._backend.randuniform(shape, boundaries, **opts)

       return self.new(data)

   


###############################################################################
###                                                                         ###
###  Standalone functions corresponding to VoidArray methods                ###
###                                                                         ###
###############################################################################


# --- Array creation methods ------------------------------------------------ #

def asarray(x, array, **opts):

    return x.asarray(array, **opts)


def zeros(x, shape, **opts):

    return x.zeros(shape, **opts)


def ones(x, shape, **opts):

    return x.ones(shape, **opts)


def unit(x, shape, idx, **opts):

    return x.unit(shape, idx, **opts)


def eye(x, N, M=None, **opts):

    return x.eye(N, M=None, **opts)


def rand(x, shape, **opts):

    return x.rand(shape, **opts)


def randn(x, shape, **opts):

    return x.randn(shape, **opts)
       

def randuniform(x, shape, boundaries, **opts):

    return x.randn(shape, boundaries, **opts)




###############################################################################
###                                                                         ###
###  Definition of Single Array (supports unary operations)                 ###
###                                                                         ###
###############################################################################


# --- One-Array ------------------------------------------------------------- #

class OneArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, data):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"OneArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._data    = data


   # --- Arraylike methods --- #

   def new(self, data):

       return self.__class__(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, other._backend, msg="OneArray.__or__"
       )

       if isinstance(other, OneArray):
          return TwoArray(backend, self._data, other._data)

       return NArray(backend, self._data, *other._datas)


   @property
   def _datas(self):

       return (self._data, )


   # --- Core methods --- #

   def copy(self, **opts):

       data = self._backend.copy(self._data, **opts)

       return self.new(data) 


   # --- Data type methods --- #

   def astype(self, **opts):

       data = self._data.astype(**opts)

       return self.new(data) 


   def dtype(self):

       return str(self._backend.dtype(self._data))


   def iscomplex(self):

       return self._backend.iscomplex(self._data)


   # --- Shape methods --- #

   def size(self):

       return self._backend.size(self._data)


   def ndim(self):

       return self._backend.ndim(self._data)


   def shape(self):

       return self._backend.shape(self._data)


   def reshape(self, shape, **opts):

       data = self._backend.reshape(self._data, shape, **opts)

       return self.new(data) 


   def transpose(self, axes):

       data = self._backend.transpose(self._data, axes)

       return self.new(data) 


   def moveaxis(self, source, destination):

       data = self._backend.moveaxis(self._data, source, destination)

       return self.new(data) 


   def squeeze(self, axis=None):

       data = self._backend.squeeze(self._data, axis)

       return self.new(data) 


   def unsqueeze(self, axis):

       data = self._backend.unsqueeze(self._data, axis)

       return self.new(data) 


   def sumover(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   def cumsum(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   # --- Value methods --- #

   def item(self, *idx):

       return self._backend.item(self._data, *idx)


   def allof(self, axis=None, **opts):

       data = self._backend.all(self._data, axis, **opts)

       return self.new(data) 


   def anyof(self, axis=None, **opts):

       data = self._backend.any(self._data, axis, **opts)

       return self.new(data) 


   def amax(self, axis=None, **opts):

       return self._backend.max(self._data, axis, **opts)


   def amin(self, axis=None, **opts):

       return self._backend.min(self._data, axis, **opts)


   def sign(self, **opts):

       data = self._backend.sign(self._data, **opts)

       return self.new(data) 


   def absolute(self, **opts):

       data = self._backend.abs(self._data, **opts)

       return self.new(data) 


   def flip(self, axis=None):

       data = self._backend.flip(self._data, axis)

       return self.new(data) 


   def clip(self, minval, maxval, **opts):

       data = self._backend.clip(self._data, minval, maxval, **opts):

       return self.new(data) 


   def count_nonzero(self, axis=None, **opts):

       data = self._backend.count_nonzero(self._data, axis, **opts)

       return self.new(data) 


   def put(self, idxs, vals, accumulate=False):

       data = self._backend.put(
                 self._data, idxs, vals, accumulate=accumulate
              )

       return self.new(data) 


   def argsort(self, axis=None, **opts):

       data = self._backend.argsort(self._data, axis, **opts)

       return self.new(data)   


   def diag(self, **opts):

       data = self._backend.diag(self._data, **opts)

       return self.new(data)      


   # --- Standard math --- #

   def conj(self):

       data = self._backend.conj(self._data)

       return self.new(data) 


   def real(self):

       data = self._backend.real(self._data)

       return self.new(data) 


   def imag(self):

       data = self._backend.imag(self._data)

       return self.new(data) 
       

   def sqrt(self):

       data = self._backend.sqrt(self._data)

       return self.new(data) 


   def log(self): 

       data = self._backend.log(self._data)

       return self.new(data) 


   def exp(self): 

       data = self._backend.exp(self._data)

       return self.new(data) 


   def neg(self):

       data = self._backend.neg(self._data)

       return self.new(data) 


   def sin(self):

       data = self._backend.sin(self._data)
       
       return self.new(data) 


   def cos(self):

       data = self._backend.cos(self._data)

       return self.new(data) 


   def tan(self):

       data = self._backend.tan(self._data)

       return self.new(data) 


   def arcsin(self):

       data = self._backend.arcsin(self._data)
       
       return self.new(data) 


   def arccos(self):

       data = self._backend.arccos(self._data)

       return self.new(data) 


   def arctan(self):

       data = self._backend.arctan(self._data)

       return self.new(data) 


   def sinh(self):

       data = self._backend.sinh(self._data)

       return self.new(data) 


   def cosh(self):

       data = self._backend.cosh(self._data)

       return self.new(data) 


   def tanh(self):

       data = self._backend.tanh(self._data)

       return self.new(data) 


   def arcsinh(self):

       data = self._backend.arcsinh(self._data)

       return self.new(data) 


   def arccosh(self):

       data = self._backend.arccosh(self._data)

       return self.new(data) 


   def arctanh(self):

       data = self._backend.arctanh(self._data)

       return self.new(data) 


   # --- Linear algebra: decompositions --- #

   def svd(self):

       U, S, VH = self._backend.svd(self._data)

       return self.new(U), self.new(S), self.new(VH)


   def qr(self):

       Q, R = self._backend.qr(self._data)

       return self.new(Q), self.new(R)


   def eig(self):

       U, S, VH = self._backend.eig(self._data)

       return self.new(U), self.new(S), self.new(VH) 


   def eigh(self):

       U, S, VH = self._backend.eigh(self._data)

       return self.new(U), self.new(S), self.new(VH) 


   # --- Linear algebra: matrix exponential --- #

   def expm(self):

       data = self._backend.expm(self._data)

       return self.new(data)


   # --- Linear algebra: norm --- #

   def norm(self, order=None, axis=None, **opts):

       data = self._backend.norm(self._data, order, axis, **opts)

       return self.new(data)       





###############################################################################
###                                                                         ###
###  Standalone functions corresponding to OneArray methods                 ###
###                                                                         ###
###############################################################################


# --- Core methods ---------------------------------------------------------- #

def copy(x, **opts):

    return x.copy(**opts)



# --- Data type methods ----------------------------------------------------- #

def astype(x, **opts):

    return x.astype(**opts)


def dtype(x):

    return x.dtype()


def iscomplex(x):

    return x.iscomplex()




# --- Shape methods --------------------------------------------------------- #

def size(x):

    return x.size() 


def ndim(x):

    return x.ndim()


def shape(x):

    return x.shape()


def reshape(x, shape, **opts):

    return x.reshape(shape, **opts)


def transpose(x, axes):

    return x.transpose(axes)


def moveaxis(x, source, destination):

    return x.moveaxis(source, destination)


def squeeze(x, axis=None):

    return x.squeeze(axis)
    

def unsqueeze(x, axis):

    return x.unsqueeze(axis)


def sumover(x, axis=None, dtype=None, **opts):

    return x.sumover(axis, dtype, **opts)


def cumsum(x, axis=None, dtype=None, **opts):

    return x.cumsum(axis, dtype, **opts)




# --- Value methods --------------------------------------------------------- #

def item(x, *idx):

    return x.item(*idx)


def allof(x, axis=None, **opts):

    return x.allof(axis, **opts)


def anyof(x, axis=None, **opts):

    return x.anyof(axis, **opts) 


def amax(x, axis=None, **opts):

    return x.amax(axis, **opts)


def amin(x, axis=None, **opts):

    return x.amin(axis, **opts)  


def sign(x, **opts):

    return x.sign(**opts)  


def absolute(x, **opts):

    return x.absolute(**opts)


def flip(x, axis=None):

    return x.flip(axis)


def clip(x, minval, maxval, **opts):

    return x.clip(minval, maxval, **opts)


def count_nonzero(x, axis=None, **opts):

    return x.count_nonzero(axis, **opts)


def put(x, idxs, vals, accumulate=False):

    return x.put(idxs, vals, accumulate=accumulate)


def argsort(x, axis=None, **opts):

    return x.argsort(axis, **opts)


def diag(x, **opts):

    return x.diag(**opts)

    


# --- Standard math --------------------------------------------------------- #

def conj(x):

    return x.conj()


def real(x):

    return x.real()


def imag(x):

    return x.imag()
       

def sqrt(x):

    return x.sqrt()


def log(x): 

    return x.log()


def exp(x): 

    return x.exp()


def neg(x):

    return x.neg()


def sin(x):

    return x.sin()


def cos(x):

    return x.cos()


def tan(x):

    return x.tan()


def arcsin(x):

    return x.arcsin()


def arccos(x):

    return x.arccos()


def arctan(x):

    return x.arctan()


def sinh(x):

    return x.sinh()


def cosh(x):

    return x.cosh()


def tanh(x):

    return x.tanh()


def arcsinh(x):

    return x.arcsinh()


def arccosh(x):

    return x.arccosh()


def arctanh(x):

    return x.arctanh()




# --- Linear algebra: decompositions ---------------------------------------- #

def svd(x):

    return x.svd()


def qr(x):

    return x.qr()


def eig(x):

    return x.eig()


def eigh(x):

    return x.eigh()




# --- Linear algebra: matrix exponential ------------------------------------ #

def expm(x):

    return x.expm()




# --- Linear algebra: norm -------------------------------------------------- #

def norm(x, order=None, axis=None, **opts):

    return x.norm(order, axis, **opts)





###############################################################################
###                                                                         ###
###  Definition of Double Array (supports binary operations)                ###
###                                                                         ###
###############################################################################


# --- Two-Array ------------------------------------------------------------- #

class TwoArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, dataA, dataB):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"TwoArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._datas   = (dataA, dataB)


   # --- Arraylike methods --- #

   def new(self, data):

       return OneArray(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, other._backend, msg="TwoArray.__or__"
       )

       return NArray(backend, *self._datas, *other._datas)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return self._backend.allclose(*self._datas, **opts)  


   def allequal(self):

       return self._backend.allequal(*self._datas) 


   def isclose(self, **opts):

       data = self._backend.isclose(*self._datas, **opts)  

       return self.new(data)
 

   def isequal(self):

       data = self._backend.isequal(*self._datas) 
 
       return self.new(data)


   def notequal(self):

       data = self._backend.notequal(*self._datas)

       return self.new(data)


   def logical_and(self):

       data = self._backend.logical_and(*self._datas)

       return self.new(data)


   def logical_or(self):

       data = self._backend.logical_or(*self._datas)

       return self.new(data)


   # --- Elementwise binary algebra --- #

   def add(self):

       data = self._backend.add(*self._datas)

       return self.new(data)

       
   def sub(self):

       data = self._backend.sub(*self._datas)

       return self.new(data)      


   def mul(self):
       
       data = self._backend.mul(*self._datas)

       return self.new(data)


   def div(self):

       data = self._backend.div(*self._datas)

       return self.new(data)


   def power(self):

       data = self._backend.power(*self._datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def dot(self):

       data = self._backend.dot(*self._datas)

       return self.new(data)
       

   def kron(self):

       data = self._backend.kron(*self._datas)

       return self.new(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TwoArray methods                 ###
###                                                                         ###
###############################################################################


# --- Logical operations ---------------------------------------------------- #

def allclose(x, y, **opts):

    return (x | y).allclose(**opts) 


def allequal(x, y):

    return (x | y).allequal()


def isclose(x, y, **opts):

    return (x | y).isclose()


def isequal(x, y):

    return (x | y).isequal()


def notequal(x, y):

    return (x | y).notequal()


def logical_and(x, y):

    return (x | y).logical_and()


def logical_or(x, y):

    return (x | y).logical_or()




# --- Elementwise binary algebra -------------------------------------------- #

def add(x, y):

    return (x | y).add()


def sub(x, y):

    return (x | y).sub()


def mul(x, y):

    return (x | y).mul()

       
def div(x, y):

    return (x | y).div()


def power(x, y):

    return (x | y).power()




# --- Linear algebra: products ---------------------------------------------- #

def dot(x, y):

    return (x | y).dot()

       
def kron(x, y):

    return (x | y).kron()





###############################################################################
###                                                                         ###
###  Definition of N-tuple Array (supports nary operations)                 ###
###                                                                         ###
###############################################################################


# --- N-Array --------------------------------------------------------------- #

class NArray(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, *datas):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"NArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._datas   = datas


   # --- Arraylike methods --- #

   def new(self, data):

       return OneArray(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, other._backend, msg="NArray.__or__"
       )

       return NArray(backend, *self._datas, *other._datas)


   # --- Value methods --- #

   def where(self): 

       data = self._backend.where(*self.datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def einsum(self, equation, optimize=True):

       data = self._backend.einsum(equation, *self._datas, optimize=optimize)

       return self.new(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to NArray methods                   ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def where(condition, x, y): 

    return (condition | x | y).where()




# --- Linear algebra: products ---------------------------------------------- #

def einsum(equation, *xs, optimize=True):

    array = reduce(operator.or_, xs) 
            
    return array.einsum(equation, optimize=optimize)




