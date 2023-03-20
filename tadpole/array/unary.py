#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util     as util
import tadpole.backends as backends

import tadpole.array.core   as core
import tadpole.array.binary as binary
import tadpole.array.nary   as nary

from tadpole.array.core import ArrayLike




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Decorator to adjust input axis ---------------------------------------- #

def adjust_axis(fun):

    def wrap(x, axis=None, *args, **kwargs):

        if isinstance(axis, (list, tuple, util.TupleLike)):

           if len(axis) == 0:
              axis = None

           if len(axis) == 1:
              axis, = axis

        return fun(x, axis, *args, **kwargs)

    return wrap




# --- Type cast for unary functions ----------------------------------------- #

def typecast(fun):

    def wrap(x, *args, **kwargs):

        try:
            return fun(x, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            return fun(asarray(x), *args, **kwargs)
         
    return wrap




###############################################################################
###                                                                         ###
###  Definition of Unary Array (supports unary operations)                  ###
###                                                                         ###
###############################################################################


# --- Array factory --------------------------------------------------------- #

def asarray(data, **opts):

    if isinstance(data, ArrayLike):
       return data

    backend = backends.get_from(opts)                            
    data    = backend.asarray(data, **opts)

    return Array(backend, data)




# --- Unary Array ----------------------------------------------------------- #

class Array(ArrayLike):

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
          self._backend, other._backend, msg=f"{type(self).__name__}.__or__"
       )

       if isinstance(other, self.__class__):
          return binary.Array(backend, self._data, other._data)

       return nary.Array(backend, self._data, *other._datas)


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


   @property
   def dtype(self):

       return str(self._backend.dtype(self._data))


   @property
   def iscomplex(self):

       return self._backend.iscomplex(self._data)


   # --- Element access --- #

   def __getitem__(self, idx):

       return self._data[idx]


   # --- Shape methods --- #

   @property
   def size(self):

       return self._backend.size(self._data)


   @property
   def ndim(self):

       return self._backend.ndim(self._data)


   @property
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


   @adjust_axis
   def sumover(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   @adjust_axis
   def cumsum(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   # --- Value methods --- #

   def item(self, *idx):

       return self._backend.item(self._data, *idx)


   @adjust_axis
   def allof(self, axis=None, **opts):

       data = self._backend.all(self._data, axis, **opts)

       return self.new(data) 


   @adjust_axis
   def anyof(self, axis=None, **opts):

       data = self._backend.any(self._data, axis, **opts)

       return self.new(data) 


   @adjust_axis
   def amax(self, axis=None, **opts):

       return self._backend.max(self._data, axis, **opts)


   @adjust_axis
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


   @adjust_axis
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
###  Standalone functions corresponding to Unary Array methods              ###
###                                                                         ###
###############################################################################


# --- Core methods ---------------------------------------------------------- #

def copy(x, **opts):

    return x.copy(**opts)



# --- Data type methods ----------------------------------------------------- #

def astype(x, **opts):

    return x.astype(**opts)


def dtype(x):

    return x.dtype


@typecast
def iscomplex(x):

    return x.iscomplex




# --- Shape methods --------------------------------------------------------- #

def size(x):

    return x.size


def ndim(x):

    return x.ndim


def shape(x):

    return x.shape


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

@typecast
def conj(x):

    return x.conj()


@typecast
def real(x):

    return x.real()


@typecast
def imag(x):

    return x.imag()
       

@typecast
def sqrt(x):

    return x.sqrt()


@typecast
def log(x): 

    return x.log()


@typecast
def exp(x): 

    return x.exp()


@typecast
def neg(x):

    return x.neg()


@typecast
def sin(x):

    return x.sin()


@typecast
def cos(x):

    return x.cos()


@typecast
def tan(x):

    return x.tan()


@typecast
def arcsin(x):

    return x.arcsin()


@typecast
def arccos(x):

    return x.arccos()


@typecast
def arctan(x):

    return x.arctan()


@typecast
def sinh(x):

    return x.sinh()


@typecast
def cosh(x):

    return x.cosh()


@typecast
def tanh(x):

    return x.tanh()


@typecast
def arcsinh(x):

    return x.arcsinh()


@typecast
def arccosh(x):

    return x.arccosh()


@typecast
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




