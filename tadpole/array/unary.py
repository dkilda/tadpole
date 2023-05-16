#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util

import tadpole.array.backends as backends
import tadpole.array.types    as types
import tadpole.array.space    as space
import tadpole.array.binary   as binary
import tadpole.array.nary     as nary




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for unary functions ----------------------------------------- #

def typecast(fun):

    def wrap(x, *args, **kwargs):

        if isinstance(x, types.Array):
           return fun(x, *args, **kwargs)   

        return fun(asarray(x), *args, **kwargs)
         
    return wrap




###############################################################################
###                                                                         ###
###  Definition of Unary Array (supports unary operations)                  ###
###                                                                         ###
###############################################################################


# --- Array factory --------------------------------------------------------- #

def asarray(data, **opts):

    if isinstance(data, types.Array):
       return data

    backend = backends.get_from(opts)                            
    data    = backend.asarray(data, **opts)

    return Array(backend, data)




# --- Unary Array ----------------------------------------------------------- #

class Array(types.Array):

   # --- Construction --- #

   def __init__(self, backend, data):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"{type(self).__name__}: "
             f"backend must be an instance of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._data    = data


   # --- Array methods --- #

   def new(self, data):

       return asarray(data, backend=self._backend)


   def nary(self):

       return nary.Array(self._backend, *self._datas)


   def __or__(self, other):

       backend = backends.common(
          self._backend, 
          other._backend, 
          msg=f"{type(self).__name__}.__or__"
       )

       if isinstance(other, self.__class__):
          return binary.Array(backend, self._data, other._data)

       return nary.Array(backend, self._data, *other._datas)


   @property
   def _datas(self):

       return (self._data, )


   # --- Comparison --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
 
       if bool(log):
          return self._backend.allequal(self._data, other._data) 

       return False


   # --- Element access and arithmetics --- #

   def __getitem__(self, idx):

       return self.new(self._data[idx])


   def __add__(self, other):

       return binary.add(self, other)


   def __sub__(self, other):

       return binary.sub(self, other)


   def __mul__(self, other):

       return binary.mul(self, other)


   def __truediv__(self, other):

       return binary.div(self, other)


   def __pow__(self, other):

       return binary.power(self, other)


   def __radd__(self, other):

       return binary.add(other, self)


   def __rsub__(self, other):

       return binary.sub(other, self)


   def __rmul__(self, other):

       return binary.mul(other, self)


   def __rtruediv__(self, other):

       return binary.div(other, self)


   def __rpow__(self, other):

       return binary.power(other, self)


   # --- Core methods --- #

   def copy(self, **opts):

       data = self._backend.copy(self._data, **opts)

       return self.new(data) 


   def space(self):

       return space.ArraySpace(self._backend, self.shape, self.dtype)


   # --- Data type methods --- #

   def astype(self, **opts):

       data = self._backend.astype(self._data, **opts)

       return self.new(data) 


   @property
   def dtype(self):

       return str(self._backend.dtype(self._data))


   @property
   def iscomplex(self):

       return self._backend.iscomplex(self._data)


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


   def sumover(self, axis=None, dtype=None, **opts):

       data = self._backend.sumover(self._data, axis, dtype, **opts) 

       return self.new(data) 


   def cumsum(self, axis=None, dtype=None, **opts):

       data = self._backend.cumsum(self._data, axis, dtype, **opts) 

       return self.new(data) 


   def broadcast_to(self, shape):

       data = self._backend.broadcast_to(self._data, shape)

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

       data = self._backend.max(self._data, axis, **opts)

       return self.new(data)


   def amin(self, axis=None, **opts):

       data = self._backend.min(self._data, axis, **opts)

       return self.new(data)


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

       data = self._backend.clip(self._data, minval, maxval, **opts)

       return self.new(data) 


   def count_nonzero(self, axis=None, **opts):

       data = self._backend.count_nonzero(self._data, axis, **opts)

       return self.new(data) 


   def put(self, idxs, vals, accumulate=False):

       return nary.put(self, idxs, vals, accumulate=accumulate)


   def argsort(self, axis=None, **opts):

       data = self._backend.argsort(self._data, axis, **opts)

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


   def floor(self):

       data = self._backend.floor(self._data)

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


   # --- Linear algebra: other methods --- #

   def expm(self):

       data = self._backend.expm(self._data)

       return self.new(data)


   def norm(self, axis=None, order=None, **opts):

       data = self._backend.norm(self._data, axis, order, **opts)

       return self.new(data) 


   def trace(self, **opts):  

       data = self._backend.trace(self._data, **opts)

       return self.new(data)    


   def det(self):  

       data = self._backend.det(self._data)

       return self.new(data)    


   def inv(self):  

       data = self._backend.inv(self._data)

       return self.new(data)  


   def tril(self, **opts):  

       data = self._backend.tril(self._data, **opts)

       return self.new(data)  


   def triu(self, **opts):  

       data = self._backend.triu(self._data, **opts)

       return self.new(data)


   def diag(self, **opts):

       data = self._backend.diag(self._data, **opts)

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


def broadcast_to(x, shape):

    return x.broadcast_to(shape)




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
def floor(x):

    return x.floor()


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


def eig(x):

    return x.eig()


def eigh(x):

    return x.eigh()


def qr(x):

    return x.qr()


def lq(x):

    if iscomplex(x):
       x = conj(x)

    Q, R = qr(transpose(x, (1,0)))
    
    L = transpose(R, (1,0))
    Q = transpose(Q, (1,0))

    if iscomplex(x):
       L = conj(L) 
       Q = conj(Q) 

    return L, Q




# --- Linear algebra: other methods ----------------------------------------- #

def expm(x):

    return x.expm()


def norm(x, axis=None, order=None, **opts):

    return x.norm(axis, order, **opts)


def trace(x, **opts):

    return x.trace(**opts)


def det(x):

    return x.det()


def inv(x):  

    return x.inv()


def tril(x, **opts):  

    return x.tril(**opts)


def triu(x, **opts):  

    return x.triu(**opts)


def diag(x, **opts):

    return x.diag(**opts)




