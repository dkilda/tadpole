#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.space           as sp
import tadpole.tensor.contraction     as contraction
import tadpole.tensor.elemwise_unary  as unary
import tadpole.tensor.elemwise_binary as binary


from tadpole.tensor.types import (
   Tensor, 
   Grad,
   Pluggable,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Special tensor types for gradients                                     ###
###                                                                         ###
###############################################################################


# --- Null gradient --------------------------------------------------------- #

class NullGrad(Tensor, Grad, Pluggable):

   # --- Construction --- #

   def __init__(self, space):

       self._space = space


   # --- Plugging into an engine --- #

   def pluginto(self, engine):

       return self.todense().pluginto(engine)


   # --- Gradient operations --- #

   def addto(self, other):

       if isinstance(other, self.__class__):
          return self

       return other.addto(self)


   def todense(self):

       return self.space().zeros()


   # --- Basic functionality --- #

   def copy(self):

       return self.__class__(self._space)


   def withdata(self, data):

       return self.space().fillwith(data) 


   def space(self):

       return self._space


   def item(self, *pos):

       return self.todense().item(*pos) 


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self.space().dtype

   @property 
   def size(self):
       return self.space().size

   @property 
   def ndim(self):
       return self.space().ndim

   @property 
   def shape(self):
       return self.space().shape


   # --- Comparisons --- #
       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       return bool(log)


   def __bool__(self):

       return False


   # --- Arithmetics and element access --- #

   def __getitem__(self, pos):

       return self.todense()[pos]


   def __neg__(self):

       return self.todense()


   def __add__(self, other): 

       return other


   def __sub__(self, other): 

       return -other 


   def __mul__(self, other):

       return self.todense()


   def __truediv__(self, other):

       return self.todense()


   def __pow__(self, other):

       return self.todense() 


   def __radd__(self, other): 

       return other  


   def __rsub__(self, other): 

       return other


   def __rmul__(self, other):

       return self.todense()


   def __rtruediv__(self, other):

       raise ValueError(
          f"{type(self).__name__}.__rtruediv__: "
          f"division by zero encountered!"
       )


   def __rpow__(self, other):

       if not isinstance(other, Tensor):
          other = astensor(other)

       return other.space().ones()




# --- Sparse gradient ------------------------------------------------------- #

class SparseGrad(Tensor, Grad, Pluggable):

   # --- Construction --- #

   def __init__(self, space, pos, vals):

       self._space = space
       self._pos   = pos
       self._vals  = vals


   # --- Plugging into an engine --- #

   def pluginto(self, engine):

       return self.todense().pluginto(engine)


   # --- Gradient operations --- #

   def addto(self, other):

       if not other:
          other = self.space().zeros()

       if isinstance(other, SparseGrad):
          other = other.todense()

       assert self.space() == other.space(), (
          f"{type(self).__name__}.addto: "
          f"gradient accumulation cannot be performed for tensors "
          f"with non-matching spaces {self.space()} != {other.space()}"
       )

       data = ar.put(other._data, self._pos, self._vals, accumulate=True)
       return other.withdata(data)


   def todense(self):

       zeros = self.space().zeros()
       op    = unary.tensor_elemwise_unary(zeros)

       return op.put(self._pos, self._vals)

       
   # --- Basic functionality --- #

   def copy(self):

       return self.__class__(self._space, self._pos, self._vals)


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return self._space


   def item(self, *pos):

       return self.todense().item(*pos) 


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self._space.dtype 

   @property 
   def size(self):
       return self._space.size

   @property 
   def ndim(self):
       return self._space.ndim 

   @property 
   def shape(self):
       return self._space.shape


   # --- Comparisons --- #
       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._space, other._space)
          log.val(self._pos,   other._pos)

       if bool(log):
          for i in range(len(self._pos)):
              log.val(self._vals[i], other._vals[i])  

       return bool(log)


   # --- Arithmetics and element access --- # 

   def __getitem__(self, pos):

       return self.todense()[pos]


   def __neg__(self):

       return -self.todense()


   def __add__(self, other): 

       return self.todense() + other


   def __sub__(self, other): 

       return self.todense() - other   


   def __mul__(self, other):

       return self.todense() * other 


   def __truediv__(self, other):

       return self.todense() / other 


   def __pow__(self, other):

       return self.todense() ** other 


   def __radd__(self, other): 

       return other + self.todense() 


   def __rsub__(self, other): 

       return other - self.todense() 


   def __rmul__(self, other):

       return other * self.todense()


   def __rtruediv__(self, other):

       return other / self.todense()


   def __rpow__(self, other):

       return other ** self.todense() 




###############################################################################
###                                                                         ###
###  General tensor                                                         ###
###                                                                         ###
###############################################################################


# --- Tensor factories ------------------------------------------------------ #

def astensor(data, inds=None, **opts):

    if isinstance(data, Tensor):
       return data

    if isinstance(data, ar.Array):
       return TensorGen(data, inds)

    data = ar.asarray(data, **opts)
    return TensorGen(data, inds)




# --- General tensor -------------------------------------------------------- #

class TensorGen(Tensor, Grad, Pluggable):

   # --- Construction --- #

   def __init__(self, data, inds=None):

       if inds is None:
          inds = Indices()

       if not isinstance(inds, Indices):
          inds = Indices(*inds)

       if data.shape != inds.shape:
          raise ValueError((
             f"{type(self).__name__}: "
             f"data and indices must have matching shapes, "
             f"but data shape {data.shape} != index shape {inds.shape}"
          ))

       self._data = data
       self._inds = inds


   # --- Plugging into an engine --- #

   def pluginto(self, engine):

       return engine.attach(self._data, self._inds)


   # --- Gradient operations --- #

   def addto(self, other):

       if not other:
          return self

       if isinstance(other, SparseGrad):
          return other.addto(self)

       assert self._inds == other._inds, (
          f"{type(self).__name__}.addto: "
          f"gradient accumulation cannot be performed for tensors "
          f"with non-matching indices {self._inds} != {other._inds}"
       )

       data = ar.add(self._data, other._data)
       return other.withdata(data)


   def todense(self):

       return self


   # --- Basic functionality --- #

   def copy(self, virtual=False, **opts):

       data = self._data if virtual else self._data.copy(**opts)

       return self.__class__(data, self._inds)


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return sp.TensorSpace(self._data.space(), self._inds) 


   def item(self, *pos): 

       return self._data.item(*pos)


   # --- Tensor properties --- #

   @property
   def dtype(self):
       return self._data.dtype 

   @property 
   def size(self):
       return self._data.size

   @property 
   def ndim(self):
       return self._data.ndim  

   @property
   def shape(self):
       return self._data.shape 


   # --- Comparisons --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._inds, other._inds)
 
       if bool(log):
          log.val(self._data, other._data) 

       return bool(log)


   # --- Arithmetics and element access --- # 

   def __getitem__(self, pos):

       return unary.getitem(self, pos)


   def __neg__(self):

       return unary.neg(self)


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


   def __matmul__(self, other):

       return contraction.contract(self, other)


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


   def __rmatmul__(self, other):

       return contraction.contract(other, self)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Tensor methods                   ###
###                                                                         ###
###############################################################################


# --- Basic functionality --------------------------------------------------- #

@ad.nondifferentiable
def copy(x, **opts):

    return x.copy(**opts)


@ad.nondifferentiable
def todense(x):

    return x.todense()


@ad.nondifferentiable
def withdata(x, data):

    return x.withdata(data)


@ad.nondifferentiable
def space(x):

    return x.space()


@ad.nondifferentiable
def item(x, *pos):

    return x.item(*pos)




# --- Tensor properties ----------------------------------------------------- #

@ad.nondifferentiable
def dtype(x):

    return x.dtype


@ad.nondifferentiable
def size(x):

    return x.size


@ad.nondifferentiable
def ndim(x):

    return x.ndim


@ad.nondifferentiable
def shape(x):

    return x.shape




