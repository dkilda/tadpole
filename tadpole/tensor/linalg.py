#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core       as core
import tadpole.tensor.truncation as truncation


from tadpole.tensor.types import (
   Tensor,
   Grad,
   Pluggable,
   Engine,
   CutoffMode,
   ErrorMode,
   Trunc,
)


from tadpole.tensor.engine import (
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.truncation import (
   TruncNull,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)





class Matrix(Tensor, Grad, Pluggable):

   # --- Construction --- #

   def __init__(self, data, inds, indmap):

       self._data = data
       self._inds = inds
      
       self._indmap = indmap


   # --- Private helpers --- #

   @property
   def _lind(self):
       return self._inds[0]

   @property
   def _rind(self):
       return self._inds[1]


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


   def tonull(self):

       return NullGrad(self.space())


   # --- Basic functionality --- #

   def copy(self, virtual=False, **opts):

       data = self._data if virtual else self._data.copy(**opts)

       return self.__class__(data, self._inds, self._indmap)


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return MatrixSpace(self._data.space(), self._inds) 


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
          log.val(self._indmap, other._indmap)
 
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


   def __mod__(self, other):

       return binary.mod(self, other)


   def __floordiv__(self, other):

       return binary.floordiv(self, other)


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


   def __rmod__(self, other):

       return binary.mod(other, self)


   def __rfloordiv__(self, other):

       return binary.floordiv(other, self)


   def __rpow__(self, other):

       return binary.power(other, self)


   def __rmatmul__(self, other):

       return contraction.contract(other, self)


   # --- Matrix-specific methods --- # 

   @property
   def T(self):
     
       return 
       

   @property
   def H(self):
     
       return 


   @property
   def C(self):

       return self.__class__()

   
       























































