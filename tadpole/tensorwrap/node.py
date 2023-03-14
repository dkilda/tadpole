#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util   as util
import tadpole.tensor as tn

import tadpole.autodiff.node         as an
import tadpole.tensorwrap.operations as op

from tadpole.tensor import Tensor
from tadpole.tensor import ZeroGrad, SparseGrad




# --- Node wrapper for TensorLike objects ----------------------------------- #

class Node(an.Node, tn.TensorLike): 

   # --- Tensor methods: gradient accumulation --- #

   def addto(self, other):

       if not other:
          other = ZeroGrad()

       return tn.addgrads(self, other)


   # --- Tensor methods: basic functionality --- #

   def copy(self, **opts):

       return tn.copy(self, **opts)

 
   def todense(self):

       return tn.todense(self)


   def withdata(self, data):

       return tn.withdata(self, data)


   def space(self):
 
       return tn.space(self)


   def item(self, *idx):

       return tn.item(self, *idx)


   # --- Tensor methods: tensor properties --- #

   @property
   def dtype(self):
       return tn.dtype(self)  

   @property 
   def size(self):
       return tn.size(self) 

   @property 
   def ndim(self):
       return tn.ndim(self) 

   @property
   def shape(self):
       return tn.shape(self) 


   # --- Tensor methods: comparisons --- #

   def allequal(self, other):

       return tn.allequal(self, other)


   def allclose(self, other, **opts):

       return tn.allequal(self, other)


   # --- Tensor methods: arithmetics and element access --- # 

   def __getitem__(self, idx):

       return tn.getitem(self, idx) 


   def __neg__(self):

       return tn.neg(self)


   def __add__(self, other):

       return tn.add(self, other)


   def __sub__(self, other):

       return tn.sub(self, other)


   def __mul__(self, other):

       return tn.mul(self, other)


   def __truediv__(self, other):

       return tn.div(self, other)


   def __pow__(self, other):

       return tn.power(self, other)


   def __radd__(self, other):

       return tn.add(other, self)

 
   def __rsub__(self, other):

       return tn.sub(other, self)


   def __rmul__(self, other):

       return tn.mul(other, self)


   def __rtruediv__(self, other):

       return tn.div(other, self)


   def __rpow__(self, other):

       return tn.power(other, self)




an.register(Tensor,     Node)
an.register(SparseGrad, Node)
an.register(ZeroGrad,   Node)





