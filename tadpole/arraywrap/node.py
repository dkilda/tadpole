#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util  as util
import tadpole.array as td

import tadpole.autodiff.node        as an
import tadpole.arraywrap.operations as op

from tadpole.array import ArrayLike, Array
from tadpole.array import ZeroGrad, SparseGrad




# --- Node-Array wrapper class----------------------------------------------- #

class Node(an.Node, ArrayLike): 

   # --- Array methods: gradient accumulation --- #

   def addto(self, other):

       if not other:
          other = ZeroGrad()

       return td.addgrads(self, other)


   # --- Array methods: basic functionality --- #

   def copy(self, **opts):

       return td.copy(self, **opts)

 
   def todense(self):

       return td.todense(self)


   def withdata(self, data):

       return td.withdata(self, data)


   def space(self):
 
       return td.space(self)


   def item(self, *idx):

       return td.item(self, *idx)


   # --- Array methods: array properties --- #

   @property
   def dtype(self):
       return td.dtype(self)  

   @property 
   def size(self):
       return td.size(self) 

   @property 
   def ndim(self):
       return td.ndim(self) 

   @property
   def shape(self):
       return td.shape(self) 


   # --- Array methods: comparisons --- #

   def allequal(self, other):

       return td.allequal(self, other)


   def allclose(self, other, **opts):

       return td.allequal(self, other)


   # --- Array methods: arithmetics and element access --- # 

   def __getitem__(self, idx):

       return td.getitem(self, idx) 


   def __neg__(self):

       return td.neg(self)


   def __add__(self, other):

       return td.add(self, other)


   def __sub__(self, other):

       return td.sub(self, other)


   def __mul__(self, other):

       return td.mul(self, other)


   def __truediv__(self, other):

       return td.div(self, other)


   def __pow__(self, other):

       return td.power(self, other)


   def __radd__(self, other):

       return td.add(other, self)

 
   def __rsub__(self, other):

       return td.sub(other, self)


   def __rmul__(self, other):

       return td.mul(other, self)


   def __rtruediv__(self, other):

       return td.div(other, self)


   def __rpow__(self, other):

       return td.power(other, self)




an.register(Array,      Node)
an.register(SparseGrad, Node)
an.register(ZeroGrad,   Node)





