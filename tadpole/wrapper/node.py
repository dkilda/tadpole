#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util  as util
import tadpole.array as ar

import tadpole.autodiff.node      as an
import tadpole.wrapper.operations as op




# --- Node-Array wrapper class----------------------------------------------- #

class Node(an.Node, ar.types.ArrayLike): 


   # --- Array methods: gradient accumulation --- #

   def addto(self, other):

       return op.addgrads(self, other)


   # --- Array methods: basic functionality --- #

   def copy(self, **opts):

       return op.copy(self, **opts)

 
   def todense(self):

       return op.todense(self)


   def withdata(self, data):

       return op.withdata(self, data)


   def space(self):
 
       return op.space(self)


   def item(self, *idx):

       return op.item(self, *idx)


   # --- Array methods: array properties --- #

   @property
   def dtype(self):
       return op.dtype(self)  

   @property 
   def size(self):
       return op.size(self) 

   @property 
   def ndim(self):
       return op.ndim(self) 

   @property
   def shape(self):
       return op.shape(self) 


   # --- Array methods: comparisons --- #

   def allequal(self, other):

       return op.allequal(self, other)


   def allclose(self, other, **opts):

       return op.allequal(self, other)


   # --- Array methods: arithmetics and element access --- # 

   def __getitem__(self, idx):

       return op.getitem(self, idx) 


   def __neg__(self):

       return op.neg(self)


   def __add__(self, other):

       return op.add(self, other)


   def __sub__(self, other):

       return op.sub(self, other)


   def __mul__(self, other):

       return op.mul(self, other)


   def __radd__(self, other):

       return self.__add__(other)

 
   def __rsub__(self, other):

       return -self.__sub__(other)


   def __rmul__(self, other):

       return self.__mul__(other)


an.register(ar.core.Array,    Node)
an.register(ar.grad.ZeroGrad, Node)





