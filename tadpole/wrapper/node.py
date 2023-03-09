#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util  as util
import tadpole.array as ar

import tadpole.autodiff.node      as ad
import tadpole.wrapper.operations as td




# --- Node-Array wrapper class----------------------------------------------- #

class Node(ad.Node, ar.types.ArrayLike): 


   # --- Array methods: gradient accumulation --- #

   def addto(self, other):

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


   def __radd__(self, other):

       return self.__add__(other)

 
   def __rsub__(self, other):

       return -self.__sub__(other)


   def __rmul__(self, other):

       return self.__mul__(other)


ad.NodeScape.register(ar.core.Array,    Node)
ad.NodeScape.register(ar.grad.ZeroGrad, Node)





