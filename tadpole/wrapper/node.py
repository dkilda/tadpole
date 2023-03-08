#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.array.operations as op




# --- Node-Array wrapper class----------------------------------------------- #

class Node(ad.node.NodeLike, ar.types.ArrayLike):

   # --- Construction --- #

   def __init__(self, node):

       self._node = node


   # --- Equality, hashing, representation --- #

   def __repr__(self):

       return repr(self._node)


   def __hash__(self):

       return hash(self._node)


   def __eq__(self, other):

       if isinstance(other, self.__class__):
          return self._node == other._node

       return self._node == other


   # --- Node methods --- #

   def concat(self, concatenable):

       return self._node.concat(concatenable)


   def flow(self):

       return self._node.flow()


   def trace(self, traceable):

       return self._node.trace(traceable)


   def grads(self, grads):

       return self._node.grads(grads)


   # --- Array methods: basic functionality --- #

   """
   def copy(self, **opts):

       return self.

       source = self._source.copy(**opts)

       return self.__class__(source, self._layer, self._gate)


   def asarray(self, data):

       return self._source.asarray(data)
   """


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






































