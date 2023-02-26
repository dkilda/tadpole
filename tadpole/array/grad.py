#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.operations as ops
import tadpole.array.array      as array
import tadpole.array.util       as util




###############################################################################
###                                                                         ###
###  A general framework for array gradients                                ###
###                                                                         ###
###############################################################################


# --- Sparse gradient class ------------------------------------------------- #

class SparseGrad(array.ArrayLike):

   def __init__(self, space, idxs, vals):

       self._space = space
       self._idxs  = idxs
       self._vals  = vals

 
   @property
   def _array(self):

       return self._todense()


   def todense(self):

       return ops.put(self._space.zeros(), idxs, self._vals)


   def copy(self):

       return self._array.copy()


   def space(self):

       return self._space 


   def pluginto(self, funcall):

       return self._array.attach(funcall)


   def __getitem__(self, coords):

       return self._array[coords]


   @property
   def dtype(self):
       return self._space.dtype

   @property 
   def ndim(self):
       return self._space.ndim

   @property
   def shape(self):
       return self._space.shape


   def __neg__(self):

       return -self._array


   def __add__(self, other): 

       if other == 0:
          other = self._space.zeros()

       return ops.put(other, self._idxs, self._vals, accumulate=True)


   def __mul__(self, other):

       return self._array * other 


   def __radd__(self, other): 

       return self.__add__(other)  


   def __rmul__(self, other):

       return other * self._array   




