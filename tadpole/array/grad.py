#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.operations as op

from tadpole.array.core import ArrayLike




###############################################################################
###                                                                         ###
###  A general framework for array gradients                                ###
###                                                                         ###
###############################################################################


# --- Sparse gradient class ------------------------------------------------- #

class SparseGrad(ArrayLike):

   def __init__(self, space, idxs, vals):

       self._space = space
       self._idxs  = idxs
       self._vals  = vals


   @property
   def _array(self):

       return self.todense()


   def todense(self):

       return op.put(self._space.zeros(), self._idxs, self._vals)


   def copy(self, **opts):

       return self._array.copy(**opts)


   def space(self):

       return self._space 


   def pluginto(self, funcall):

       return self._array.pluginto(funcall)


   @property
   def dtype(self):
       return self._space.dtype

   @property 
   def ndim(self):
       return self._space.ndim

   @property
   def shape(self):
       return self._space.shape


   def allclose(self, other, **opts):

       log = util.LogicalChain()
       log.typ(self, other)
       log.val(self._space, other._space)
       log.val(self._idxs,  other._idxs)

       if bool(log):
          return util.allclose(self._vals, other._vals, **opts)   

       return False

       
   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._space, other._space)
          log.val(self._idxs,  other._idxs)

       if bool(log):
          return util.allequal(self._vals, other._vals)  

       return False


   def __getitem__(self, coords):

       return self._array[coords]


   def __neg__(self):

       return -self._array


   def __add__(self, other): 

       if other == 0:
          other = self._space.zeros()

       return op.put(other, self._idxs, self._vals, accumulate=True)


   def __mul__(self, other):

       return self._array * other 


   def __radd__(self, other): 

       return self.__add__(other)  


   def __rmul__(self, other):

       return other * self._array   




