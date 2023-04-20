#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid




###############################################################################
###                                                                         ###
###  General scalar and associated functions                                ###
###                                                                         ###
###############################################################################


# --- General scalar -------------------------------------------------------- #

class ScalarGen(util.Scalar, tn.Grad):

   # --- Construction --- #

   def __init__(self, source):

       self._source = source


   # --- Gradient operations --- #

   def tonull(self):

       return self.__class__(0)


   def todense(self):

       return self


   def addto(self, other):

       if not other:
          return self

       return self.__class__(self._source + other._source)


   # --- Conversion to standard numeric types --- #

   def __bool__(self):

       return bool(self._source)


   def __int__(self):

       return int(self._source)


   def __long__(self):

       return long(self._source)


   def __float__(self):

       return float(self._source)


   def __complex__(self):

       return complex(self._source)

 
   # --- Comparison and hashing --- #

   def _cmp(self, other, op):

       if type(self) == type(other):
          return op(self._source, other._source)

       return op(self._source, other)


   def __eq__(self, other):

       return self._cmp(other, lambda x, y: x == y)


   def __ne__(self, other):

       return self._cmp(other, lambda x, y: x != y)


   def __gt__(self, other):

       return self._cmp(other, lambda x, y: x > y)


   def __lt__(self, other):

       return self._cmp(other, lambda x, y: x < y)


   def __ge__(self, other):

       return self._cmp(other, lambda x, y: x >= y)


   def __le__(self, other):

       return self._cmp(other, lambda x, y: x <= y)


   def __hash__(self):

       return hash(self._source)


   # --- Arithmetics --- #

   def _unary(self, op):

       return self.__class__(op(self._source))


   def _binary(self, other, op):

       if type(self) == type(other):
          return self.__class__(op(self._source, other._source))

       return self.__class__(op(self._source, other))


   def __neg__(self):

       return self._unary(tn.neg)  


   def __add__(self, other): 

       return self._binary(other, lambda x, y: tn.add(x, y))


   def __sub__(self, other): 

       return self._binary(other, lambda x, y: tn.sub(x, y)) 


   def __mul__(self, other):

       return self._binary(other, lambda x, y: tn.mul(x, y))


   def __truediv__(self, other):

       return self._binary(other, lambda x, y: tn.div(x, y)) 


   def __mod__(self, other):

       return self._binary(other, lambda x, y: tn.mod(x, y)) 


   def __pow__(self, other):

       return self._binary(other, lambda x, y: tn.pow(x, y))  


   def __radd__(self, other): 

       return self._binary(other, lambda x, y: tn.add(y, x)) 


   def __rsub__(self, other): 

       return self._binary(other, lambda x, y: tn.sub(y, x)) 


   def __rmul__(self, other):

       return self._binary(other, lambda x, y: tn.mul(y, x)) 


   def __rtruediv__(self, other):

       return self._binary(other, lambda x, y: tn.div(y, x)) 


   def __rmod__(self, other):

       return self._binary(other, lambda x, y: tn.mod(y, x)) 


   def __rpow__(self, other):

       return self._binary(other, lambda x, y: tn.pow(y, x)) 




