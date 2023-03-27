#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Definition of tensor interfaces                                        ###
###                                                                         ###
###############################################################################


# --- Pluggable interface --------------------------------------------------- #

class Pluggable(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, funcall):
       pass




# --- TensorLike interface -------------------------------------------------- #

class TensorLike(abc.ABC):

   # --- Gradient accumulation --- #

   @abc.abstractmethod
   def addto(self, other):
       pass


   # --- Basic functionality --- #

   @abc.abstractmethod
   def copy(self, **opts):
       pass

   @abc.abstractmethod
   def todense(self):
       pass

   @abc.abstractmethod
   def withdata(self, data):
       pass

   @abc.abstractmethod
   def space(self):
       pass

   @abc.abstractmethod
   def item(self, *idx):
       pass


   # --- Tensor indices --- #

   @abc.abstractmethod
   def inds(self, *tags):
       pass

   @abc.abstractmethod
   def __and__(self, other):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass

   @abc.abstractmethod
   def __xor__(self, other):
       pass


   # --- Tensor properties --- #

   @property
   @abc.abstractmethod
   def dtype(self):
       pass

   @property
   @abc.abstractmethod
   def size(self):
       pass

   @property
   @abc.abstractmethod 
   def ndim(self):
       pass

   @property
   @abc.abstractmethod
   def shape(self):
       pass


   # --- Comparisons --- #

   @abc.abstractmethod
   def __eq__(self, other):
       pass


   # --- Arithmetics and element access --- # 

   @abc.abstractmethod
   def __getitem__(self, idx):
       pass

   @abc.abstractmethod
   def __neg__(self):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __sub__(self, other):
       pass

   @abc.abstractmethod
   def __mul__(self, other):
       pass

   @abc.abstractmethod
   def __truediv__(self, other):
       pass

   @abc.abstractmethod
   def __pow__(self, other):
       pass

   @abc.abstractmethod
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def __rsub__(self, other):
       pass

   @abc.abstractmethod
   def __rmul__(self, other):
       pass

   @abc.abstractmethod
   def __rtruediv__(self, other):
       pass

   @abc.abstractmethod
   def __rpow__(self, other):
       pass




