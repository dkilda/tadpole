#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc


# TODO MOVE ALL IFACE DEFS HERE!


###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def copy(self, **opts):
       pass

   @abc.abstractmethod
   def space(self):
       pass

   @abc.abstractmethod
   def pluginto(self, funcall):
       pass

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

   @abc.abstractmethod
   def allclose(self, other, **opts):
       pass

   @abc.abstractmethod
   def __eq__(self, other):
       pass

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
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def __rsub__(self, other):
       pass

   @abc.abstractmethod
   def __rmul__(self, other):
       pass








