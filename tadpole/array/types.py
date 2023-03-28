#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Array modules: unary, binary, nary                                     ###
###                                                                         ###
###############################################################################


# --- Array ----------------------------------------------------------------- #

class Array(abc.ABC):

   @abc.abstractmethod
   def new(self, data):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass




###############################################################################
###                                                                         ###
###  Space                                                                  ###
###                                                                         ###
###############################################################################


# --- Space ----------------------------------------------------------------- #

class Space(abc.ABC):

   # --- Space properties --- #

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


   # --- Array creation methods --- #

   @abc.abstractmethod
   def zeros(self, **opts):
       pass

   @abc.abstractmethod
   def ones(self, **opts):
       pass

   @abc.abstractmethod
   def unit(self, idx, **opts):
       pass

   @abc.abstractmethod
   def rand(self, **opts):
       pass

   @abc.abstractmethod
   def randn(self, **opts):
       pass

   @abc.abstractmethod
   def randuniform(self, boundaries, **opts):
       pass


   # --- Array generators --- #

   @abc.abstractmethod
   def units(self, **opts):
       pass

   @abc.abstractmethod
   def basis(self):
       pass




