#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Array space                                                            ###
###                                                                         ###
###############################################################################


# --- Space interface ------------------------------------------------------- #

class Space(abc.ABC):

   # --- Factories --- #

   @abc.abstractmethod
   def sparse(self, idxs, vals):
       pass

   @abc.abstractmethod
   def zeros(self):
       pass

   @abc.abstractmethod
   def zeros(self):
       pass

   @abc.abstractmethod
   def ones(self):
       pass

   @abc.abstractmethod
   def unit(self):
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

   @abc.abstractmethod
   def units(self):
       pass

   @abc.abstractmethod
   def basis(self):
       pass


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




###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- Pluggable interface --------------------------------------------------- #

class Pluggable(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, funcall):
       pass




# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   # --- Using in gradient accumulations --- #

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


   # --- Array properties --- #

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
   def allequal(self, other):
       pass

   @abc.abstractmethod
   def allclose(self, other, **opts):
       pass

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
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def __rsub__(self, other):
       pass

   @abc.abstractmethod
   def __rmul__(self, other):
       pass




###############################################################################
###                                                                         ###
###  General framework for array function calls.                            ###
###                                                                         ###
###############################################################################


# --- ContentLike interface ------------------------------------------------- #

class ContentLike(abc.ABC):

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def attach(self, backend, data):
       pass




# --- Function call interface ----------------------------------------------- #

class FunCall(abc.ABC):

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def attach(self, backend, data):
       pass

   @abc.abstractmethod
   def execute(self):
       pass







