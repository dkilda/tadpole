#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Core                                                                   ###
###                                                                         ###
###############################################################################


# --- Pluggable ------------------------------------------------------------- #

class Pluggable(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, engine):
       pass




# --- Grad ------------------------------------------------------------------ #

class Grad(abc.ABC):

   @abc.abstractmethod
   def addto(self, other):
       pass

   @abc.abstractmethod
   def todense(self):
       pass

   @abc.abstractmethod
   def tonull(self):
       pass




# --- Tensor ---------------------------------------------------------------- #

class Tensor(abc.ABC):

   # --- Basic functionality --- #

   @abc.abstractmethod
   def copy(self, **opts):
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
   def __getitem__(self, pos):
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




###############################################################################
###                                                                         ###
###  Space                                                                  ###
###                                                                         ###
###############################################################################


# --- Space ----------------------------------------------------------------- #

class Space(abc.ABC):

   # --- Fill the space with data --- #

   @abc.abstractmethod
   def fillwith(self, data):
       pass


   # --- Gradient factories --- #

   @abc.abstractmethod
   def sparsegrad(self, pos, vals):
       pass

   @abc.abstractmethod
   def nullgrad(self):
       pass


   # --- Tensor factories --- #

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
###  Element                                                                ###
###                                                                         ###
###############################################################################


# --- Element interface ----------------------------------------------------- #

class Element(abc.ABC):

   @abc.abstractmethod
   def positions(self, tensor_inds):
       pass

   @abc.abstractmethod
   def inds(self, tensor_inds):
       pass


       

###############################################################################
###                                                                         ###
###  Engine                                                                 ###
###                                                                         ###
###############################################################################


# --- Tensor operations engine ---------------------------------------------- #

class Engine(abc.ABC): 

   @abc.abstractmethod
   def attach(self, data, inds):
       pass

   @abc.abstractmethod
   def operator(self):
       pass




###############################################################################
###                                                                         ###
###  Contraction                                                            ###
###                                                                         ###
###############################################################################


# --- Index product --------------------------------------------------------- #

class IndexProduct(abc.ABC):

   @abc.abstractmethod
   def __call__(self, inds):
       pass




###############################################################################
###                                                                         ###
###  Decomp                                                                 ###
###                                                                         ###
###############################################################################


# --- Alignment ------------------------------------------------------------- #

class Alignment(abc.ABC):

   @abc.abstractmethod
   def linds(self, inds):
       pass

   @abc.abstractmethod
   def rinds(self, inds):
       pass




###############################################################################
###                                                                         ###
###  Truncation                                                             ###
###                                                                         ###
###############################################################################


# --- Cutoff mode ----------------------------------------------------------- #

class CutoffMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S):
       pass




# --- Error mode ------------------------------------------------------------ #

class ErrorMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S, rank):
       pass




# --- Truncation ------------------------------------------------------------ #

class Trunc(abc.ABC):

   @abc.abstractmethod
   def rank(self, S):
       pass

   @abc.abstractmethod
   def error(self, S):
       pass

   @abc.abstractmethod
   def apply(self, U, S, VH):
       pass




