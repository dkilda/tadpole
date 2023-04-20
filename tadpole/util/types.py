#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Minimal interface for scalar types                                     ###
###                                                                         ###
###############################################################################


# --- Scalar ---------------------------------------------------------------- #

class Scalar(abc.ABC):

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
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
   def __mod__(self, other):
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
   def __rmod__(self, other):
       pass  

   @abc.abstractmethod
   def __rpow__(self, other):
       pass 




###############################################################################
###                                                                         ###
###  Minimal interface for container types                                  ###
###                                                                         ###
###############################################################################


# --- Container ------------------------------------------------------------- #

class Container(abc.ABC):

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def __contains__(self, x):
       pass

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __getitem__(self, idx):
       pass




