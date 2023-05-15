#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




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



