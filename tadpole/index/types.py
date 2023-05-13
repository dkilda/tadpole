#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Index                                                                  ###
###                                                                         ###
###############################################################################


# --- Index ----------------------------------------------------------------- #

class Index(abc.ABC):

   # --- String representation --- #

   @abc.abstractmethod
   def __repr__(self):
       pass


   # --- Equality and hashing --- #

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass


   # --- General methods --- #

   @abc.abstractmethod
   def all(self, *tags):
       pass

   @abc.abstractmethod
   def any(self, *tags):
       pass

   @abc.abstractmethod
   def resized(self, start, end):
       pass

   @abc.abstractmethod
   def retagged(self, tags):
       pass















































