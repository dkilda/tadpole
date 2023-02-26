#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
### Interface for tuple-like types.                                         ###
###                                                                         ###
###############################################################################


# --- TupleLike interface --------------------------------------------------- #

class TupleLike(abc.ABC):

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




