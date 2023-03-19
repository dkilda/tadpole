#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  General ArrayLike interface for OneArray/TwoArray/NArray/etc objects   ###
###                                                                         ###
###############################################################################


# --- ArrayLike interface --------------------------------------------------- #

class ArrayLike(abc.ABC):

   @abc.abstractmethod
   def new(self, data):
       pass

   @abc.abstractmethod
   def __or__(self, other):
       pass










