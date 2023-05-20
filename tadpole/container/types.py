#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util as util




###############################################################################
###                                                                         ###
###  Core                                                                   ###
###                                                                         ###
###############################################################################


# --- Generic container ----------------------------------------------------- #

class Container(util.Container):
   pass




# --- Tensor container ------------------------------------------------------ #

class TensorContainer(util.Container):

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
   def item(self, pos):
       pass




