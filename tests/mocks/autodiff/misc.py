#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from tests.mocks.common import NULL, mockify




###############################################################################
###                                                                         ###
###  General function                                                       ###
###                                                                         ###
###############################################################################


# --- Function return ------------------------------------------------------- #

class FunReturn:

   pass




# --- General function ------------------------------------------------------ #

class Fun:

   def __init__(self, valency=0):

       self._valency = valency    


   def __call__(self, *args):

       if self._valency is NULL:
          return FunReturn()

       return tuple([FunReturn()]*self._valency)










































































































