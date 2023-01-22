#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tests.common.fakes import NULL
from tests.common       import make_tuple


def randn(seed=1):

    np.random.seed(seed)    
    return np.random.randn()




###############################################################################
###                                                                         ###
###  General function                                                       ###
###                                                                         ###
###############################################################################


# --- Function return ------------------------------------------------------- #

class FunReturn:
  
   def __add__(self, other):

       return self 


   def __mul__(self, other):

       return self 




# --- General function ------------------------------------------------------ #

class Fun:

   def __init__(self, out=0):

       self._out = out    


   def __call__(self, *args):

       if self._out == 0:
          return FunReturn()

       if isinstance(self._out, int):
          return make_tuple(FunReturn, self._out) 

       return self._out[args]











































































































