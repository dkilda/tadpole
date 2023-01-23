#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tests.common.ntuple as tpl

from tests.common.fakes import NULL



def randn(seed=1):

    np.random.seed(seed)    
    return np.random.randn()




# --- Trivial map ----------------------------------------------------------- #

class TrivMap:

   def __init__(self, out):

       self._out = out


   def __getitem__(self, key):

       return self._out




# --- Generic map ----------------------------------------------------------- #

class Map:

   def __init__(self, out):

       self._out = out


   def __getitem__(self, key):

       try:
          return self._out[key]

       except KeyError:

          keys = list(self._out.keys())
          return self._out[keys[keys.index(key)]]




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
          return tpl.repeat(FunReturn, self._out) 

       return self._out[args]











































































































