#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.backends as backends
import tadpole.array.util     as util




# --- Gradient interface ---------------------------------------------------- #

class Grad(abc.ABC):

   @abc.abstractmethod
   def __iadd__(self, other):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __mul__(self, other):
       pass



# --- Dense gradient -------------------------------------------------------- #

class DenseGrad(Grad):

   def __init__(self):


   def __add__(self, other):

        
















































