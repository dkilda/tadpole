#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




# --- Tensor operations engine ---------------------------------------------- #

class Engine(abc.ABC): 

   @abc.abstractmethod
   def attach(self, data, inds):
       pass

   @abc.abstractmethod
   def create(self):
       pass
