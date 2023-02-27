#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util




###############################################################################
###                                                                         ###
###  General framework for array function calls.                            ###
###                                                                         ###
###############################################################################


# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, fun, content=util.Sequence()):

       self._fun     = fun
       self._content = content


   def attach(self, array, data):

       return self.__class__(self._content.push((array, data)))


   def size(self):

       return len(self._content)


   def execute(self):

       arrays, datas = zip(*self._content)
       space         = arrays[0].space() 

       return space.apply(self._fun, *datas) 




# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def __len__(self):

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __getitem__(self, idx):

       return self._args[idx]


   def pluginto(self, funcall):

       for arg in self._args:
           funcall = arg.pluginto(funcall)

       return funcall.execute()




