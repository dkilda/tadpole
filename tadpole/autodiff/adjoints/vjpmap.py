#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from tadpole.autodiff.adjoints.adjoints import AdjMap




###############################################################################
###                                                                         ###
###  VJP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class NetVjpFun:

   def __init__(self, vjpfun):

       self._vjpfun = vjpfun


   def __call__(self, adxs, out, *args):

       return lambda g: (self._vjpfun(g, adx, out, *args) for adx in adxs)




# --- VJP map --------------------------------------------------------------- #

class VjpMap(AdjMap):

   def __init__(self):

       super().__init__(lambda adjfun: NetVjpFun(adjfun))




# --- A global map instance and its access points --------------------------- # 

_VJPMAP = VjpMap()


def get(fun):

    return _VJPMAP.get(fun)


def add(fun, *adjfuns, adxs=None):

    return _VJPMAP.add(fun, *adjfuns, adxs=adxs)


def add_combo(fun, adjfun):

    return _VJPMAP.add_combo(fun, adjfun)


def add_raw(fun, adjfun):

    return _VJPMAP.add_raw(fun, adjfun)


     
  
