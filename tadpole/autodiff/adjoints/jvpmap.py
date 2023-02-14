#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from tadpole.autodiff.adjoints.adjoints import AdjMap




###############################################################################
###                                                                         ###
###  JVP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #

class NetJvpFun:

   def __init__(self, jvpfun):

       self._jvpfun = jvpfun


   def __call__(self, adxs, out, *args):

       return lambda gs: (self._jvpfun(g, adx, out, *args) 
                                       for g, adx in zip(gs, adxs))




# --- JVP map --------------------------------------------------------------- #

class JvpMap(AdjMap):

   def __init__(self):

       super().__init__(lambda adjfun: NetJvpFun(adjfun))




# --- A global map instance and its access points --------------------------- # 

_JVPMAP = JvpMap()


def get(fun):

    return _JVPMAP.get(fun)


def add(fun, *adjfuns, adxs=None):

    return _JVPMAP.add(fun, *adjfuns, adxs=adxs)


def add_combo(fun, adjfun):

    return _JVPMAP.add_combo(fun, adjfun)


def add_raw(fun, adjfun):

    return _JVPMAP.add_raw(fun, adjfun)



























