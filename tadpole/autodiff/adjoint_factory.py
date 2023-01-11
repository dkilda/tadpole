#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools




###############################################################################
###                                                                         ###
###  Common code for handling adjoint functions (both JVP and VJP)          ### 
###  from the input.                                                        ###
###                                                                         ###
###############################################################################


# --- Set up adjoint function ----------------------------------------------- #

def make_adjfun(adjfun):

    if adjfun is None: 
       adjfun = lambda g, out, *args: 0

    assert callable(adjfun), f"make_adjfun(): invalid adjfun {adjfun}"

    return adjfun




# --- Concatenate adjoint functions ----------------------------------------- #

def concatenate_adjfuns(*adjfuns, adxs=None):

    if adxs is None:
       adxs = itertools.count() 

    adjfun_by_adx = dict(zip(adxs, map(make_adjfun, adjfuns)))

    def adjfun(g, adx, out, *args):
        return adjfun_by_adx[adx](g, out, *args)

    return adjfun  




###############################################################################
###                                                                         ###
###  JVP factory                                                            ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #

class NetJvpFun:

   def __init__(self, jvpfun):

       self._jvpfun = jvpfun


   def __call__(self, gs, adxs, out, *args):

       jvps = (self._jvpfun(g, adx, out, *args) for g, adx in zip(gs, adxs))

       return jvps # reduce(tdmanip.add_grads, jvps, None) # FIXME




# --- JVP factory ----------------------------------------------------------- #

class JvpFactory:

   _map = {}

   def __init__(self, fun):

       self._fun = fun


   def jvp(self, parent_gs, adxs, out, *args):

       return type(self)._map[self._fun](parent_gs, adxs, out, *args)


   @classmethod
   def add(cls, fun, *jvpfuns, adxs=None):

       cls._map[fun] = NetJvpFun(concatenate_adjfuns(*jvpfuns, adxs=adxs))
       return cls


   @classmethod
   def add_combo(cls, fun, jvpfun):

       cls._map[fun] = NetJvpFun(jvpfun)
       return cls




###############################################################################
###                                                                         ###
###  VJP factory                                                            ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class NetVjpFun:

   def __init__(self, vjpfun):

       self._vjpfun = vjpfun


   def __call__(self, adxs, out, *args):

       return lambda g: (self._vjpfun(g, adx, out, *args) for adx in adxs)


"""
class Vjp:

   def __init__(self, fun, vjpfun, adxs, out, *args):

       self._fun  = fun
       self._vjpfun = vjpfun

       self._adxs = adxs
       self._out  = out
       self._args = args


   def __call__(self, grad):

       return self._vjpfun(
"""



# --- VJP factory ----------------------------------------------------------- #

class VjpFactory:

   _map = {}

   def __init__(self, fun):

       self._fun = fun


   def vjp(self, adxs, out, *args):

       return type(self)._map[self._fun](adxs, out, *args)


   @classmethod
   def add(cls, fun, *vjpfuns, adxs=None):

       cls._map[fun] = NetVjpFun(concatenate_adjfuns(*vjpfuns, adxs=adxs))
       return cls


   @classmethod
   def add_combo(cls, fun, vjpfun):

       cls._map[fun] = NetVjpFun(vjpfun)
       return cls





























