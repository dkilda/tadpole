#!/usr/bin/env python3
# -*- coding: utf-8 -*-




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
       adxs = (adx for adx in range(len(adjfuns)))  

    adjfun_by_adx = {adx: make_adjfun(adjfuns[adx]) for adx in adxs}

    def adjfun(g, adx, out, *args):
        return adjfun_by_adx[adx](g, out, *args)

    return adjfun  




# --- Adjoint map ----------------------------------------------------------- #

class AdjointMap:

   def __init__(self, factory):

       self._map     = {}
       self._factory = factory


   def get(self, fun): 

       return self._map[fun] 


   def _add(self, fun, adjfun):

       self._map[fun] = self._factory(adjfun)
       return self


   def add(self, fun, *adjfuns, adxs=None):

       return self._add(fun, concatenate_adjfuns(*adjfuns, adxs=adxs))


   def add_combo(self, fun, adjfun):

       return self._add(fun, make_adjfun(adjfun))


   def add_raw(self, fun, adjfun):

       self._map[fun] = make_adjfun(adjfun)
       return self




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

class VjpMap(AdjointMap):

   def __init__(self):

       super().__init__(lambda adjfun: NetVjpFun(adjfun))




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

class JvpMap(AdjointMap):

   def __init__(self):

       super().__init__(lambda adjfun: NetJvpFun(adjfun))






     
  
