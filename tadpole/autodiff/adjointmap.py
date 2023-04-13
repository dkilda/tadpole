#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc




###############################################################################
###                                                                         ###
###  Common code for handling adjoint functions (both JVP and VJP)          ### 
###  from the input.                                                        ###
###                                                                         ###
###############################################################################


# --- Special adjoint functions --------------------------------------------- #

class AdjFunNull:

   def __call__(self, g, out, *args, **kwargs):

       return 0




class AdjFunLinear:

   def __init__(self, fun, adx):
  
       if not isinstance(fun, AdjFunComboLinear):
          fun = AdjFunComboLinear(fun)

       if not callable(fun):
          raise ValueError(
             f"AdjFunLinear: fun must be callable, but is {fun}"
          )

       self._fun = fun
       self._adx = adx


   def __call__(self, g, out, *args, **kwargs):

       return self._fun(g, self._adx, out, *args, **kwargs)




class AdjFunComboLinear:

   def __init__(self, fun):

       if not callable(fun):
          raise ValueError(
             f"AdjFunComboLinear: fun must be callable, but is {fun}"
          )
  
       self._fun = fun


   def __call__(self, g, adx, out, *args, **kwargs):

       args      = list(args)
       args[adx] = g

       return self._fun(*args, **kwargs)




# --- Set up adjoint function ----------------------------------------------- #

def make_adjfun(adjfun, fun=None, adx=None):

    if adjfun is None or adjfun == "null": 
       return AdjFunNull()  

    if adjfun == "linear" and adx is not None:
       return AdjFunLinear(fun, adx)

    if adjfun == "linear":
       return AdjFunComboLinear(fun)

    assert callable(adjfun), f"make_adjfun(): invalid adjfun {adjfun}"

    return adjfun




# --- Concatenate adjoint functions ----------------------------------------- #

def concatenate_adjfuns(fun, *adjfuns, adxs=None):

    if adxs is None:
       adxs = (adx for adx in range(len(adjfuns)))  

    adjfun_by_adx = {adx: make_adjfun(adjfuns[adx], fun, adx) for adx in adxs}

    def adjfun(g, adx, out, *args, **kwargs):
        return adjfun_by_adx[adx](g, out, *args, **kwargs)

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

       return self._add(fun, concatenate_adjfuns(fun, *adjfuns, adxs=adxs))


   def add_combo(self, fun, adjfun):

       return self._add(fun, make_adjfun(adjfun, fun))


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


   def __call__(self, adxs, out, *args, **kwargs):

       return lambda g: (
          self._vjpfun(g, adx, out, *args, **kwargs) 
             for adx in adxs
       )




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


   def __call__(self, adxs, out, *args, **kwargs):

       return lambda gs: (
          self._jvpfun(g, adx, out, *args, **kwargs) 
             for g, adx in zip(gs, adxs)
       )




# --- JVP map --------------------------------------------------------------- #

class JvpMap(AdjointMap):

   def __init__(self):

       super().__init__(lambda adjfun: NetJvpFun(adjfun))






     
  
