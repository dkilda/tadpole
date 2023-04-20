#!/usr/bin/env python3
# -*- coding: utf-8 -*-




###############################################################################
###                                                                         ###
###  Setting up adjoint functions                                           ###
###                                                                         ###
###############################################################################


# --- Setting up VJP functions ---------------------------------------------- #

def make_vjpfun(adjfun, fun=None, adx=None):

    if adjfun == "null":
       return lambda g, out, *args, **kwargs: args[adx].tonull()

    if adjfun == "identity":
       return lambda g, out, *args, **kwargs: g

    assert callable(adjfun), f"make_vjpfun(): invalid adjfun {adjfun}"

    return adjfun




# --- Setting up JVP functions ---------------------------------------------- #

def make_jvpfun(adjfun, fun=None, adx=None):

    if adjfun == "null":
       return lambda g, out, *args, **kwargs: out.tonull()

    if adjfun == "identity":
       return lambda g, out, *args, **kwargs: g

    if adjfun == "linear":
       return lambda g, out, *args, **kwargs: (
                 linear_adjfun(fun)(g, adx, out, *args, **kwargs)
              )

    assert callable(adjfun), f"make_jvpfun(): invalid adjfun {adjfun}"

    return adjfun




# --- Special adjoint functions --------------------------------------------- #

def linear_adjfun(fun):

    def adjfun(g, adx, out, *args, **kwargs):

        args      = list(args)
        args[adx] = g

        return fun(*args, **kwargs) 
       
    return adjfun




###############################################################################
###                                                                         ###
###  Constructing net (concatenated) adjoint functions                      ###
###                                                                         ###
###############################################################################


# --- Concatenate adjoint functions ----------------------------------------- #

def concat_adjfuns(make_adjfun):

    def concat(fun, *adjfuns, adxs=None):

        if adxs is None:
           adxs = (adx for adx in range(len(adjfuns)))

        adjfun_by_adx = {adx: make_adjfun(adjfuns[adx], fun, adx) 
                              for adx in adxs}        

        def adjfun(g, adx, out, *args, **kwargs):
            return adjfun_by_adx[adx](g, out, *args, **kwargs)

        return adjfun 

    return concat




# --- Concatenate VJP functions --------------------------------------------- #

def concat_vjpfuns(*args, **kwargs):

    return concat_adjfuns(make_vjpfun)(*args, **kwargs)




# --- Concatenate JVP functions --------------------------------------------- #

def concat_jvpfuns(*args, **kwargs):

    return concat_adjfuns(make_jvpfun)(*args, **kwargs)




# --- Net (concatenated) VJP function --------------------------------------- #

def make_net_vjpfun(adjfun):

    if isinstance(adjfun, (tuple, list)):

       adjfun = dict()[funtype](fun)

    return NetVjpFun(adjfun)




class NetVjpFun:

   def __init__(self, vjpfun):

       self._vjpfun = vjpfun


   def __call__(self, adxs, out, *args, **kwargs):

       return lambda g: (
          self._vjpfun(g, adx, out, *args, **kwargs) 
             for adx in adxs
       )




# --- Net (concatenated) JVP function --------------------------------------- #

def make_net_jvpfun(adjfun):

    if isinstance(adjfun, (tuple, list)):

       funtype, fun = adjfun

       adjfun = {
                 "linear": linear_adjfun,
                }[funtype](fun)

    return NetJvpFun(adjfun)




class NetJvpFun:

   def __init__(self, jvpfun):

       self._jvpfun = jvpfun


   def __call__(self, adxs, out, *args, **kwargs):

       return lambda gs: (
          self._jvpfun(g, adx, out, *args, **kwargs) 
             for g, adx in zip(gs, adxs)
       )




###############################################################################
###                                                                         ###
###  Adjoint maps                                                           ###
###                                                                         ###
###############################################################################


# --- Generic adjoint map --------------------------------------------------- #

class AdjointMap:

   def __init__(self, concat, factory):

       self._map     = {}
       self._concat  = concat
       self._factory = factory


   def get(self, fun): 

       return self._map[fun] 


   def _add(self, fun, adjfun):

       self._map[fun] = self._factory(adjfun)
       return self


   def add(self, fun, *adjfuns, adxs=None):

       return self._add(fun, self._concat(fun, *adjfuns, adxs=adxs))


   def add_combo(self, fun, adjfun):

       if isinstance(adjfun, str):
          adjfun = (adjfun, fun)

       return self._add(fun, adjfun)


   def add_raw(self, fun, adjfun):

       self._map[fun] = adjfun 
       return self




# --- VJP map --------------------------------------------------------------- #

class VjpMap(AdjointMap):

   def __init__(self):

       super().__init__(concat_vjpfuns, make_net_vjpfun)




# --- JVP map --------------------------------------------------------------- #

class JvpMap(AdjointMap):

   def __init__(self):

       super().__init__(concat_jvpfuns, make_net_jvpfun)




