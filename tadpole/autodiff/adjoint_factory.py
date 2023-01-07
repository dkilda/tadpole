#!/usr/bin/env python3
# -*- coding: utf-8 -*-


@differentiable
def sin(x):
    return np.sin(x)

@differentiable
def cos(x):
    return np.cos(x)

@differentiable
def uneg(x):
    return -x

@differentiable
def add(x, y):
    return x + y


def add_grads(net_g, g): # TODO impl and use add() function, with @diffable decorator 
                         #      (or overload __add__ operator to make it @diffable)
    if net_g is None:  
       return g

    return add(net_g, g)




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

       return reduce(add_grads, jvps, None)




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




































































