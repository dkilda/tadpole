#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.autodiff.grad  import add_grads
from tadpole.autodiff.graph import differentiable, nondifferentiable


@differentiable
def sin(x):
    return np.sin(x)

@differentiable
def cos(x):
    return np.cos(x)

@differentiable
def neg(x):
    return -x

@differentiable
def add(x, y):
    return x + y

@differentiable
def sub(x, y):
    return x - y

@differentiable
def mul(x, y):
    return x * y


@nondifferentiable
def floor(x, n):
    return x // n

@nondifferentiable
def equals(x, y):
    return x == y 




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




###############################################################################
###                                                                         ###
###  Some VJPs and JVPs                                                     ###
###                                                                         ###
###############################################################################


# --- VJPs ------------------------------------------------------------------ #

VjpFactory.add(add, lambda g, out, x, y: g, 
                    lambda g, out, x, y: g)

VjpFactory.add(sub, lambda g, out, x, y: g, 
                    lambda g, out, x, y: neg(g))

VjpFactory.add(mul, lambda g, out, x, y: mul(y, g), 
                    lambda g, out, x, y: mul(x, g))

VjpFactory.add(neg, lambda g, out, x: neg(g))
VjpFactory.add(sin, lambda g, out, x: mul(g, cos(x)))
VjpFactory.add(cos, lambda g, out, x: neg(mul(g, sin(x))))




# --- JVPs ------------------------------------------------------------------ #

JvpFactory.add(add, lambda g, out, x, y: g, 
                    lambda g, out, x, y: g)

JvpFactory.add(sub, lambda g, out, x, y: g, 
                    lambda g, out, x, y: neg(g))

JvpFactory.add(mul, lambda g, out, x, y: mul(y, g), 
                    lambda g, out, x, y: mul(x, g))

JvpFactory.add(neg, lambda g, out, x: neg(g))
JvpFactory.add(sin, lambda g, out, x: mul(g, cos(x)))
JvpFactory.add(cos, lambda g, out, x: neg(mul(g, sin(x))))




















































