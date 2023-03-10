#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake

import tadpole.util           as util
import tadpole.autodiff.misc  as misc
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- FunWithAdjoint interface ---------------------------------------------- #

class FunWithAdjoint(ag.FunWithAdjoint):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def __call__(self, *args, **kwargs):

       return self._fun["call", fake.Value()](*args, **kwargs)


   def vjp(self, *args, **kwargs):

       return self._fun["vjp", lambda g: fake.Value()](*args, **kwargs)


   def jvp(self, *args, **kwargs):

       return self._fun["jvp", lambda g: fake.Value()](*args, **kwargs)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- ArgsLike interface ---------------------------------------------------- #

class ArgsLike(ag.ArgsLike):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def nodify(self):

       return self._fun["nodify", arepeat(fake.NodeLike, 2)]()

 
   def concat(self):

       return self._fun["concat", Concatenable()]()

 
   def pack(self):

       return self._fun["pack", Packable()]()

 
   def deshelled(self):

       return self._fun["deshelled", self.__class__()]()



# --- Concatenable interface ------------------------------------------------ #

class Concatenable(ag.Concatenable):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def attach(self, node, source, layer):

       return self._fun["attach", self.__class__()](node, source, layer)




# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(ag.Cohesive):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def innermost(self):

       return self._fun["innermost", self.layer() == misc.minlayer()]()


   def layer(self):

       return self._fun["layer", fake.Value()]()

       
   def adxs(self):

       return self._fun["adxs", fake.Value()]()


   def parents(self):

       return self._fun["parents", fake.Parental()]()


   def deshell(self):

       return self._fun["deshell", ArgsLike()]()




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Packable interface ---------------------------------------------------- #

class Packable(ag.Packable):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def innermost(self):

       return self._fun["innermost", False]()


   def deshell(self):

       return self._fun["deshell", ArgsLike()]()


   def deshelled(self):

       return self._fun["deshelled", self.__class__()]()


   def fold(self, funwrap, out):

       return self._fun["fold", fake.NodeLike()](funwrap, out)




# --- EnvelopeLike interface ------------------------------------------------ #

class EnvelopeLike(ag.EnvelopeLike): 

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def packs(self):

       default = util.Loop(
                           Packable(), 
                           lambda x: Packable(), 
                           lambda x: True
                          )

       return self._fun["packs", default]()


   def apply(self, fun):

       return self._fun["apply", fake.Value()](fun)


   def applywrap(self, funwrap, fun):

       return self._fun["applywrap", fake.NodeLike()](funwrap, fun)




