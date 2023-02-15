#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tests.common               as common
import tests.autodiff.fakes.util  as util
import tests.autodiff.fakes.node  as node
import tests.autodiff.fakes.grad  as grad

import tadpole.autodiff.util  as tdutil
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- ArgsLike interface ---------------------------------------------------- #

class ArgsLike(tdgraph.ArgsLike):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)

 
   def concat(self):

       return self._fun["concat", Concatenable()]()

 
   def pack(self):

       return self._fun["pack", Packable()]()




# --- Concatenable interface ------------------------------------------------ #

class Concatenable(tdgraph.Concatenable):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def attach(self, node, source, layer):

       return self._fun["attach", self.__class__()](node, source, layer)




# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(tdgraph.Cohesive):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def layer(self):

       return self._fun["layer", util.Value()]()

       
   def adxs(self):

       return self._fun["adxs", util.Value()]()


   def parents(self):

       return self._fun["parents", node.Parental()]()


   def deshell(self):

       return self._fun["deshell", ArgsLike()]()




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Packable interface ---------------------------------------------------- #

class Packable(tdgraph.Packable):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def innermost(self):

       return self._fun["innermost", False]()


   def deshell(self):

       return self._fun["deshell", ArgsLike()]()


   def deshelled(self):

       return self._fun["deshelled", self.__class__()]()


   def fold(self, funwrap, out):

       return self._fun["fold", NodeLike()](funwrap, out)




# --- EnvelopeLike interface ------------------------------------------------ #

class EnvelopeLike(tdgraph.EnvelopeLike): 

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def packs(self):

       default = tdutil.Loop(
                             Packable(), 
                             lambda x: Packable(), 
                             lambda x: True
                            )

       return self._fun["packs", default]()


   def apply(self, fun):

       return self._fun["apply", util.Value()](fun)


   def applywrap(self, funwrap, fun):

       return self._fun["applywrap", node.NodeLike()](funwrap, fun)




