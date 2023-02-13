#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tests.common               as common
import tests.autodiff.fakes.util  as util
import tests.autodiff.fakes.node  as node
import tests.autodiff.fakes.grad  as grad

import tadpole.autodiff.grad as tdgrad



###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Traceable interface --------------------------------------------------- #

class Traceable(tdgrad.Traceable):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def record(self, node, parents):
       
       return self._fun["record", self](node, parents)




# --- Countable interface --------------------------------------------------- #

class Countable(tdgraph.Countable):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def collect(self, node):

       return self._fun["countable", tuple()](node)


   def increase(self, node):

       return self._fun["increase", tuple()](node)


   def decrease(self, node):
       
       return self._fun["decrease", tuple()](node)




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Cumulative interface -------------------------------------------------- #

class Cumulative(tdgraph.Cumulative):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def add(self, nodes, parents):

       return self._fun["add", self](nodes, parents)


   def pop(self, nodes):

       return self._fun["pop", util.Value()]


   def result(self, node):

       return self._fun["result", util.Value()]




