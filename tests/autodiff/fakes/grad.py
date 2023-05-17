#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad




###############################################################################
###                                                                         ###
###  Gradient propagation through the AD computation graph.                 ###
###                                                                         ###
###############################################################################


# --- Gradient propagation interface ---------------------------------------- #

class Propagation(at.Propagation):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def apply(self, fun):

       return self._fun["apply", fake.Node()](fun)


   def grads(self, seed):

       return self._fun["grads", Cumulative()](seed)




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Traceable interface --------------------------------------------------- #

class Traceable(at.Traceable):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def record(self, node, parents):
       
       return self._fun["record", self](node, parents)




# --- Countable interface --------------------------------------------------- #

class Countable(at.Countable):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


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

class Cumulative(at.Cumulative):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def add(self, nodes, parents):

       return self._fun["add", self](nodes, parents)


   def pick(self, nodes):

       return self._fun["pick", fake.Value()](nodes)


   def result(self):

       return self._fun["result", fake.Value()]()




