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

       return self._fun["grads", GradCumulative()](seed)




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Node log interface ---------------------------------------------------- #

class NodeLog(at.NodeLog):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   @property
   def _items(self):

       return self._fun["items", tuple()]()


   def __eq__(self, other):

       return self is other


   def __len__(self):

       return len(self._items)


   def __iter__(self):

       return iter(self._items)


   def __bool__(self):

       return bool(self._items)


   def push(self, *nodes):
       
       return self._fun["push", self](*nodes)


   def pop(self):
       
       return self._fun["pop", fake.Node()]()




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Cumulative gradient interface ----------------------------------------- #

class GradCumulative(at.GradCumulative):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def add(self, nodes, parents):

       return self._fun["add", self](nodes, parents)


   def pick(self, nodes):

       return self._fun["pick", fake.Value()](nodes)


   def result(self):

       return self._fun["result", fake.Value()]()




