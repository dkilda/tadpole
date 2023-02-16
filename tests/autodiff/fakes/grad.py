#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tests.common               as common
import tests.autodiff.fakes.util  as util
import tests.autodiff.fakes.node  as node
import tests.autodiff.fakes.grad  as grad

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad




###############################################################################
###                                                                         ###
###  Gradient propagation through the AD computation graph.                 ###
###                                                                         ###
###############################################################################


# --- Gradient propagation interface ---------------------------------------- #

class Propagation(tdgrad.Propagation):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def graphop(self, fun, x):

       graphop = tdgrad.GraphOp(node.GateLike(), fun, x)

       return self._fun["graphop", graphop](fun, x)


   def accum(self, end, seed):

       return self._fun["accum", Cumulative()](end, seed)




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

class Countable(tdgrad.Countable):

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

class Cumulative(tdgrad.Cumulative):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def add(self, nodes, parents):

       return self._fun["add", self](nodes, parents)


   def pick(self, nodes):

       return self._fun["pick", util.Value()](nodes)


   def result(self):

       return self._fun["result", util.Value()]()




