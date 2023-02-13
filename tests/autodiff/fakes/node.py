#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tests.common               as common
import tests.autodiff.fakes.util  as util
import tests.autodiff.fakes.graph as graph
import tests.autodiff.fakes.grad  as grad

import tadpole.autodiff.node as tdnode


"""

If we don't spec e.g. trace, do we really care what .trace() returns? We won't be able to assert it anyway...


   def trace(self, node, traceable):

       if self._trace is None:
          return grad.Traceable()

       return self._trace(node, traceable) # FIXME it calls trace function, which returns a traceable object
                                           # if not provided, just return a generic traceable (means we don't care)
                                           # every data input should be a function, w/ or w/o args!


"""




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint interface ----------------------------------------------------- #

class Adjoint(tdnode.Adjoint):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def vjp(self, seed):

       return self._fun["vjp", seed](seed)


   def jvp(self, seed):

       return self._fun["jvp", seed](seed)




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- FlowLike interface ---------------------------------------------------- #

class FlowLike(tdnode.FlowLike):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def __eq__(self, other):

       return self._fun["eq", id(self) == id(other)](other)       


   def __hash__(self):

       return self._fun["hash", id(self)]()


   def __add__(self, other):

       return self._fun["add", self](other)

       
   def __radd__(self, other):

       return self.__add__(other)

 
   def gate(self, parents, op):

       return self._fun["gate", GateLike()](parents, op)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- GateLike interface ---------------------------------------------------- #

class GateLike(tdnode.GateLike):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def flow(self):

       return self._fun["flow", FlowLike()]()


   def trace(self, node, traceable):

       return self._fun["trace", traceable](node, traceable)


   def grads(self, node, grads):

       return self._fun["grads", grads](node, grads)  




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- NodeLike interface ---------------------------------------------------- #

class NodeLike(tdnode.NodeLike):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def tovalue(self):

       return self._fun["tovalue", util.Value()]()


   def concat(self, concatenable):

       return self._fun["concat", concatenable](concatenable)


   def flow(self):

       return self._fun["flow", FlowLike()]()


   def trace(self, traceable):

       return self._fun["trace", traceable](traceable)


   def grads(self, grads):

       return self._fun["grads", grads](grads)




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- Parental interface ---------------------------------------------------- #

class Parental(tdnode.Parental):

   def __init__(self, **data):  

       self._fun = util.FunMap(**data)


   def next(self, source, layer, op):

       return self._fun["next", NodeLike()](source, layer, op)




