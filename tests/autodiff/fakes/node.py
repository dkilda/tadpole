#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.common import arepeat, arange, amap

import tests.autodiff.fakes  as fake
import tadpole.autodiff.node as anode




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint interface ----------------------------------------------------- #

class Adjoint(anode.Adjoint):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


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

class FlowLike(anode.FlowLike):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


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

class GateLike(anode.GateLike):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


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

class NodeLike(anode.NodeLike):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def connected(self):
 
       return self._fun["connected", True]()


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

class Parental(anode.Parental):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def next(self, source, layer, op):

       return self._fun["next", NodeLike()](source, layer, op)




