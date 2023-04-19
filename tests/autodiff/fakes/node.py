#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.common import arepeat, arange, amap

import tests.autodiff.fakes   as fake
import tadpole.autodiff.types as at




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint operator ------------------------------------------------------ #

class AdjointOp(at.AdjointOp):

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


# --- Flow ------------------------------------------------------------------ #

class Flow(at.Flow):

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

       return self._fun["gate", Gate()](parents, op)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Gate type ------------------------------------------------------------- #

class Gate(at.Gate):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def flow(self):

       return self._fun["flow", Flow()]()


   def trace(self, node, traceable):

       return self._fun["trace", traceable](node, traceable)


   def grads(self, node, grads):

       return self._fun["grads", grads](node, grads)  




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node type ------------------------------------------------------------- #

class Node(at.Node):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def connected(self, other):
 
       return self._fun["connected", True](other)


   def concat(self, concatenable):

       return self._fun["concat", concatenable](concatenable)


   def flow(self):

       return self._fun["flow", Flow()]()


   def trace(self, traceable):

       return self._fun["trace", traceable](traceable)


   def grads(self, grads):

       return self._fun["grads", grads](grads)




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- Parents type ---------------------------------------------------------- #

class Parents(at.Parents):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   @property
   def _items(self):

       return self._fun["items", tuple()]()


   def __eq__(self, other):

       return id(self) == id(other)


   def __hash__(self):

       return id(self)


   def __len__(self):

       return len(self._items)


   def __contains__(self, x):

       return x in self._items


   def __iter__(self):

       return iter(self._items)


   def __getitem__(self, idx):

       return self._items[idx] 


   def next(self, source, layer, op):

       return self._fun["next", Node()](source, layer, op)




