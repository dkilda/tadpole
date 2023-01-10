#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.tests.mocks.util import NULL, mockify

from tadpole.autodiff.node import Forward, Reverse
from tadpole.autodiff.node import Node, Gate, GateInputs



###############################################################################
###                                                                         ###
###  Gates of the autodiff circuit                                          ###
###                                                                         ###
###############################################################################


# --- Gate ------------------------------------------------------------------ #

class MockGate(Gate):

   def __init__(self, node=NULL, next_input=NULL):

       self._node       = node
       self._next_input = next_input


   @mockify
   def node(self, source, layer):

       return self._node


   @mockify
   def next_input(self, others, adxs, args, source):

       return self._next_input




# --- Forward gate ---------------------------------------------------------- #

class MockForwardGate(Gate, Forward): 

   def __init__(self, core=NULL, grad=NULL):

       self._core = core
       self._grad = grad


   @mockify
   def node(self, source, layer):

       return self._core.node(source, layer)


   @mockify
   def next_input(self, others, adxs, args, source):

       return self._core.next_input(others, adxs, args, source)


   @mockify
   def grad(self):

       return self._grad




# --- Reverse gate ---------------------------------------------------------- #

class MockReverseGate(Gate, Reverse): 

   def __init__(self, core=NULL, parents=NULL, grads=NULL): 

       self._core    = core
       self._parents = parents
       self._grads   = grads


   @mockify
   def node(self, source, layer):

       return self._core.node(source, layer)


   @mockify
   def next_input(self, others, adxs, args, source):

       return self._core.next_input(others, adxs, args, source)


   @mockify
   def accumulate_parent_grads(self, grads):

       for parent, grad in zip(self._parents, self._grads):
           grads.accumulate(parent, grad) 
       return self


   @mockify
   def add_to_childcount(self, childcount):
 
       childcount.add(self, self._parents)
       return self


   @mockify
   def add_to_toposort(self, toposort):
 
       toposort.add(self)
       return self




# --- Gate inputs ----------------------------------------------------------- #

class MockGateInputs(GateInputs):

   def __init__(self, transform=NULL):

       self._transform = transform


   @mockify
   def transform(self, fun):

       return self._transform[fun]




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #

class MockNode(Node):

   def __init__(self, reduce_=NULL, topoint_=NULL, glue_=NULL):
 
       self._reduce  = reduce_
       self._topoint = topoint_
       self._glue    = glue_


   @mockify
   def reduce(self):

       return self._reduce


   @mockify
   def topoint(self):

       return self._topoint


   @mockify
   def glue(self, *others):

       return self._glue




# --- Forward node ---------------------------------------------------------- #

class MockForwardNode(Node, Forward):

   def __init__(self, core=NULL, grad=NULL):

       self._core = core
       self._grad = grad
 

   @mockify
   def reduce(self):

       return self._core.reduce()


   @mockify
   def topoint(self):

       return self._core.topoint()


   @mockify
   def glue(self, *others):

       return self._core.glue()


   @mockify
   def grad(self):

       return self._grad




# --- Reverse node ---------------------------------------------------------- #

class MockReverseNode(Node, Reverse):

   def __init__(self, core=NULL, grads=NULL, parents=NULL):  
 
       self._core    = core
       self._parents = parents
       self._grads   = grads


   @mockify
   def reduce(self):

       return self._core.reduce()


   @mockify
   def topoint(self):

       return self._core.topoint()


   @mockify
   def glue(self, *others):

       return self._core.glue()


   @mockify
   def accumulate_parent_grads(self, grads):

       for parent, grad in zip(self._parents, self._grads):
           grads.accumulate(parent, grad) 
       return self


   @mockify
   def add_to_childcount(self, childcount):
 
       childcount.add(self, self._parents)
       return self


   @mockify
   def add_to_toposort(self, toposort):
 
       toposort.add(self)
       return self




# --- Point ----------------------------------------------------------------- #

class MockPoint(Node):

   def __init__(self, reduce_=NULL, glue_=NULL):

       self._reduce = reduce_
       self._glue   = glue_


   @mockify
   def reduce(self):

       return self._reduce


   @mockify
   def topoint(self):

       return self


   @mockify
   def glue(self, *others):

       return self._glue

























































