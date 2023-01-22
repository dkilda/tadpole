#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.node as tdnode

from tests.common.fakes import NULL, fakeit




###############################################################################
###                                                                         ###
###  Logic of forward and reverse propagation, creates logic gates.         ###
###                                                                         ###
###############################################################################


# --- Logic ----------------------------------------------------------------- #

class Logic(tdnode.Logic):

   def __init__(self, gate=NULL):

       self._gate = gate


   @fakeit
   def gate(self, fun):

       return self._gate




# --- Forward logic --------------------------------------------------------- #

class ForwardLogic(Logic):
   pass




# --- Reverse logic --------------------------------------------------------- #

class ReverseLogic(Logic):
   pass




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Gate ------------------------------------------------------------------ #

class Gate(tdnode.Gate):

   def __init__(self, nodify=NULL):

       self._nodify = nodify


   @fakeit   
   def nodify(self, nodule):

       return self._nodify[nodule]




# --- Forward logic gate ---------------------------------------------------- #

class ForwardGate(Gate):

   def __init__(self, nodify=NULL, grad=NULL):

       super().__init__(nodify)
       self._grad = grad


   @fakeit
   def grad(self):

       return self._grad

       


# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(Gate):

   def __init__(self, nodify=NULL, parents=NULL, grads=NULL):

       super().__init__(nodify)
       self._parents = parents
       self._grads   = grads


   @fakeit
   def accumulate_parent_grads(self, seed, grads):

       for parent, grad in zip(self._parents, self._grads):
           grads.accumulate(parent, grad) 

       return self


   @fakeit
   def add_to_childcount(self, childcount):

       childcount.add(self._parents)

       return self          
    

   @fakeit
   def add_to_toposort(self, toposort):

       for parent in self._parents:
           toposort.add(parent)

       return self




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #

class Node(tdnode.Node):

   def __init__(self, tovalue=NULL, attach=NULL):

       self._tovalue = tovalue
       self._attach  = attach


   @fakeit
   def tovalue(self):

       return self._tovalue


   @fakeit
   def attach(self, train):

       return self._attach[train]




# --- Nodule: a node kernel ------------------------------------------------- #

class Nodule(Node):
   pass
  



# --- Forward node ---------------------------------------------------------- #

class ForwardNode(Node):

   def __init__(self, tovalue=NULL, attach=NULL, logic=NULL, gate=NULL):

       super().__init__(tovalue, attach)

       self._logic = logic
       self._gate  = gate


   @fakeit
   def logic(self, others, adxs, source, args):

       return self._logic


   @fakeit
   def grad(self):

       return self._gate.grad()




# --- Reverse node ---------------------------------------------------------- #

class ReverseNode(Node):

   def __init__(self, tovalue=NULL, attach=NULL, logic=NULL, gate=NULL):

       super().__init__(tovalue, attach)

       self._logic = logic
       self._gate  = gate


   @fakeit
   def logic(self, others, adxs, source, args):

       return self._logic


   @fakeit
   def accumulate_parent_grads(self, grads):

       seed = grads.pop(self)
       self._gate.accumulate_parent_grads(seed, grads)
       return self


   @fakeit
   def add_to_childcount(self, childcount):

       childcount.visit(self)
       self._gate.add_to_childcount(childcount)
       return self          
    

   @fakeit
   def add_to_toposort(self, toposort):

       self._gate.add_to_toposort(toposort)
       return self




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class Point(Node):
   pass





