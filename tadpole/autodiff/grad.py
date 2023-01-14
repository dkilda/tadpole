#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.util    as tdutil
import tadpole.autodiff.nary_op as tdnary
import tadpole.autodiff.node    as tdnode
import tadpole.autodiff.graph   as tdgraph
import tadpole.autodiff.manip   as tdmanip




###############################################################################
###                                                                         ###
###  Specialized differential operators                                     ###
###                                                                         ###
###############################################################################


# --- Gradient -------------------------------------------------------------- #

@tdnary.make_nary_op
def grad(fun, x):  

    print(f"\ngrad: {fun}, {x}")
  
    return ReverseDiffOp(fun, x).grad(1)




# --- Derivative ------------------------------------------------------------ #

@tdnary.make_nary_op
def deriv(fun, x):

    return ForwardDiffOp(fun, x).grad(1)




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential operator interface --------------------------------------- #

class DiffOp(abc.ABC):

   @abc.abstractmethod
   def evaluate(self):
       pass

   @abc.abstractmethod
   def grad(self, seed):
       pass

   @abc.abstractmethod
   def evaluate_and_grad(self, seed):
       pass

 


# --- Forward differential operator ----------------------------------------- #

class ForwardDiffOp(DiffOp): 

   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def _compute(self, seed):

       with tdgraph.Graph(self._fun, self._x) as graph:
          top_node = graph.build(tdnode.ForwardRootGate(seed))   

       return top_node.reduce(), top_node.grad()


   def evaluate(self, seed=None):

       if seed is None:
          seed = 1

       return self._compute(seed)[0]


   def grad(self, seed):

       return self._compute(seed)[1]


   def evaluate_and_grad(self, seed):

       return self._compute(seed)




# --- Reverse differential operator ----------------------------------------- #

class ReverseDiffOp(DiffOp): 

   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   @tdutil.cacheable
   def _compute(self):

       with tdgraph.Graph(self._fun, self._x) as graph:
          top_node = graph.build(tdnode.ReverseRootGate())    

       return top_node.reduce(), Backprop(top_node)


   def evaluate(self):

       return self._compute()[0]


   def grad(self, seed):

       backprop = self._compute()[1]
       return backprop(seed)


   def evaluate_and_grad(self, seed):

       val, backprop = self._compute()
       return val, backprop(seed)




###############################################################################
###                                                                         ###
###  Backpropagation through the computation graph.                         ###
###                                                                         ###
###############################################################################


# --- Child-node counter ---------------------------------------------------- #

class ChildCount:

   def __init__(self, top_node):

       self._top_node = top_node
       self._count    = {}
       self._pool     = None

       self._last_visited = None 
 

   def visit(self, node):

       try:
           self._count[node] += 1
       except KeyError:
           self._count[node] = 1

       self._last_visited = node
       return self


   def add(self, nodes):

       if self._count.get(self._last_visited) == 1:
          self._pool.extend(nodes)

       return self


   def compute(self):

       self._pool = [self._top_node]

       while self._pool:

          node = self._pool.pop()
          node.add_to_childcount(self)

       return self


   def iterate(self):

       return iter(self._count.items())


   def toposort(self):

       return TopoSort(self._count, self._top_node)

       


# --- Topological sort ------------------------------------------------------ #

class TopoSort:

   def __init__(self, count, top_node):

       self._top_node = top_node
       self._count    = dict(count)
       self._pool     = None


   def add(self, node):

       self._count[parent] -= 1 

       if self._count[parent] == 0:
          self._pool.append(parent) 

       return self


   def iterate(self):

       self._pool = [self._top_node]

       while self._pool:

          node = self._pool.pop()
          yield node

          node.add_to_toposort(self)




# --- Create a topologically sorted iterator over the computation graph ----- #

def toposort(top_node):
    return (
            ChildCount(top_node).compute()
                                .toposort()
                                .iterate()
           )




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum:

   def __init__(self):

       self._map  = {}
       self._last = None


   def result(self):

       return self._last


   def push(self, node, grad):

       self._map[node] = grad
       return self


   def pop(self, node):

       # FIXME # print(f"\nGradAccum.pop(): node = {node}, map = {self._map}")

       self._last = self._map.pop(node)
       return self._last


   def accumulate(self, node, grad):

       # FIXME # print(f"\nGradAccum.accumulate(), BEFORE: node = {node}, grad = {grad}, map = {self._map}")

       self._map[node] = tdmanip.add_grads(self._map.get(node), grad)

       # FIXME # print(f"\nGradAccum.accumulate(), AFTER: node = {node}, grad = {grad}, map = {self._map}")
       return self




# --- Backpropagation ------------------------------------------------------- # 

class Backprop:

   def __init__(self, top_node): 

       self._top_node = top_node


   def __call__(self, seed):

       grads = GradAccum()  
       grads.push(self._top_node, seed)

       for node in toposort(self._top_node): 
           node.accumulate_parent_grads(grads)

       return grads.result()











































































