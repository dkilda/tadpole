#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import collections

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

       return top_node.tovalue(), top_node.grad()


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

       with tdgraph.Graph(self._fun, self._x) as graph: # FIXME input Graph instead!
          top_node = graph.build(tdnode.ReverseRootGate())  

       return top_node.tovalue(), Backprop(top_node)


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


# --- Traceable interface --------------------------------------------------- #

class Traceable(abc.ABC):

   @abc.abstractmethod
   def record(self, node, parents):
       pass




# --- Child count ----------------------------------------------------------- #

class ChildCount(Traceable):

   def __init__(self, parents=None, count=None):

       if parents is None: parents = {}
       if count   is None: count   = {}

       self._parents = parents
       self._count   = count


   def record(self, node, parents):

       self._parents[node] = parents
       return self


   def collect(self, node): 

       node.trace(self) 
       return self._parents[node]


   def increase(self, node):

       try:
           self._count[node] +=1
           return tuple()

       except KeyError:

           self._count[node] = 1
           return self._parents[node]


   def decrease(self, node):

       def fun(x):

           if self._count[x] == 0:
              return tuple()

           self._count[x] -= 1 

           if self._count[x] == 0:
              return (x,)

           return tuple()

       return sum(map(fun, self._parents[node]), tuple())




# --- Traversal ------------------------------------------------------------- #

class Traversal:

   def __init__(self, end):

       self._end = end


   def sweep(self, step): 

       pool = [self._end]

       while pool:

          node = pool.pop()
          yield node
          pool.extend(step(node))


   def apply(self, step):

       collections.deque(self.sweep(step), maxlen=0)
       return self   




# --- Topological sort ------------------------------------------------------ #

class TopoSort:

   def __init__(self, traversal, count):

       self._traversal = traversal
       self._count     = count


   @tdutil.cacheable
   def traverse(self):

       self._traversal.apply(self._count.collect)
       self._traversal.apply(self._count.increase)

       return self._traversal.sweep(self._count.decrease)


   def __iter__(self):

       return self.traverse()
       

        

# --- Create a topologically sorted iterator over the computation graph ----- #

def toposort(end):

    return TopoSort(Traversal(end), ChildCount())




# --- Gradient summation ---------------------------------------------------- #

class GradSum:

   def __init__(self, **grads):

       self._grads = grads


   def add(self, node, grads):

       self._grads[node] = reduce(tdmanip.add_grads, grads, None)
       return self


   def pop(self, node):

       return self._grads.pop(node)


   def get(self, node):

       return self._grads.get(node)




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum:

   def __init__(self, **grads):

       self._grads = grads


   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = tdmanip.add_grads(self.get(node), grad)

       return self


   def pop(self, node): 
 
       grad = self._grads.pop(node)

       self._grads[None] = grad
       return grad


   def get(self, node=None):

       return self._grads.get(node)




# --- Backpropagation ------------------------------------------------------- # 

class Backprop:

   def __init__(self, end): 

       self._end = end


   def __call__(self, seed):

       grads = GradAccum(**{self._end: seed})  

       for node in toposort(self._end): 
           grads = node.grads(grads)

       return grads.get()











































































