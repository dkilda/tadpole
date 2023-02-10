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
  
    return ReverseDifferentialOp(fun, x).grad(1)




# --- Derivative ------------------------------------------------------------ #

@tdnary.make_nary_op
def deriv(fun, x):

    return ForwardDifferentialOp(fun, x).grad(1)




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential interface ------------------------------------------------ #

class Differential(abc.ABC):

   @abc.abstractmethod
   def graphop(self):
       pass

   @abc.abstractmethod
   def end(self):
       pass

   @abc.abstractmethod
   def evaluate(self):
       pass

   @abc.abstractmethod
   def accum(self):
       pass

   @abc.abstractmethod
   def grad(self):
       pass




# --- Forward differential operator ----------------------------------------- #

class ForwardDifferentialOp(Differential):

   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   @tdutil.cacheable
   def graphop(self):

       return GraphOp(tdnode.ForwardGate(), self._fun, self._x)


   @tdutil.cacheable
   def end(self):

       return self.graphop().end()


   @tdutil.cacheable
   def evaluate(self):

       return self.graphop().evaluate()


   def accum(self, seed):

       return GradSum({self._x: seed}) 


   def grad(self, seed):

       grads = self.accum(seed)
       grads = self.end().grads(grads)

       return grads.result(self.end())




# --- Reverse differential operator ----------------------------------------- #

class ReverseDifferentialOp(Differential):

   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   @tdutil.cacheable
   def graphop(self):

       return GraphOp(tdnode.ReverseGate(), self._fun, self._x)


   @tdutil.cacheable
   def end(self):

       return self.graphop().end()


   @tdutil.cacheable
   def evaluate(self):

       return self.graphop().evaluate()


   def accum(self, seed):

       return GradAccum({self.end(): seed}) 


   def grad(self, seed):

       grads = self.accum(seed)

       for node in toposort(self.end()): 
           grads = node.grads(grads)

       return grads.result()




# --- Graphable interface --------------------------------------------------- #

class Graphable(abc.ABC):

   @abc.abstractmethod
   def graph(self):
       pass

   @abc.abstractmethod
   def end(self):
       pass

   @abc.abstractmethod
   def evaluate(self):
       pass




# --- Graph operator -------------------------------------------------------- #

class GraphOp(Graphable):

   def __init__(self, root, fun, x):

       self._root = root
       self._fun  = fun
       self._x    = x


   def graph(self):

       return Graph(self._root)


   @tdutil.cacheable
   def end(self):

       with self.graph() as graph:
          end = graph.build(self._fun, self._x)   

       return end
         

   @tdutil.cacheable
   def evaluate(self):

       return self.end().tovalue()




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
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




# --- Traversable interface ------------------------------------------------- #

class Traversable(abc.ABC):

   @abc.abstractmethod
   def sweep(self, step):
       pass

   @abc.abstractmethod
   def apply(self, step):
       pass




# --- Traversal ------------------------------------------------------------- #

class Traversal(Traversable):

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




# --- Sortable interface ---------------------------------------------------- #

class Sortable(abc.ABC):

   @abc.abstractmethod
   def traverse(self):
       pass

   @abc.abstractmethod
   def __iter__(self):
       pass




# --- Topological sort ------------------------------------------------------ #

class TopoSort(Sortable):

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




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Cumulative interface -------------------------------------------------- #

class Cumulative(abc.ABC):

   @abc.abstractmethod
   def add(self, nodes, parents):
       pass

   @abc.abstractmethod
   def pop(self, node):
       pass

   @abc.abstractmethod
   def result(self, node):
       pass




# --- Gradient summation ---------------------------------------------------- #

class GradSum(Cumulative):

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads


   def add(self, node, grads):

       self._grads[node] = reduce(tdmanip.add_grads, grads, None)
       return self


   def pop(self, node):

       return self._grads.pop(node)


   def result(self, node):

       return self._grads.get(node)




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum(Cumulative):

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads


   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = tdmanip.add_grads(self.get(node), grad)

       return self


   def pop(self, node): 
 
       grad = self._grads.pop(node)

       self._grads[None] = grad
       return grad


   def result(self, node=None): 

       return self._grads.get(node)




