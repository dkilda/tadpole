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


# --- Forward differential operator ----------------------------------------- #

class ForwardDifferentialOp:

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

       return grads.get()




# --- Reverse differential operator ----------------------------------------- #

class ReverseDifferentialOp:

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

       return grads.get()




# --- Graph operator -------------------------------------------------------- #

class GraphOp:

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




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Gradient summation ---------------------------------------------------- #

class GradSum:

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads


   def add(self, node, grads):

       self._grads[node] = reduce(tdmanip.add_grads, grads, None)
       return self


   def pop(self, node):

       return self._grads.pop(node)


   def get(self, node): # FIXME rename get -> result?

       return self._grads.get(node)




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum:

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


   def get(self, node=None): # FIXME rename get -> result?

       return self._grads.get(node)












"""
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




class DifferentialOp:

   def __init__(self, fun, x, make_graph=None):

       if make_graph is None:
          def make_graph(fun, x):
              return tdgraph.Graph(fun, x)

       self._fun   = fun
       self._x     = x
       self._graph = make_graph


   @tdutil.cacheable
   def _compute(self):

       with self._graph(self._fun, self._x) as graph:
          end = graph.build()   # tdnode.ForwardRootGate(seed))   

       return end

 


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

"""





