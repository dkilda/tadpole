#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
from functools import reduce

import tadpole.util as util

import tadpole.autodiff.nary  as nary
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag


from tadpole.autodiff.types import (
   Node,
   Propagation,
   Traceable,
   Countable,
   Cumulative,
)




###############################################################################
###                                                                         ###
###  Specialized differential operators                                     ###
###                                                                         ###
###############################################################################


# --- Gradient -------------------------------------------------------------- #

@nary.nary_op
def gradient(fun, x):  

    op  = DifferentialOpReverse(fun, x)
    out = op.evaluate()

    return op.grad(out.space().ones())




# --- Derivative ------------------------------------------------------------ #

@nary.nary_op
def derivative(fun, x):

    op  = DifferentialOpForward(fun, x)
    out = op.evaluate()

    return op.grad(out.space().ones())




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential operator ------------------------------------------------- #

class DifferentialOp:

   def __init__(self, propagation, fun, x):

       self._prop = propagation
       self._fun  = fun
       self._x    = x


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("prop", self._prop)
       rep.val("fun",  self._fun)
       rep.ref("x",    self._x)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._prop, other._prop)
          log.val(self._fun,  other._fun)
          log.val(self._x,    other._x)

       return bool(log)


   #@util.cacheable
   def graphop(self):

       return self._prop.graphop(self._fun, self._x)


   def start(self):

       return self.graphop().start()


   def end(self):

       return self.graphop().end()


   def evaluate(self):

       return self.graphop().evaluate()


   def grad(self, seed):

       grads = self._prop.accum(self.start(), self.end(), seed)       
       return grads.result()




# --- Forward differential operator ----------------------------------------- #

class DifferentialOpForward(DifferentialOp):

   def __init__(self, fun, x):

       super().__init__(PropagationForward(), fun, x)




# --- Reverse differential operator ----------------------------------------- #

class DifferentialOpReverse(DifferentialOp):

   def __init__(self, fun, x):

       super().__init__(PropagationReverse(), fun, x)




###############################################################################
###                                                                         ###
###  Gradient propagation through the AD computation graph.                 ###
###                                                                         ###
###############################################################################


# --- Forward gradient propagation ------------------------------------------ #

class PropagationForward(Propagation):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def graphop(self, fun, x):

       return GraphOp(an.GateForward(), fun, x)


   def accum(self, start, end, seed):

       if not end.connected(start):
          return GradSum(seed, end.tonull())

       return end.grads(GradSum(seed))




# --- Reverse gradient propagation ------------------------------------------ #

class PropagationReverse(Propagation):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def graphop(self, fun, x):

       return GraphOp(an.GateReverse(), fun, x)


   def accum(self, start, end, seed):

       if not end.connected(start):
          return GradAccum(start.tonull())

       grads = GradAccum({end: seed})

       for node in toposort(end): 
           grads = node.grads(grads)

       return grads




###############################################################################
###                                                                         ###
###  Computation graph operator                                             ###
###                                                                         ###
###############################################################################


# --- Graph operator -------------------------------------------------------- #

class GraphOp:

   def __init__(self, root, fun, x):

       self._root = root
       self._fun  = fun
       self._x    = x


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("root", self._root)
       rep.val("fun",  self._fun)
       rep.ref("x",    self._x)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._root, other._root)
          log.val(self._fun,  other._fun)
          log.val(self._x,    other._x)

       return bool(log)


   #@util.cacheable
   def _build(self):

       with self.graph() as graph:
          start, end = graph.build(self._fun, self._x)  

       return start, end


   def graph(self):

       return ag.Graph(self._root)


   def start(self):

       return self._build()[0]


   def end(self):

       return self._build()[1]


   #@util.cacheable
   def evaluate(self):

       args = ag.ArgsGen(self.end())
       args = args.deshelled()

       return args[0]




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Child count ----------------------------------------------------------- #

class ChildCount(Traceable, Countable):

   def __init__(self, parents=None, count=None):

       if parents is None: parents = {}
       if count   is None: count   = {}

       self._parents = parents
       self._count   = count


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("nodes",   list(self._parents.keys()))
       rep.ref("parents", list(self._parents.values()))
       rep.val("count",   list(self._count.values()))

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._parents, other._parents)
          log.val(self._count,   other._count)

       return bool(log)


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


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("end", self._end)

       return str(rep)


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


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("traversal", self._traversal)
       rep.ref("count",     self._count)

       return str(rep)


   #@util.cacheable
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


# --- Gradient addition function (a shortcut) ------------------------------- #

def addgrads(x, y):

    return y.addto(x)




# --- Gradient summation ---------------------------------------------------- #

class GradSum(Cumulative):

   def __init__(self, seed, grads=None):

       if grads is None:
          grads = {}

       if not isinstance(grads, dict):
          grads = {None: grads}

       self._last  = None
       self._seed  = seed
       self._grads = grads


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("nodes", list(self._grads.keys()))
       rep.val("grads", list(self._grads.values()))

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._seed,  other._seed)
          log.val(self._grads, other._grads)

       return bool(log)


   def add(self, node, grads):

       self._grads[node] = reduce(addgrads, grads)  
       self._last        = node
       return self


   def pick(self, nodes):

       if not nodes:
          return (self._seed,)   

       return tuple(map(self._grads.__getitem__, nodes))


   def result(self):

       last = self._last
       if last is None:
          last = list(self._grads)[-1]

       return self._grads[last]




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum(Cumulative):

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       if not isinstance(grads, dict):
          grads = {None: grads}

       self._grads = grads


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("nodes", list(self._grads.keys()))
       rep.val("grads", list(self._grads.values()))

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._grads, other._grads)

       return bool(log)


   def _netgrad(self, node):

       return self._grads.get(node) 


   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = addgrads(self._netgrad(node), grad) 
      
       return self


   def pick(self, node): 
 
       grad = self._grads.pop(node)

       self._grads[None] = grad
       return grad


   def result(self): 

       try:
          return self._grads[None] 

       except KeyError:
          last = list(self._grads)[-1]
          return self._grads[last]




