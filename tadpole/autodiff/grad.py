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

    op = diffop_reverse(fun, x)

    return op.grad(op.value().space().ones())




# --- Derivative ------------------------------------------------------------ #

@nary.nary_op
def derivative(fun, x):

    op = diffop_forward(fun, x)

    return op.grad(op.value().space().ones())




###############################################################################
###                                                                         ###
###  Differential operator                                                  ###
###                                                                         ###
###############################################################################


# --- Create reverse differential operator ---------------------------------- #

def diffop_reverse(fun, x):

    evalop     = EvalOp(fun, x)
    start, end = evalop.execute(an.GateReverse())

    return DifferentialOp(PropagationReverse(start, end))




# --- Create forward differential operator ---------------------------------- #

def diffop_forward(fun, x):

    evalop     = EvalOp(fun, x)
    start, end = evalop.execute(an.GateForward())

    return DifferentialOp(PropagationForward(start, end))




# --- Differential operator ------------------------------------------------- #

class DifferentialOp:

   def __init__(self, prop):

       self._prop = prop
 

   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("prop", self._prop)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._prop, other._prop)

       return bool(log)


   def value(self):

       return self._prop.apply(lambda x: ag.ArgsGen(x).deshelled()[0])
       

   def grad(self, seed):

       return self._prop.grads(seed).result()




###############################################################################
###                                                                         ###
###  Gradient propagation through AD computation graph                      ###
###                                                                         ###
###############################################################################


# --- Forward gradient propagation ------------------------------------------ #

class PropagationForward(Propagation):

   def __init__(self, start, end):

       self._start = start
       self._end   = end


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("start", self._start)
       rep.val("end",   self._end)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._start, other._start)
          log.val(self._end,   other._end)

       return bool(log)


   def apply(self, fun):

       return fun(self._end)

       
   def grads(self, seed):

       if not self._end.connected(self._start):
          return GradSum(seed, self._end.tonull())

       return self._end.grads(GradSum(seed))




# --- Reverse gradient propagation ------------------------------------------ #

class PropagationReverse(Propagation):

   def __init__(self, start, end):

       self._start = start
       self._end   = end


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("start", self._start)
       rep.val("end",   self._end)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._start, other._start)
          log.val(self._end,   other._end)

       return bool(log)


   def apply(self, fun):

       return fun(self._end)


   def grads(self, seed):

       print("GRADS: ", self._end, self._start, self._end.connected(self._start))

       if not self._end.connected(self._start):
          return GradAccum(self._start.tonull())

       grads = GradAccum({self._end: seed})

       for node in toposort(self._end): 
           grads = node.grads(grads)

       return grads




###############################################################################
###                                                                         ###
###  Function evaluation operator (builds AD computation graph)             ###
###                                                                         ###
###############################################################################


# --- Function evaluation operator ------------------------------------------ #

class EvalOp:

   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("fun",  self._fun)
       rep.ref("x",    self._x)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._fun,  other._fun)
          log.val(self._x,    other._x)

       return bool(log)


   def graph(self, root):

       return ag.Graph(root)


   def execute(self, root):

       with self.graph(root) as graph:
          start, end = graph.build(self._fun, self._x)  

       return start, end




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




