#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import functools

import tadpole.util as util 
from tadpole.util import TupleLike
       
import tadpole.autodiff.graph   as agraph
import tadpole.autodiff.map_jvp as jvpmap
import tadpole.autodiff.map_vjp as vjpmap




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint interface ----------------------------------------------------- #

class Adjoint(abc.ABC):

   @abc.abstractmethod
   def vjp(self, seed):
       pass

   @abc.abstractmethod
   def jvp(self, seed):
       pass




# --- Adjoint operator ------------------------------------------------------ #

class AdjointOp(Adjoint):

   def __init__(self, fun, adxs, out, args):

       if not isinstance(out, util.Outputs):
          out = util.Outputs(out)
 
       self._fun  = fun
       self._adxs = adxs 
       self._out  = out
       self._args = args


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("fun",  self._fun)
       rep.val("adxs", self._adxs)
       rep.val("out",  self._out)
       rep.val("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._fun,  other._fun)
       log.val(self._adxs, other._adxs)
       log.val(self._out,  other._out)
       log.val(self._args, other._args)

       return bool(log)


   def __hash__(self):

       return id(self)


   def _apply(self, fun):

       return fun(self._adxs, self._out.unpack(), *self._args)


   def vjp(self, seed):

       vjpfun = vjpmap.get(self._fun)

       return self._apply(vjpfun)(seed)


   def jvp(self, seed):

       jvpfun = jvpmap.get(self._fun)

       return self._apply(jvpfun)(seed)




# --- Null adjoint operator ------------------------------------------------- #

class NullAdjointOp(Adjoint):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def __eq__(self, other):

       return True


   def __hash__(self):

       return id(self)


   def vjp(self, seed):

       return tuple()


   def jvp(self, seed):

       return seed




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- FlowLike interface ---------------------------------------------------- #

class FlowLike(abc.ABC):

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def gate(self, parents, op):
       pass




# --- Flow ------------------------------------------------------------------ # 

class Flow(FlowLike):

   def __init__(self, name, fun):

       self._name = name
       self._fun  = fun


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("direction", self._name)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._name, other._name)

       return bool(log)


   def __hash__(self):

       return hash(self._name)


   def __add__(self, other):

       if not other:
          return self

       if self == other:
          return self

       raise ValueError((f"Flow.__add__: cannot add flows "
                         f"with different directions {self}, {other}"))


   def __radd__(self, other):

       return self.__add__(other)


   def gate(self, parents, op):

       return self._fun(parents, op)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- GateLike interface ---------------------------------------------------- #

class GateLike(abc.ABC):

   @abc.abstractmethod
   def flow(self):
       pass

   @abc.abstractmethod
   def trace(self, node, traceable):
       pass

   @abc.abstractmethod
   def grads(self, node, grads):
       pass




# --- Null logic gate ------------------------------------------------------- #

class NullGate(GateLike):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def flow(self):

       return Flow(
                   "NULL", 
                   lambda parents, op: self.__class__()
                  )


   def trace(self, node, traceable):

       return traceable


   def grads(self, node, grads):

       return grads




# --- Forward logic gate ---------------------------------------------------- #

class ForwardGate(GateLike):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = NullAdjointOp()

       self._parents = parents
       self._op      = op


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("parents", self._parents)
       rep.val("op",      self._op)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._parents, other._parents)
       log.val(self._op,      other._op)

       return bool(log)


   def __hash__(self):

       return id(self)


   def flow(self):

       return Flow(
                   "FORWARD", 
                   lambda parents, op: self.__class__(parents, op)
                  )


   def trace(self, node, traceable):

       return traceable.record(node, self._parents)


   def grads(self, node, grads): 

       for parent in self._parents:
           grads = parent.grads(grads)

       seed = grads.pick(self._parents)

       return grads.add(node, self._op.jvp(seed))




# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(GateLike):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = NullAdjointOp()

       self._parents = parents
       self._op      = op


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("parents", self._parents)
       rep.val("op",      self._op)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._parents, other._parents)
       log.val(self._op,      other._op)

       return bool(log)


   def __hash__(self):

       return id(self)


   def flow(self):

       return Flow(
                   "REVERSE", 
                   lambda parents, op: self.__class__(parents, op)
                  )


   def trace(self, node, traceable):

       return traceable.record(node, self._parents)


   def grads(self, node, grads):

       seed = grads.pick(node)

       try:
          print("\nGATE-1: ", node, seed._data, tuple(x._source._source._data for x in self._op.vjp(seed)), self._parents)
          #tuple(x._data for x in self._op.vjp(seed)), self._parents)
       except AttributeError:
          print("\nGATE-1: ", node, seed, tuple(self._op.vjp(seed)), self._parents)

       return grads.add(self._parents, self._op.vjp(seed))




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- NodeLike interface ---------------------------------------------------- #

class NodeLike(abc.ABC):

   @abc.abstractmethod
   def flow(self):
       pass

   @abc.abstractmethod
   def concat(self, concatenable):
       pass

   @abc.abstractmethod
   def trace(self, traceable):
       pass

   @abc.abstractmethod
   def grads(self, grads):
       pass




# --- Node ------------------------------------------------------------------ #

class Node(NodeLike):

   def __init__(self, source, layer, gate): 

       if not (layer > agraph.minlayer()):
          raise ValueError((f"Node: the input layer {layer} must be higher "
                            f"than the minimum layer {agraph.minlayer()}."))

       if not isinstance(source, NodeLike):
          source = Point(source)
                                            
       self._source = source              
       self._layer  = layer 
       self._gate   = gate


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("source", self._source)
       rep.val("layer",  self._layer)
       rep.ref("gate",   self._gate)

       return str(rep)


   def __eq__(self, other):

       if type(self) != type(other):
          return False

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._source, other._source)
       log.val(self._layer,  other._layer)
       log.val(self._gate,   other._gate)

       return bool(log)


   def __hash__(self):

       return id(self)


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


   def flow(self):

       return self._gate.flow()


   def trace(self, traceable): 

       return self._gate.trace(self, traceable)


   def grads(self, grads):

       return self._gate.grads(self, grads)




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class Point(NodeLike): 

   def __init__(self, source):

       self._source = source
       self._layer  = agraph.minlayer()
       self._gate   = NullGate()


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("source", self._source)

       return str(rep)


   def __eq__(self, other):

       if type(self) != type(other):
          return False

       log = util.LogicalChain()

       log.typ(self, other) 
       log.ref(self._source, other._source)

       return bool(log)


   def __hash__(self):

       return id(self)


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


   def flow(self):

       return self._gate.flow()


   def trace(self, traceable): 

       return self._gate.trace(self, traceable)


   def grads(self, grads):

       return self._gate.grads(self, grads)




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- Parental interface ---------------------------------------------------- #

class Parental(abc.ABC):

   @abc.abstractmethod
   def next(self, source, layer, op):
       pass




# --- Parents --------------------------------------------------------------- #

class Parents(Parental, TupleLike):  

   def __init__(self, *parents):

       if any(isinstance(parent, Point) for parent in parents):
          raise ValueError((f"Parents: parent nodes {parents} "
                            f"must not be Points. "))

       self._parents = parents

    
   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("parents", self._parents)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.ref(self._parents, other._parents)

       return bool(log)


   def __hash__(self):

       return hash(self._parents)


   def __len__(self):

       return len(self._parents)


   def __contains__(self, x):

       return x in self._parents


   def __iter__(self):

       return iter(self._parents)


   def __getitem__(self, idx):

       return self._parents[idx]
   

   def next(self, source, layer, op):

       flow = sum(parent.flow() for parent in self)
       return Node(source, layer, flow.gate(self, op))




