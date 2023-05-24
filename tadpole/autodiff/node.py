#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools

import tadpole.util          as util     
import tadpole.autodiff.misc as misc


from tadpole.autodiff.types import (
   AdjointOp, 
   Flow,
   Gate,
   Node,
   Parents,
)




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- General adjoint operator ---------------------------------------------- #

class AdjointOpGen(AdjointOp):

   def __init__(self, fun, adxs, out, args, kwargs=None):

       if kwargs is None:
          kwargs = {} 
 
       self._fun    = fun
       self._adxs   = adxs 
       self._out    = out
       self._args   = args
       self._kwargs = kwargs


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("fun",    self._fun)
       rep.val("adxs",   self._adxs)
       rep.val("out",    self._out)
       rep.val("args",   self._args)
       rep.val("kwargs", self._kwargs)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
          log.val(self._fun,    other._fun)
          log.val(self._adxs,   other._adxs)
          log.val(self._out,    other._out)
          log.val(self._args,   other._args)
          log.val(self._kwargs, other._kwargs)

       return bool(log)


   def __hash__(self):

       return id(self)


   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args, **self._kwargs)


   def vjp(self, seed):

       return self._apply(self._fun.vjp)(seed)


   def jvp(self, seed):

       return self._apply(self._fun.jvp)(seed)




# --- Null adjoint operator ------------------------------------------------- #

class AdjointOpNull(AdjointOp):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def __eq__(self, other):

       return type(self) == type(other)


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


# --- General flow ---------------------------------------------------------- # 

class FlowGen(Flow):

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

       if bool(log):
          log.val(self._name, other._name)

       return bool(log)


   def __hash__(self):

       return hash(self._name)


   def __add__(self, other):

       if not other:
          return self

       if self == other:
          return self

       raise ValueError(
          f"{type(self).__name__}.__add__: cannot add "
          f"flows with different directions {self}, {other}"
       )


   def __radd__(self, other):

       return self.__add__(other)


   def gate(self, parents, op):

       return self._fun(parents, op)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Null logic gate ------------------------------------------------------- #

class GateNull(Gate):

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       return str(rep)


   def __eq__(self, other):

       return type(self) == type(other)


   def flow(self):

       return FlowGen(
          "NULL", lambda parents, op: self.__class__()
       )


   def trace(self, node, traceable):

       return traceable


   def grads(self, node, grads):

       return grads




# --- Forward logic gate ---------------------------------------------------- #

class GateForward(Gate):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = AdjointOpNull()

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

       if bool(log):
          log.val(self._parents, other._parents)
          log.val(self._op,      other._op)

       return bool(log)


   def __hash__(self):

       return id(self)


   def flow(self):

       return FlowGen(
          "FORWARD", lambda parents, op: self.__class__(parents, op)
       )


   def trace(self, node, traceable):

       return traceable.record(node, self._parents)


   def grads(self, node, grads): 

       for parent in self._parents:
           grads = parent.grads(grads)

       seed = grads.pick(self._parents)

       return grads.add(node, self._op.jvp(seed))




# --- Reverse logic gate ---------------------------------------------------- #

class GateReverse(Gate):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = AdjointOpNull()

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

       if bool(log):
          log.val(self._parents, other._parents)
          log.val(self._op,      other._op)

       return bool(log)


   def __hash__(self):

       return id(self)


   def flow(self):

       return FlowGen(
          "REVERSE", lambda parents, op: self.__class__(parents, op)
       )


   def trace(self, node, traceable):

       return traceable.record(node, self._parents)


   def grads(self, node, grads):

       seed = grads.pick(node)
       """
       try:
          print("\n\nGATE.grads(): ", node, self._parents, [v._source for v in self._parents], seed, tuple(self._op.vjp(seed)), self._op._fun._fun.__name__, self._op._args)
       except AttributeError:
          print("\n\nGATE.grads(): ", node, self._parents, seed, tuple(self._op.vjp(seed)), self._op)
       """

       return grads.add(self._parents, self._op.vjp(seed))




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- General node ---------------------------------------------------------- #

class NodeGen(Node): 

   # --- Construction --- #

   def __init__(self, source, layer, gate): 
                                           
       self._source = source              
       self._layer  = layer 
       self._gate   = gate


   # --- Equality, hashing, representation --- #

   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("source", self._source)
       rep.val("layer",  self._layer)
       rep.ref("gate",   self._gate)

       return str(rep)


   def __hash__(self):

       return id(self)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
          log.val(self._source, other._source)
          log.val(self._layer,  other._layer)
          log.val(self._gate,   other._gate)

       return bool(log)


   # --- Node methods --- #

   def connected(self, other):

       if self._layer == misc.minlayer():
          return False

       if other._layer == misc.minlayer():
          return False 

       return self._layer == other._layer


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


   def flow(self):

       return self._gate.flow()


   def trace(self, traceable): 

       return self._gate.trace(self, traceable)


   def grads(self, grads):

       return self._gate.grads(self, grads)




# --- NodeScape: draws new nodes -------------------------------------------- #

class NodeScape:

   def __init__(self):

       self._types = {}


   def register(self, valtype, nodetype): 

       self._types[valtype]  = nodetype
       self._types[nodetype] = nodetype

       return self


   def _create(self, source, layer, gate):

       def _node_by_type(*typefuns):

           if  not typefuns:

               if layer > misc.minlayer():
                  raise NonDifferentiableTypeError(
                     f"Cannot differentiate wrt value of type {type(source)}."
                  )

               return NodeGen(source, layer, gate)

           try:
               typ = typefuns[0](type(source))
               return self._types[typ](source, layer, gate)

           except KeyError:
               return _node_by_type(*typefuns[1:])
           
       return _node_by_type(
                    lambda x: x, 
                    lambda x: str(x), 
                    lambda x: str(x).split("'")[1],
                    lambda x: str(x).split("'")[1].split(".")[-1]
                   ) 


   def node(self, source, layer, gate):

       if not (layer > misc.minlayer()):
          raise ValueError(
             f"{type(self).__name__}.node(): the input layer {layer} "
             f"must be higher than the minimum layer {misc.minlayer()}."
          )

       if not isinstance(source, Node):
          source = self.point(source)

       return self._create(source, layer, gate) 


   def point(self, source):

       return self._create(source, misc.minlayer(), GateNull()) 




# --- Non-differentiable type error ----------------------------------------- #

class NonDifferentiableTypeError(Exception):

   def __init__(self, value):
       self.value = value

   def __str__(self):
       return repr(self.value)




# --- A global instance of NodeScape and its access ports ------------------- #

_NODESCAPE = NodeScape()


def register(valtype, nodetype):

    return _NODESCAPE.register(valtype, nodetype)


def node(source, layer, gate):

    return _NODESCAPE.node(source, layer, gate)


def point(source):

    return _NODESCAPE.point(source)




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- General parents ------------------------------------------------------- #

class ParentsGen(Parents):  

   def __init__(self, *parents):

       if not all(parent.connected(parents[0]) for parent in parents):
          raise ValueError(
             f"{type(self).__name__}: all parent nodes {parents} "
             f"must be connected nodes. "
          )

       self._parents = parents
    

   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("parents", self._parents)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
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

       return node(source, layer, flow.gate(self, op))




