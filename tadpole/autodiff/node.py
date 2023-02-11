#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import functools

import tadpole.autodiff.util     as tdutil   
import tadpole.autodiff.manip    as tdmanip        
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.adjoints as tda




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
 
       self._fun  = fun
       self._adxs = adxs 
       self._out  = out
       self._args = args


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("fun",  self._fun)
       rep.val("adxs", self._adxs)
       rep.ref("out",  self._out)
       rep.val("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.val(self._fun,  other._fun)
       log.val(self._adxs, other._adxs)
       log.ref(self._out,  other._out)
       log.val(self._args, other._args)

       return bool(log)


   def __hash__(self):

       return id(self)


   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args)


   def vjp(self, seed):

       vjpfun = tda.vjpmap.get(self._fun)

       return self._apply(vjpfun)(seed)


   def jvp(self, seed):

       jvpfun = tda.jvpmap.get(self._fun)

       return self._apply(jvpfun)(seed)




# --- Null adjoint operator ------------------------------------------------- #

class NullAdjointOp(Adjoint):

   def __repr__(self):

       rep = tdutil.ReprChain()
       rep.typ(self)
       return str(rep)


   def vjp(self, seed):

       return tuple()


   def jvp(self, seed):

       return seed




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




# --- GateLike interface with default methods ------------------------------- #

class GateLikeDefault(GateLike):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = NullAdjointOp()

       self._parents = parents
       self._op      = op


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("parents", self._parents)
       rep.val("op",      self._op)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.val(self._parents, other._parents)
       log.val(self._op,      other._op)

       return bool(log)


   def __hash__(self):

       return id(self)


   def trace(self, node, traceable):

       return traceable.record(node, self._parents)


   def _flow(self, direction):

       return Flow(
                   direction, 
                   lambda parents, op: self.__class__(parents, op)
                  )




# --- Null logic gate ------------------------------------------------------- #

class NullGate(GateLike):

   def __repr__(self):

       rep = tdutil.ReprChain()
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

class ForwardGate(GateLikeDefault):

   def flow(self):

       return self._flow("FORWARD")


   def grads(self, node, grads): 

       for parent in self._parents:
           grads = parent.grads(grads)

       seed = grads.pop(self._parents)

       return grads.add(node, self._op.jvp(seed))




# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(GateLikeDefault):

   def flow(self):

       return self._flow("REVERSE")


   def grads(self, node, grads):

       seed = grads.pop(node)

       return grads.add(self._parents, self._op.vjp(seed))




# --- Flow: defines the direction of propagation through AD graph ----------- # 

class Flow:

   def __init__(self, name, fun):

       self._name = name
       self._fun  = fun


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("direction", self._name)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

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
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- NodeLike interface ---------------------------------------------------- #

class NodeLike(abc.ABC):

   @abc.abstractmethod
   def flow(self):
       pass

   @abc.abstractmethod
   def tovalue(self):
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

       if not (layer > tdgraph.minlayer()):
          raise ValueError((f"Node: the input layer {layer} must be higher "
                            f"than the minimum layer {tdgraph.minlayer()}."))

       source = typeconv(NodeLike)(source)
                                            
       self._source = source              
       self._layer  = layer 
       self._gate   = gate


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("source", self._source)
       rep.val("layer",  self._layer)
       rep.ref("gate",   self._gate)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.ref(self._source, other._source)
       log.val(self._layer,  other._layer)
       log.ref(self._gate,   other._gate)

       return bool(log)


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source.tovalue()


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


   def flow(self):

       return self._gate.flow()


   def trace(self, traceable): 

       return self._gate.trace(self, traceable)


   def grads(self, grads):

       return self._gate.grads(self, grads)




# --- Point (a disconnected node, only carries a value and no logic) -------- #

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.

class Point(NodeLike): 

   def __init__(self, source):

       self._source = source
       self._layer  = tdgraph.minlayer()
       self._gate   = NullGate()


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("source", self._source)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.ref(self._source, other._source)

       return bool(log)


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source


   def concat(self, concatenable):

       return concatenable.attach(self, self, self._layer)


   def flow(self):

       return self._gate.flow()


   def trace(self, traceable): 

       return self._gate.trace(self, traceable)


   def grads(self, grads):

       return self._gate.grads(self, grads)




# --- Parental interface ---------------------------------------------------- #

class Parental(abc.ABC):

   @abc.abstractmethod
   def next(self, source, layer, op):
       pass




# --- Parents --------------------------------------------------------------- #

class Parents(Parental, tdutil.TupleLike):  

   def __init__(self, parents):

       if not all(isinstance(parent, Node) for parent in parents):
          raise ValueError((f"Parents: all constructor inputs must be Nodes, "
                            f"but parents = {parents}"))

       self._parents = parents

    
   def __repr__(self):

       rep = ReprChain()

       rep.typ(self)
       rep.ref("parents", self._parents)

       return str(rep)


   def __eq__(self, other):

       log = LogicalChain()

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




###############################################################################
###                                                                         ###
###  Auxiliary methods for Node module.                                     ###
###                                                                         ###
###############################################################################


# --- Shorthand type conversions -------------------------------------------- #

def typeconv(typ):

    assert typ is NodeLike
    return tdutil.typeconv(NodeLike, Point)


def iterconv(typ):

    assert typ is NodeLike
    return tdutil.iterconv(NodeLike, Point)







       

