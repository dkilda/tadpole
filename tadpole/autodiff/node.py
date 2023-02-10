#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

from functools import reduce

import tadpole.autodiff.util     as tdutil   
import tadpole.autodiff.manip    as tdmanip        
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.adjoints as tda




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


   def __eq__(self, other):

       return all((
                   type(self)    == type(other),
                   self._parents == other._parents,
                   self._op      == other._op,
                 ))


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

       seed = map(grads.pop, self._parents)

       return grads.add(node, self._op.jvp(seed))




# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(GateLike):

   def __init__(self, parents=None, op=None):

       if parents is None: parents = tuple()
       if op      is None: op      = NullAdjointOp()

       self._parents = parents
       self._op      = op


   def __eq__(self, other):

       return all((
                   type(self)    == type(other),
                   self._parents == other._parents,
                   self._op      == other._op,
                 ))


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

       seed = grads.pop(node)

       return grads.add(self._parents, self._op.vjp(seed))




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

   def __init__(self, fun, adxs, out, *args):
 
       self._fun  = fun
       self._adxs = adxs 
       self._out  = out
       self._args = args


   def __eq__(self, other):

       return all((
                   type(self) == type(other), 
                   self._fun  == other._fun,
                 ))


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

   def vjp(self, seed):

       return tuple()


   def jvp(self, seed):

       return seed




# --- Flow: defines the direction of propagation through AD graph ----------- # 

class Flow:

   def __init__(self, name, fun):

       self._name = name
       self._fun  = fun


   def __eq__(self, other):

       return all((
                   type(self) == type(other), 
                   self._name == other._name, 
                 ))


   def __hash__(self):

       return hash(self._name)


   def __repr__(self):

       return f"Flow: {self._name}"


   def __add__(self, other):

       if self == other:
          return self

       raise ValueError(
          f"Flow.__add__: cannot add unequal flows {self}, {other}")


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
   def attach(self, train):
       pass

   @abc.abstractmethod
   def trace(self, traceable):
       pass

   @abc.abstractmethod
   def grads(self, grads):
       pass


"""
   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.ref(self._source, other._source)
       log.ref(self._gate,   other._gate)
       log.val(self._layer,  other._layer)

       return bool(log)
"""



def refid(x):

    if isinstance(x, (tuple, list)):
       return map(id, x)

    return id(x)




class LogicalChain:

   def __init__(self):

       self._chain = []


   def _add(self, cond):

       self._chain.append(cond)
       return self


   def typ(self, x, y):

       return self._add(type(x) == type(y))
 

   def ref(self, x, y):

       return self._add(refid(x) == refid(y))
        

   def val(self, x, y):

       return self._add(x == y)


   def __bool__(self):

       return all(self._chain)


   def __repr__(self):

       rep = ReprChain()
       rep.typ(self)
       return str(rep)




class ReprChain:

   def __init__(self):

       self._str = ""


   def _add(self, xstr):

       self._str.append(xstr)
       return self

       
   def typ(self, x):

       return self._add(format_obj(x))

  
   def ref(self, name, x):

       return self._add(format_ref(name, refid(x)))
 

   def val(self, name, x):
 
       return self._add(format_val(name, x))   


   def __repr__(self):

       return f"{format_obj(self)}"


   def __str__(self):

       return vjoin(self._str)



# --- Node ------------------------------------------------------------------ #

class Node(NodeLike):

   def __init__(self, source, layer, gate): 
                                            
       self._source = source              
       self._layer  = layer 
       self._gate   = gate



   """
   def _signature(self):

       return Signature().type(self)
                         .vals(self._layer)
                         .ids(self._source, self._gate)



   def __eq__(self, other):

       log = LogicalChain()

       log(type(self)  == type(other))
       log(self._layer == other._layer)

       log(id(self._source) == id(other._source))
       log(id(self._gate)   == id(other._gate))

       return bool(log)
   """



   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.ref(self._source, other._source)
       log.ref(self._gate,   other._gate)
       log.val(self._layer,  other._layer)

       return bool(log)


   """
   def __eq__(self, other):

       if type(self)  == type(other)  and
          self._layer == other._layer and
          id(self._source) == id(other._source)
          id(self._gate)   == id(other._gate)

       return all((

          type(self) == type(other),
        
          self._layer == other._layer,

          id(self._source) == id(other._source),
          id(self._gate)   == id(other._gate),
       ))

      


       # return self._signature() == other._signature()
       return all((
          type(self) == type(other)
          id(self._source) ==  


       return all((
                   type(self)       == type(other),
                   self._layer      == other._layer,
                   id(self._source) == id(other._source),
                   id(self._gate)   == id(other._gate),
                 ))
       """

   def __hash__(self):

       return id(self)


   def flow(self):

       return self._gate.flow()


   def tovalue(self):

       return self._source.tovalue()


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


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


   @property
   def _layer(self):
       return tdgraph.minlayer()

   @property
   def _gate(self):
       return NullGate()


   def __eq__(self, other):

       return all((
                   type(self)       == type(other), 
                   id(self._source) == id(other._source),
                 )) 


   def __hash__(self):

       return id(self)


   def flow(self):

       return self._gate.flow()


   def tovalue(self):

       return self._source


   def concat(self, concatenable):

       return concatenable.attach(self, self._source, self._layer)


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

class Parents(tdutil.Tuple):

   @property
   def _nodes(self): 

       return self._xs


   def next(self, source, layer, op):

       flow = sum(parent.flow() for parent in self)
       return tdnode.Node(source, layer, flow.gate(self, op))


       

