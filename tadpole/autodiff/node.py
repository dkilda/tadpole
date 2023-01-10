#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.autodiff.graph as tdg

from tadpole.autodiff.util            import StringRep
from tadpole.autodiff.adjoint_factory import JvpFactory, VjpFactory




###############################################################################
###                                                                         ###
###  Common code for nodes and gates                                        ###
###                                                                         ###
###############################################################################


# --- Forward propagation interface ----------------------------------------- #

class Forward(abc.ABC):

   @abc.abstractmethod
   def grad(self):
       pass




# --- Reverse propagation interface ----------------------------------------- #

class Reverse(abc.ABC):

   @abc.abstractmethod
   def accumulate_parent_grads(self, grads):
       pass

   @abc.abstractmethod
   def add_to_childcount(self, childcount):
       pass

   @abc.abstractmethod
   def add_to_toposort(self, toposort):
       pass




###############################################################################
###                                                                         ###
###  Gates of the autodiff circuit                                          ###
###                                                                         ###
###############################################################################


# --- Gate ------------------------------------------------------------------ #

class Gate(abc.ABC):

   @abc.abstractmethod
   def node(self, source, layer):
       pass

   @abc.abstractmethod
   def next_input(self, others, adxs, args, source):
       pass




# --- Forward gate ---------------------------------------------------------- #

class ForwardGate(Gate, Forward): 

   def __init__(self, parents, grad):

       self._parents = parents
       self._grad    = grad


   def _str(self):

       out = StringRep(self)
       out = out.with_member("parents", self._parents)
       out = out.with_member("grad",    self._grad)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._parents == other._parents


   def __hash__(self):

       return id(self)


   def node(self, source, layer):

       return ForwardNode(UndirectedNode(source, self, layer))


   def next_input(self, others, adxs, args, source):

       return ForwardGateInputs((self, *others), adxs, args, source)


   def grad(self):

       return self._grad




# --- Reverse gate ---------------------------------------------------------- #

class ReverseGate(Gate, Reverse): 

   def __init__(self, parents, vjp): 

       self._parents = parents
       self._vjp     = vjp


   def _str(self):

       out = StringRep(self)
       out = out.with_member("parents", self._parents)
       out = out.with_member("vjp",     self._vjp)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._parents == other._parents \
          and self._vjp     == other._vjp


   def __hash__(self):

       return id(self)


   def node(self, source, layer):

       return ReverseNode(UndirectedNode(source, self, layer))


   def next_input(self, others, adxs, args, source):

       return ReverseGateInputs((self, *others), adxs, args, source)


   def accumulate_parent_grads(self, grads):

       parent_grads = self._vjp(grads.pop(self))

       for p, parent in enumerate(self._parents): 
           grads.accumulate(parent, parent_grads[p])

       return self


   def add_to_childcount(self, childcount):

       childcount.add(self, self._parents)
       return self


   def add_to_toposort(self, toposort):

       for parent in self._parents:
           toposort.add(parent)

       return self




# --- Create the initial gates ---------------------------------------------- #

def make_forward_gate(seed):
    return ForwardGate(tuple(), seed)


def make_reverse_gate():
    return ReverseGate(tuple(), lambda g: ())




# --- Gate inputs ----------------------------------------------------------- #

class GateInputs(abc.ABC):

   @abc.abstractmethod
   def transform(self, fun):
       pass




# --- Forward gate inputs --------------------------------------------------- #

class ForwardGateInputs(GateInputs):

   def __init__(self, gates, adxs, args, out): 

       self._gates = gates
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def _str(self):

       out = StringRep(self)
       out = out.with_data("adxs",  self._adxs)
       out = out.with_member("gates", self._gates)
       out = out.with_member("args",  self._args)
       out = out.with_member("out",   self._out)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._gates == other._gates \
          and self._adxs  == other._adxs  \
          and self._args  == other._args  \
          and self._out   == other._out


   def transform(self, fun):

       parents = tuple(self._gates[adx] for adx in self._adxs)

       jvp = JvpFactory(fun).jvp(
                                 (p.grad() for p in parents),
                                 self._adxs, 
                                 self._out, 
                                 *self._args
                                )

       return ForwardGate(parents, jvp)




# --- Reverse gate inputs --------------------------------------------------- #

class ReverseGateInputs(GateInputs):

   def __init__(self, gates, adxs, args, out): 

       self._gates = gates
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def _str(self):

       out = StringRep(self)
       out = out.with_data("adxs",  self._adxs)
       out = out.with_member("gates", self._gates)
       out = out.with_member("args",  self._args)
       out = out.with_member("out",   self._out)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._gates == other._gates \
          and self._adxs  == other._adxs  \
          and self._args  == other._args  \
          and self._out   == other._out


   def transform(self, fun):

       parents = tuple(self._gates[adx] for adx in self._adxs)

       vjp = VjpFactory(fun).vjp(self._adxs, self._out, *self._args)

       return ReverseGate(parents, vjp)




# --- Create gate inputs ---------------------------------------------------- #

def make_gate_inputs(gates, adxs, args, out):

    return gates[0].next_input(gates[1:], adxs, args, out)




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #

class Node(abc.ABC):

   @abc.abstractmethod
   def reduce(self):
       pass

   @abc.abstractmethod
   def topoint(self):
       pass

   @abc.abstractmethod
   def glue(self, *others):
       pass




# --- Undirected node ------------------------------------------------------- #

class UndirectedNode(Node):

   def __init__(self, source, gate, layer):

       self._source = source
       self._gate   = gate
       self._layer  = layer


   def _str(self):

       out = StringRep(self)
       out = out.with_data("layer",    self._layer)
       out = out.with_member("source", self._source)
       out = out.with_member("gate",   self._gate)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._source == other._source \
          and self._gate   == other._gate   \
          and self._layer  == other._layer


   def __hash__(self):

       return id(self)


   def reduce(self):

       return self._source


   def topoint(self):

       return Point(self._source, self._layer)


   def glue(self, *others):

       nodes = (self, *others)

       sources = tuple(node._source for node in nodes) 
       gates   = tuple(node._gate   for node in nodes)
       layers  = tuple(node._layer  for node in nodes)

       return tdg.NodeGlue(tdg.Sources(nodes, sources, layers), gates)


   def visit(self, fun):

       return fun(self._gate) 




# --- Forward node ---------------------------------------------------------- #

class ForwardNode(Node, Forward): 

   def __init__(self, core):

       self._core = core


   def __str__(self):
 
       return str(self._core) 


   def __repr__(self):

       return repr(self._core)


   def __eq__(self, other):

       return self._core == other._core


   def __hash__(self):

       return hash(self._core)


   def reduce(self):

       return self._core.reduce()


   def topoint(self):

       return self._core.topoint()


   def glue(self, *others):

       return self._core.glue(*(other._core for other in others))


   def grad(self):

       return self._core.visit(lambda x: x.grad())




# --- Reverse node ---------------------------------------------------------- #

class ReverseNode(Node, Reverse): 

   def __init__(self, core):

       self._core = core


   def __str__(self):
 
       return str(self._core) 


   def __repr__(self):

       return repr(self._core)


   def __eq__(self, other):

       return self._core == other._core


   def __hash__(self):

       return hash(self._core)


   def reduce(self):

       return self._core.reduce()


   def topoint(self):

       return self._core.topoint()


   def glue(self, *others):

       return self._core.glue(*(other._core for other in others))


   def accumulate_parent_grads(self, grads):

       self._core.visit(lambda x: x.accumulate_parent_grads(grads))
       return self


   def add_to_childcount(self, childcount):

       self._core.visit(lambda x: x.add_to_childcount(childcount))
       return self


   def add_to_toposort(self, toposort):

       self._core.visit(lambda x: x.add_to_toposort(toposort))
       return self




# --- Point ----------------------------------------------------------------- #

class Point(Node):

   def __init__(self, source, layer):

       self._source = source
       self._layer  = layer


   def _str(self):

       out = StringRep(self)
       out = out.with_data("layer",    self._layer)
       out = out.with_member("source", self._source)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._source == other._source \
          and self._layer  == other._layer


   def __hash__(self):

       return id(self)


   def reduce(self):

       return self._source


   def topoint(self):

       return self


   def glue(self, *others):

       pts = (self, *others)

       sources = tuple(pt._source for pt in pts) 
       layers  = tuple(pt._layer  for pt in pts)

       return tdg.PointGlue(tdg.Sources(pts, sources, layers))




# --- Create a node --------------------------------------------------------- #

def make_node(source, gate, layer):

    return gate.node(source, layer)









