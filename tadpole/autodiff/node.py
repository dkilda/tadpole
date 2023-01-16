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
###  Logic of forward and reverse propagation, creates logic gates.         ###
###                                                                         ###
###############################################################################


# --- Logic interface ------------------------------------------------------- #

class Logic(abc.ABC):

   @abc.abstractmethod
   def gate(self, fun):
       pass




# --- Forward logic --------------------------------------------------------- #

class ForwardLogic(Logic):

   def __init__(self, parents, adxs, out, args):

       self._parents = parents
       self._adxs    = adxs
       self._out     = out
       self._args    = args


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("parents", self._parents)
                     .with_data("adxs",      self._adxs)
                     .with_member("out",     self._out)
                     .with_member("args",    self._args)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._parents == other._parents,
                   self._adxs    == other._adxs,
                   self._out     == other._out,  
                   self._args    == other._args,  
                 ))


   def _parent_grads(self):

       return tuple(p.grad() for p in self._parents)


   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args)(self._parent_grads())


   def gate(self, fun):

       jvps = self._apply(tda.jvpmap.get(fun))

       return ForwardGate(
                          self._parents, 
                          fun, 
                          reduce(tdmanip.add_grads, jvps, None)
                         )




# --- Reverse logic --------------------------------------------------------- #

class ReverseLogic(Logic):

   def __init__(self, parents, adxs, out, args):

       self._parents = parents
       self._adxs   = adxs
       self._out    = out
       self._args   = args


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("parents", self._parents)
                     .with_data("adxs",      self._adxs)
                     .with_member("out",     self._out)
                     .with_member("args",    self._args)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._parents == other._parents,
                   self._adxs    == other._adxs,
                   self._out     == other._out,  
                   self._args    == other._args,  
                 ))


   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args)


   def gate(self, fun):

       vjp = self._apply(tda.vjpmap.get(fun)) 

       return ReverseGate(
                          self._parents, 
                          fun, 
                          vjp
                         )




# --- Create logic ---------------------------------------------------------- #

def make_logic(parents, adxs, out, args):

    return parents[0].logic(parents[1:], adxs, out, args)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Gate interface -------------------------------------------------------- #

class Gate(abc.ABC):

   @abc.abstractmethod
   def nodify(self, nodule):
       pass




# --- Forward logic gate ---------------------------------------------------- #

class ForwardGate(Gate):

   def __init__(self, parents, fun, jvp):

       self._parents = parents
       self._fun     = fun
       self._jvp     = jvp


   def __repr__(self):
      
       return (
               tdutil.StringRep(self)
                     .with_member("parents", self._parents)
                     .with_member("fun",     self._fun)
                     .compile()
              )


   def __eq__(self, other):

       return all((
                   self._parents == other._parents, 
                   self._fun     == other._fun,
                 )) 


   def __hash__(self):

       return hash((self._parents, self._fun))


   def nodify(self, nodule):

       return ForwardNode(nodule, self)


   def grad(self):

       return self._jvp




# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(Gate):

   def __init__(self, parents, fun, vjp):

       self._parents = parents
       self._fun     = fun
       self._vjp     = vjp


   def __repr__(self):
      
       return (
               tdutil.StringRep(self)
                     .with_member("parents", self._parents)
                     .with_member("fun",     self._fun)
                     .compile()
              )


   def __eq__(self, other):

       return all((
                   self._parents == other._parents, 
                   self._fun     == other._fun,
                 )) 


   def __hash__(self):

       return hash((self._parents, self._fun))


   def nodify(self, nodule):

       return ReverseNode(nodule, self)


   def accumulate_parent_grads(self, seed, grads): 

       parent_grads = self._vjp(seed)

       for parent, grad in zip(self._parents, parent_grads): 
           grads.accumulate(parent, grad)

       return self


   def add_to_childcount(self, childcount):

       childcount.add(self._parents) 
       return self


   def add_to_toposort(self, toposort):

       for parent in self._parents:
           toposort.add(parent)

       return self




# --- Forward root logic gate ----------------------------------------------- #

class ForwardRootGate(ForwardGate):

   def __init__(self, seed):

       super().__init__(tuple(), None, seed)




# --- Reverse root logic gate ----------------------------------------------- #

class ReverseRootGate(ReverseGate):

   def __init__(self):

       super().__init__(tuple(), None, lambda g: ())




###############################################################################
###                                                                         ###
###  Nodule: a node kernel                                                  ###
###                                                                         ###
###############################################################################


# --- Nodule ---------------------------------------------------------------- #

class Nodule:

   def __init__(self, source, layer): 
                                            
       self._source = source              
       self._layer  = layer                 


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("source", self._source)
                     .with_data("layer",    self._layer)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._source == other._source,
                   self._layer  == other._layer,
                 ))


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source.tovalue()


   def attach(self, train):

       return train.with_meta(self._source, self._layer)




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node interface -------------------------------------------------------- #

class Node(abc.ABC):

   @abc.abstractmethod
   def tovalue(self):
       pass

   @abc.abstractmethod
   def attach(self, train):
       pass




# --- Active node ----------------------------------------------------------- #

class ActiveNode(Node):

   def __init__(self, nodule, gate):

       self._nodule = nodule
       self._gate   = gate


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("nodule", self._nodule)
                     .with_member("gate",   self._gate)
                     .compile()
              )


   def __eq__(self, other):

       return all((
                   self._nodule == other._nodule,
                   self._gate   == other._gate,
                 ))


   def __hash__(self):

       return hash((self._nodule, self._gate))


   def tovalue(self):

       return self._nodule.tovalue()


   def attach(self, train):

       return self._nodule.attach(train.with_node(self))




# --- Forward node ---------------------------------------------------------- #

class ForwardNode(ActiveNode): 

   def logic(self, others, adxs, source, args):

       return ForwardLogic((self, *others), adxs, source, args)


   def grad(self):

       return self._gate.grad()




# --- Reverse node ---------------------------------------------------------- #

class ReverseNode(ActiveNode): 

   def logic(self, others, adxs, source, args):

       return ReverseLogic((self, *others), adxs, source, args)


   def accumulate_parent_grads(self, grads): 

       seed = grads.pop(self)

       self._gate.accumulate_parent_grads(seed, grads)
       return self


   def add_to_childcount(self, childcount):

       childcount.visit(self)

       self._gate.add_to_childcount(childcount)
       return self


   def add_to_toposort(self, toposort):

       self._gate.add_to_toposort(toposort)
       return self




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class Point(Node):

   def __init__(self, source):

       self._source = source
       self._layer  = -1


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("source", self._source)
                     .with_data("layer",    self._layer)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._source == other._source,
                   self._layer  == other._layer,
                 ))


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source


   def attach(self, train):

       return (
               train.with_node(self)
                    .with_meta(self._source, self._layer)
              )


   def pluginto(self, funcall):

       return funcall.with_arg(self._source)




# --- Create a node --------------------------------------------------------- #

def nodify(x):

    if isinstance(x, Node):
       return x

    return Point(x)




def make_node(source, layer, gate):

    nodule = Nodule(nodify(source), layer)

    return gate.nodify(nodule) 






