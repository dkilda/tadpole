#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.util     as tdutil   
import tadpole.autodiff.manip    as tdmanip        
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.adjoints as tda




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

   def __init__(self, parents, adxs, out, *args):

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

       print(f"\nReverseLogic: {self._parents}, {self._adxs}, {self._out}, {self._args}")


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


   def _parents(self):

       return _make_parents(self._parents, self._adxs)


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

    print(f"\nmake_logic: {parents}, {adxs}, {out}, {args}")

    return parents[0].logic(parents[1:], adxs, out, args)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Gate interface -------------------------------------------------------- #

class Gate(abc.ABC):

   @abc.abstractmethod
   def integrate_with(self, source, layer):
       pass




# --- Forward logic gate ---------------------------------------------------- #

class ForwardGate(Gate, Forward):

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


   def integrate_with(self, source, layer):

       return ForwardNode(Nodule(source, layer), self)


   def grad(self):

       return self._jvp




# --- Reverse logic gate ---------------------------------------------------- #

class ReverseGate(Gate, Reverse):

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


   def integrate_with(self, source, layer):

       return ReverseNode(Nodule(source, layer), self)


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
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node interface -------------------------------------------------------- #

class Node(abc.ABC):

   @abc.abstractmethod
   def reduce(self):
       pass

   @abc.abstractmethod
   def attach(self, train):
       pass




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


   def reduce(self):

       return self._source


   def attach(self, train):

       return (
               train.with_node(self)
                    .with_meta(self._source, self._layer)
              )


   def pluginto(self, funcall):

       return funcall.with_arg(self._source)




# --- Forward node ---------------------------------------------------------- #

class ForwardNode(Node, Forward): 

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


   def logic(self, others, adxs, source, args):

       return ForwardLogic((self, *others), adxs, source, args)


   def reduce(self):

       return self._nodule.reduce()


   def attach(self, train):

       return self._nodule.attach(train.with_node(self))


   def grad(self):

       return self._gate.grad()




# --- Reverse node ---------------------------------------------------------- #

class ReverseNode(Node, Reverse): 

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


   def logic(self, others, adxs, source, args):

       return ReverseLogic((self, *others), adxs, source, args)


   def reduce(self):

       return self._nodule.reduce()


   def attach(self, train):

       return self._nodule.attach(train.with_node(self))


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




# --- Create a node --------------------------------------------------------- #

def make_node(source, layer, gate):

    return gate.integrate_with(source, layer) 




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


   def reduce(self):

       return self._source


   def attach(self, train):

       return train.with_meta(self._source, self._layer)

















"""


# --- Jvp factory ----------------------------------------------------------- #

class JvpFactory:

   def __init__(self, fun):

       self._fun = fun


   def _jvpfun(self, *args, **kwargs):

       return tda.jvpmap.get(self._fun)(*args, **kwargs)


   def create(self, parent_gs, adxs, out, *args):

       jvps = self._jvpfun(adxs, out, *args)(parent_gs)

       return Jvp(self._fun, reduce(tdmanip.add_grads, jvps, None))




# --- Vjp factory ----------------------------------------------------------- #

class VjpFactory:

   def __init__(self, fun):

       self._fun = fun


   def _vjpfun(self, *args, **kwargs):

       return tda.vjpmap.get(self._fun)(*args, **kwargs)


   def create(self, adxs, out, *args):

       vjp = self._vjpfun(adxs, out, *args)

       return Vjp(self._fun, vjp)


"""



"""

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

   def __init__(self, parents, jvp):

       self._parents = parents
       self._jvp     = jvp


   def __repr__(self):

       out = tdutil.StringRep(self)
       out = out.with_member("parents", self._parents)
       out = out.with_member("jvp",     self._jvp)

       return out.compile()
       

   def __eq__(self, other):

       return self._parents == other._parents \
          and self._jvp     == other._jvp


   def __hash__(self):

       return id(self)


   def node(self, source, layer):

       return ForwardNode(UndirectedNode(source, self, layer))


   def grad(self):

       return self._jvp()




# --- Reverse gate ---------------------------------------------------------- #

class ReverseGate(Gate, Reverse): 

   def __init__(self, parents, vjp): 

       self._parents = parents
       self._vjp     = vjp


   def __repr__(self):

       out = tdutil.StringRep(self)
       out = out.with_member("parents", self._parents)
       out = out.with_member("vjp",     self._vjp)

       return out.compile()


   def __eq__(self, other):

       return self._parents == other._parents \
          and self._vjp     == other._vjp


   def __hash__(self):

       return id(self)


   def node(self, source, layer):

       return ReverseNode(UndirectedNode(source, self, layer))


   def next_input(self, others, adxs, args, source):

       return ReverseGateInputs((self, *others), adxs, args, source)


   def accumulate_parent_grads(self, grads): # FIXME make parents Nodes, not other Gates!

       parent_grads = self._vjp(grads.pop(self))

       for p, parent in enumerate(self._parents): 
           grads.accumulate(parent, parent_grads[p])

       return self


   def add_to_childcount(self, childcount):

       childcount.add(self, self._parents) # FIXME we must extract the owning node of each parent gate
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




# --- Gate parents ----------------------------------------------------------- #

class Inputs(abc.ABC):

   @abc.abstractmethod
   def transform(self, fun):
       pass




# --- Forward node parents --------------------------------------------------- #

class ForwardInputs(Inputs):

   def __init__(self, nodes, adxs, args, out): 

       self._nodes = nodes
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def __repr__(self):

       out = tdutil.StringRep(self)
       out = out.with_data("adxs",    self._adxs)
       out = out.with_member("nodes", self._nodes)
       out = out.with_member("args",  self._args)
       out = out.with_member("out",   self._out)

       return out.compile()
       

   def __eq__(self, other):

       return self._nodes == other._nodes \
          and self._adxs  == other._adxs  \
          and self._args  == other._args  \
          and self._out   == other._out


   def transform(self, fun):

       parents = tuple(self._nodes[adx] for adx in self._adxs) 

       jvp = JvpFactory(fun).create(
                                    (p.grad() for p in parents),
                                    self._adxs, 
                                    self._out, 
                                    *self._args
                                   )

       return ForwardGate(parents, jvp)




# --- Reverse node parents --------------------------------------------------- #

class ReverseInputs(Inputs):

   def __init__(self, nodes, adxs, args, out):

       self._nodes = nodes
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def __repr__(self):

       out = tdutil.StringRep(self)
       out = out.with_data("adxs",    self._adxs)
       out = out.with_member("nodes", self._nodes)
       out = out.with_member("args",  self._args)
       out = out.with_member("out",   self._out)

       return out.compile()
       

   def __eq__(self, other):

       return self._nodes == other._nodes \
          and self._adxs  == other._adxs  \
          and self._args  == other._args  \
          and self._out   == other._out


   def transform(self, fun):

       parents = tuple(self._nodes[adx] for adx in self._adxs) 

       vjp = VjpFactory(fun).create(
                                    self._adxs, 
                                    self._out, 
                                    *self._args
                                   )

       return ReverseGate(parents, vjp)




# --- Create node parents ---------------------------------------------------- #

def make_parents(nodes, adxs, args, out):

    return nodes[0].next_input(nodes[1:], adxs, args, out)




# --- Adhesive -------------------------------------------------------------- #

class Adhesive:

   def __init__(self, nodes):

       self._nodes = nodes


   def glue(self, sources, layers):

       return tdgraph.NodeGlue(
                               nodes, 
                               tdgraph.Sources(nodes, sources, layers)
                              )


"""




"""
###############################################################################
###                                                                         ###
###  Auxiliary code for Node objects.                                       ###
###                                                                         ###
###############################################################################


# --- Adhesive -------------------------------------------------------------- #

class Adhesive:

   def __init__(self, data=None):

       if data is None:
          data = tdutil.Stack()

       self._data = data


   def attach(self, source, layer):
 
       return self.__class__(self._data.push((source, layer)))
 

   def glue(self, nodes):

       sources, layers = zip(*self._data.riter())
 
       return tdgraph.NodeGlue(
                               nodes, 
                               tdgraph.Sources(nodes, sources, layers)
                              )




# --- Adhesive pipeline ----------------------------------------------------- #

class AdhesivePipeline:

   def __init__(self, nodes, nodules):

       self._nodes   = nodes
       self._nodules = nodules

       print(f"\nAdhesivePipeline: {nodes}, {nodules}")


   def glue(self):

       adhesive = Adhesive()

       for nodule in self._nodules:
           adhesive = nodule.attach_to(adhesive)
 
       return adhesive.glue(self._nodes)
"""




