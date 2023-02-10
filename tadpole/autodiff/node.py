#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

from functools import reduce

import tadpole.autodiff.util     as tdutil   
import tadpole.autodiff.manip    as tdmanip        
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.adjoints as tda


"""

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

       return logic_eq(self, other)


   def __hash__(self):

       return id(self)


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

       return logic_eq(self, other)


   def __hash__(self):

       return id(self)
       

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

       return gate_eq(self, other)


   def __hash__(self):

       return id(self)


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

       return gate_eq(self, other)


   def __hash__(self):

       return id(self)


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
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################






   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args)(self._parent_grads())


   def gate(self, fun):

       jvps = self._apply(tda.jvpmap.get(fun))

       return ForwardGate(
                          self._parents, 
                          fun, 
                          reduce(tdmanip.add_grads, jvps, None)
                         )
"""


"""

gate = ReverseGate(fun, args, out) 

--> where fun  is AdjointableFun
--> where args is ConcatArgs


class AdjointableFun:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)

  
   def vjp(self, *args, **kwargs): 

       return tda.vjpmap.get(self._diff_fun)(*args, **kwargs)


   def jvp(self, *args, **kwargs): 

       return tda.jvpmap.get(self._diff_fun)(*args, **kwargs)
 


--> in gate: 

adj     = self._fun.adjoint(tda.vjpmap)(self._args.adxs(), self._out, *self._args.deshelled())
parents = self._args.parents()


NB. IF args is too fat, we can use Adapter pattern to introduce stripped down args, e.g. AdjointArgs
with only the methods that are needed, like adxs, deshelled, parents...

Maybe there is a different way to name it?


--> downside: fat param args! introduce this instead:

op   = AdjointOp(fun, adxs, out, *args)
gate = ReverseGate(parents, op)
vjp  = self._op.vjp() # in gate


class AdjointOp:

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


   def vjp(self):

       return self._fun.vjp(self._adxs, self._out, *self._args)


   def jvp(self):

       return self._fun.jvp(self._adxs, self._out, *self._args)


--> now we just need to get info from Parents:

    if parents.reverse():
       return ReverseGate(parents, fun, adjoint)

    if parents.forward():
       return ForwardGate(parents, fun, adjoint)


--> make a factory:

def make_gate(parents, fun, adjoint):

    return {
            Reverse: ReverseGate, 
            Forward: ForwardGate, 
           }[parents.type()](parents, fun, adjoint)


UTILIZE FACTORY MAPS HERE (see your notes)!

We collect parent nodes from args:

--> ctr Reverse or Forward parents depending on the type?
--> Q: Actually, if we no longer have Reverse/Forward nodes, how can we tell the type?
--> A: we'll need to collect gate info...

node.attach()
node.trace()
node.grads()

Use node.attach(train) for that! Collect type info as well.
If all nodes are Reverse, can we infer this and create a ReverseGate w/o checking the gate/logic types?

But even in a statically typed language, doing ReverseGate(parents, fun, adjoint) we cannot tell the logic behind parents,
cuz they all have the same type = Node!

Should we let: 

node.logic()    --> Reverse/Forward, which retrieves the hidden logic from an encapsulated Gate.
parents.logic() --> combine node.logic() of all nodes (or throw), except we should make sure about this when creating Parents...

logic can be a str, or it can be a factory, 
e.g. Reverse.gate(parents, fun, adjoint), 
e.g. Reverse.node(source, layer, parents, fun, adjoint)


What can Logic do?

logic = parents.logic() // or // logic = node.logic()
gate  = logic.gate(parents, fun, adjoint)

So Gate has parents, fun, ..., but also knows .grads() impl and adjmap type
--> all the FW/RV logic must be handled by Gate!


*** USE THIS !!! ***

--> altway: let def gate.flow(self): 
                    return Flow(lambda args: self.__class__(args), "reverse")

--> flows compare equal if they have the same str!

def parents.flow(self):
    flow, = set([p.flow() for p in parents])
    return flow


flow = parents.flow()
gate = flow.create(parents, fun, adjoint)

-- VS --

gate = parents.next_gate(fun, adjoint)

// in which case we could decorate FlowingParents(parents) with .next_gate() method

-- VS --

def make_gate(parents, fun, adjoint):

    flow, = set([p.flow() for p in parents])
    return flow.create(parents, fun, adjoint)



#############################################################################################
"""





"""


jvp  = tda.jvpmap.get(fun)(self._adxs, self._out, *self._args)
gate = ForwardGate(self._parents, fun, vjp)


vjp  = tda.vjpmap.get(fun)(self._adxs, self._out, *self._args)
gate = ReverseGate(self._parents, fun, vjp)



Can we isolate this logic to Gates?

-->  def adjoint(adjmap, fun):
         return adjmap.get(fun)(self._adxs, self._out, *self._args) # FIXME could also just pass (self._adxs, self._out, *self._args) to gate

and pass it to gate:

gate = ForwardGate(self._parents, fun, adjoint)
adj  = adjoint(tda.jvpmap, self._fun) # in gate


--> Altway: define AdjointFun class?

adjoint = AdjointFun(adxs, out, *args)
adj     = adjoint.map(tda.jvpmap, self._fun) # in gate


--> Can we avoid FunWithGate?


class FunWithAdjoint:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)


   def adjoint(self, adjmap):

       return adjmap.get(self._diff_fun) 


   def adjoint(self, adxs, out, *args):

       return AdjointFun(adxs, out, *args)




Consider this:

class ActivePack(Pack):

   def __init__(self, source, layer, logic):

       self._source = source
       self._layer  = layer
       self._logic  = logic


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)

       return tdnode.make_node(
                               source, 
                               self._layer, 
                               fun.gate(self._logic) 
                              )



class FunWithGate:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)


   def gate(self, logic):

       return logic.gate(self._diff_fun)



class ReverseLogic(Logic):

   def __init__(self, parents, adxs, out, args):

       self._parents = parents
       self._adxs   = adxs
       self._out    = out
       self._args   = args


   def gate(self, fun):

       vjp = self._apply(tda.vjpmap.get(fun)) 

       return ReverseGate(
                          self._parents, 
                          fun, 
                          vjp
                         )

"""


class LogicSwitch(Logic):

   def __init__(self, parents, adxs, out, args):

       self._parents = parents
       self._adxs    = adxs
       self._out     = out
       self._args    = args


   def __eq__(self, other):

       return logic_eq(self, other)


   def __hash__(self):

       return id(self)
       

   def _apply(self, fun):

       return fun(self._adxs, self._out, *self._args)


   def gate(self, fun):

       vjp = self._apply(tda.vjpmap.get(fun)) 

       return ReverseGate(
                          self._parents, 
                          fun, 
                          vjp
                         )

   def node(self):








class Nodal(abc.ABC):

   @abc.abstractmethod
   def tovalue(self):
       pass

   @abc.abstractmethod
   def attach(self, train):
       pass

   @abc.abstractmethod
   def grads(self, grads):
       pass

   @abc.abstractmethod
   def tosort(self, sort):
       pass





class Node(Nodal):

   def __init__(self, source, layer, gate): 
                                            
       self._source = source              
       self._layer  = layer 
       self._gate   = gate


   def __eq__(self, other):

       return all((
                   type(self)       == type(other),
                   self._layer      == other._layer,
                   id(self._source) == id(other._source),
                   id(self._gate)   == id(other._gate),
                 ))


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source.tovalue()


   def attach(self, train):

       return (
               train.with_node(self)
                    .with_meta(self._source, self._layer)
              )

   def grads(self, grads):

       return self._gate.grads(self, grads)


   def tosort(self, sort): 

       return self._gate.tosort(self, sort)




"""

ForwardGate.grads(grads) needs no seed input, but ReverseGate.grads(seed, grads) does!

grads.pop(seed): ForwardNode doesn't have a seed yet, ReverseNode does!

Our options:

1) Pass just grads and let Gate retrieve seed from it (how to do it with ReverseGate? we need node input...)

2) Pass the owning node to gate.grads(node, grads)

3) Give seed a different meaning: 




"""




class ForwardGate(Gate):

   def __init__(self, parents, fun, jvp):

       self._parents = parents
       self._fun     = fun
       self._jvp     = jvp


   def __eq__(self, other):

       return all((
                   type(self)    == type(other),
                   self._parents == other._parents,
                   self._fun     == other._fun,
                 ))


   def __hash__(self):

       return id(self)


   def grads(self, node, grads): 

       seed = tuple(map(grads.pop, self._parents))

       return grads.add(node, self._jvp(seed))

       
   def tosort(self, node, sort):

       return sort.add(node, self._parents)





class ReverseGate:

   def __init__(self, parents, fun, vjp):

       self._parents = parents
       self._fun     = fun
       self._vjp     = vjp


   def __eq__(self, other):

       return all((
                   type(self)    == type(other),
                   self._parents == other._parents,
                   self._fun     == other._fun,
                 ))


   def __hash__(self):

       return id(self)


   def grads(self, node, grads):

       seed = grads.pop(node)

       return grads.add(self._parents, self._vjp(seed))


   def tosort(self, node, sort):

       return sort.add(node, self._parents)






class GradSum:

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads
       self._last  = None


   def add(self, node, grads):

       self._grads[node] = reduce(tdmanip.add_grads, grads, None)

       return self


   def pop(self, node):

       self._last = self._grads.pop(node)
       return self._last








class Parents: # FIXME add methods to check FW/RV

   def __init__(self, *nodes):

       self._nodes = nodes


   def __eq__(self, other):

       return tuple(map(id, self._nodes)) 
           == tuple(map(id, other._nodes))


   def __hash__(self):

       return hash(self._nodes)


   def __len__(self):

       return len(self._nodes)


   def __contains__(self, node):

       return node in self._nodes


   def __iter__(self):

       return iter(self._nodes)


   def __getitem__(self, idx):

       return self._nodes[idx]
 





class GradAccum_:

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads
       self._last  = None


   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = tdmanip.add_grads(self._grads.get(node), grad)

       return self


   def push(self, node, grad):

       return self.add([node], [grad])


   def pop(self, node): # FIXME do we really need to pop, or just read?
 
       self._last = self._grads.pop(node)
       return self._last


   def result(self):

       return self._last # TODO find a way to cache this (shall we make a Dict cls, similar to Sequence?)




   
class GradAccum(GradAccum_):

   def __init__(self, node=None, grad=None):

       if   node is None:
            grads = {}
       else:
            grads = {node: grad}

       super().__init__(grads)



##########################################################################

class TopoSort:

   def __init__(self, parents):

       self._parents = parents
       self._count   = None
       self._pool    = None


   def add(self, node):

       if self._count[node] == 0:
          return self

       self._count[node] -= 1 

       if self._count[node] == 0:
          self._pool.append(node) 

       return self


   def count(self):

       self._count = collections.Counter(sum(self._parents.values(), tuple()))       


   def sort(self):

       

       







##########################################################################


# count = Children(parents).count(top, count)


class ChildCount:

   def __init__(self, count=None):

       if count is None:
          count = {}

       self._count = count


   def copy(self):

       return self.__class__(self._count.copy())
    

   def increment(self, node):

       try:
           self._count[node] += 1
       except KeyError:
           self._count[node] = 1

       return self


   def decrement(self, node):

       self._count[node] -= 1
       return self


   def zero(self, node):

       return self._count.get(node, 0) == 0


   def one(self, node):

       return self._count.get(node) == 1




class Children:

   def __init__(self, parents=None, count=None):

       if parents is None: parents = {}
       if count   is None  count   = ChildCount()

       self._parents = parents
       self._count   = count


   def parentsof(self, node):

       return self._parents.get(node, tuple())


   def add(self, node, parents):

       self._parents[node] = parents
       return self


   def count(self, topnode):

       count = self._count.copy()
       pool  = [topnode]

       while pool:

          node  = pool.pop()
          count = count.increment(node)

          if count.one(node):
             pool.extend(self.parentsof(node)) 

       return self.__class__(self._parents, count)


   def visit(self, collection, node): # FIXME the logic here is native to TopoSort

       count = self._count.copy()

       for parent in self.parentsof(node):

           if count.zero(node):
              continue

           count.decrement(node) 

           if count.zero(node):  
              collection.append(node)

       return collection

             



class TopoSort:

   def __init__(self, topnode, children=None):

       if children is None:
          children = Children()

       self._topnode  = topnode
       self._children = children
      

   def add(self, node, parents):

       self._children.add(node, parents)
       return self


   def sort(self):

       pool = []

       self._children.count(self._topnode)
       self._children.visit(pool, self._topnode)

       while pool:

          node = pool.pop()
          yield node

          self._children.visit(pool, node)


   def __iter__(self):

       return self.sort()

 


       
 













############################################################################




class ChildCount:

   def __init__(self, count):

       self._count = count
    

   def increment(self, node):

       try:
           self._count[node] += 1
       except KeyError:
           self._count[node] = 1

       return self


   def decrement(self, node):

       self._count[node] -= 1

       return self


   def zero(self, node):

       return self._count.get(node, 0) == 0


   def one(self, node):

       return self._count.get(node) == 1


   def count(self, parents, top): # FIXME this could be an external manipulator

       pool = [top]

       while pool:

          node = pool.pop()

          self.increment(node)

          if self.one(node):
             pool.extend(parents[node]) # self._parents.pop(node, tuple())

       return self



class TopoSort:

   def __init__(self, childcount, parents):

       self._childcount = childcount
       self._parents    = parents


   def add(self, node, parents):

       self._parents[node] = parents
       return self 


   def sort(self):

       (parents).count(top, count)

       self._childcount(self._parents)
       
 




class TopoSort:
       
   def __init__(self, top, childcount):

       self._top        = top
       self._childcount = childcount

       self._nodes   = []
       self._parents = []
       self._pool    = []


   def with_node(self, node): # FIXME should add node to childcount

       self._nodes.append(node)
       return self
       

   def with_parents(self, parents): # FIXME should add parents to childcount

       self._parents.append(parents)
       return self 


   def add(self, node):

       if self._childcount.zero(node):
          return self

       self._childcount.decrement(node) 

       if self._childcount.zero(node):  
          self._pool.append(node) 

       return self




   def add(self, top):

       self._childcount.add(top)

       


   def sort(self):

       self._childcount.add(childcount)


       nodes = []

       self.add(self._top)

       while self._pool:

          node = self._pool.pop()
          nodes.append(node)

          for parent in self._parents[node]: # FIXME add to pool
              self.add(parent)

       return nodes


   def __iter__(self):

       return iter(self.sort())



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################






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


       













"""
# --- Node interface -------------------------------------------------------- #

class Node(abc.ABC):

   @abc.abstractmethod
   def tovalue(self):
       pass

   @abc.abstractmethod
   def attach(self, train):
       pass




# --- Nodule: a node kernel ------------------------------------------------- #

class Nodule(Node):

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

       return nodule_eq(self, other)


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source.tovalue()


   def attach(self, train):

       return train.with_meta(self._source, self._layer)




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

       return node_eq(self, other)


   def __hash__(self):

       return id(self)


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

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.


class Point(Node): 

   def __init__(self, source):

       self._source = source
       self._layer  = tdgraph.minlayer()


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("source", self._source)
                     .with_data("layer",    self._layer)
                     .compile()
              )


   def __eq__(self, other):

       return point_eq(self, other)


   def __hash__(self):

       return id(self)


   def tovalue(self):

       return self._source


   def attach(self, train):

       return (
               train.with_node(self)
                    .with_meta(self, self._layer)
              )




# --- Create a node --------------------------------------------------------- #

def nodify(x):

    if isinstance(x, Node):
       return x

    return Point(x)




def make_node(source, layer, gate):

    nodule = Nodule(nodify(source), layer)

    return gate.nodify(nodule) 

"""


"""

###############################################################################
###                                                                         ###
###  Equality comparisons (treated as 'friends' of the actual objects)      ###
###                                                                         ###
###############################################################################


def ids(x):

    return tuple(map(id, x._args))




# --- Logic equality -------------------------------------------------------- #

def logic_eq(x, y):

    return all((
                type(x)    == type(y),
                x._parents == y._parents,
                x._adxs    == y._adxs,
                x._out     == y._out,   # FIXME make sure PointPack returns a Point!
                x._args    == y._args,  # FIXME make sure all args are Nodes!
              ))




# --- Gate equality --------------------------------------------------------- #

def gate_eq(x, y):

    return all((
                type(x)    == type(y),
                x._parents == y._parents,
                x._fun     == y._fun,
              ))




# --- Node equality --------------------------------------------------------- #

def node_eq(x, y):

    return all((
                type(x)   == type(y), 
                x._nodule == y._nodule,
                x._gate   == y._gate,               
              ))




# --- Nodule equality ------------------------------------------------------- #

def nodule_eq(x, y):

    return all((
                type(x)   == type(y),
                x._source == y._source,
                x._layer  == y._layer,
              ))




# --- Point equality -------------------------------------------------------- #

def point_eq(x, y):

    return all((
                type(x)       == type(y),
                id(x._source) == id(y._source),
                x._layer      == y._layer,
              ))

"""












