#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.util as tdutil
import tadpole.autodiff.node as tdnode




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class Graph:

   _layer = -1


   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, gate):

       root = tdnode.make_node(self._x, type(self)._layer, gate) 

       return self._fun(root)




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Function with gate ---------------------------------------------------- #

class FunWithGate:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)


   def gate(self, logic):

       return logic.gate(self._diff_fun)




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       fun  = FunWithGate(self, self._fun)
       glue = Active(Glue(args)) 

       return (
               glue.pack()
                   .pluginto(fun) 
              )




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       glue = Passive(Glue(args))

       return (
               glue.pack()
                   .pluginto(self._fun)
              )




# --- Shorthand decorators for autodiff functions --------------------------- #

def differentiable(fun):

    return Differentiable(fun)




def nondifferentiable(fun):

    return NonDifferentiable(fun)




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Composite interface --------------------------------------------------- #

class Composite(abc.ABC):  

   @abc.abstractmethod
   def iter(self):
       pass


   @abc.abstractmethod
   def reduced(self):
       pass




# --- Adhesive interface ---------------------------------------------------- #

class Adhesive(abc.ABC):

   @abs.abstractmethod
   def layer(self):
       pass


   @abs.abstractmethod
   def adxs(self):
       pass


   @abs.abstractmethod
   def args(self):
       pass


   @abs.abstractmethod
   def parents(self):
       pass



# --- Packable interface ---------------------------------------------------- #

class Packable(abc.ABC):

   @abs.abstractmethod
   def pack(self):
       pass




# --- Node train ------------------------------------------------------------ #

class NodeTrain:

   def __init__(self, nodes=None, meta=None):

       if nodes is None: nodes = tdutil.Stack()       
       if meta  is None: meta  = tdutil.Stack()

       self._nodes = nodes
       self._meta  = meta


   def with_node(self, node):

       return self.__class__(self._nodes.push(node), self._meta)


   def with_meta(self, source, layer): 
 
       return self.__class__(self._nodes, self._meta.push((source, layer))

       
   def concatenate(self):

       sources, layers = zip(*self._meta.riter())

       return GlueEngine(self._nodes.tolist(), sources, layers)




# --- Glue engine ----------------------------------------------------------- #

class GlueEngine(Adhesive): 

   def __init__(self, nodes, sources, layers):

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   @tdutil.cacheable
   def layer(self):

       return max(self._layers)


   @tdutil.cacheable
   def adxs(self):

       if self.layer() == -1:
          return tuple()

       return tuple(i for i, x in enumerate(self._layers) 
                                           if x == self.layer())

   @tdutil.cacheable
   def args(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return tuple(args)


   @tdutil.cacheable
   def parents(self):

       return tuple(self._nodes[adx] for adx in self.adxs())




# --- Helper methods for glue ----------------------------------------------- #

def _nodify(x):

    if isinstance(x, Node):
       return x

    return Point(x)




# --- Glue (without packing capability) ------------------------------------- #

class Glue(Composite, Adhesive):

   def __init__(self, args):

       self._args = args


   @property
   @tdutil.cacheable  
   def _order(self):

       concat = NodeTrain()

       for arg in self.iter():
           train = arg.attach(train)

       return train.order()


   @tdutil.cacheable
   def iter(self):

       return iter(map(_nodify, self._args))


   def reduced(self):

       return self.__class__(self.args()) 


   def layer(self):

       return self._order.layer()


   def adxs(self):

       return self._order.adxs()


   def args(self):

       return self._order.args()


   def parents(self):

       return self._order.parents()




# --- Packing glue ---------------------------------------------------------- #

class PackingGlue(Composite, Adhesive, Packable):

   def __init__(self, inputs):

       self._inputs = inputs


   def iter(self):

       return self._inputs.iter()


   def reduced(self):

       return self.__class__(self.args()) 


   def layer(self):

       return self._inputs.layer()


   def adxs(self):

       return self._inputs.adxs()


   def args(self):

       return self._inputs.args()


   def parents(self):

       return self._inputs.parents()




# --- Decorator with the default packing behavior --------------------------- #

def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.adxs():
           return PointPack(self.args())

        source = self.reduced().pack()

        return fun(self, source)

    return wrap




# --- Active glue (a specific type of packing glue) ------------------------- #

class Active(PackingGlue):

   @default_pack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.parents(),
                                 self.adxs(), 
                                 source,
                                 self.args() 
                                )  
 
       return ActivePack(source, self.layer(), logic)




# --- Passive glue (a specific type of packing glue) ------------------------ #

class PassiveGlue(PackingGlue):

   @default_pack
   def pack(self, source):
    
       return PassivePack(source)




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Pack interface -------------------------------------------------------- #

class Pack(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, fun):
       pass




# --- Active pack ----------------------------------------------------------- #

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




# --- Passive pack ---------------------------------------------------------- #

class PassivePack(Pack):

   def __init__(self, source):

       self._source = source


   def pluginto(self, fun): 

       return self._source.pluginto(fun)




# --- Point pack ------------------------------------------------------------ #

class PointPack(Pack):

   def __init__(self, nodes):

       self._nodes = nodes


   def pluginto(self, fun):

       funcall = FunCall(fun)

       for node in self._nodes:
           funcall = node.pluginto(funcall)

       return funcall.execute() 




# --- Function call, operating on Point objects ----------------------------- #

class FunCall:

   def __init__(self, fun, args=None):

       if args is None:
          args = tdutil.Stack()

       self._fun  = fun
       self._args = args


   def with_arg(self, arg):

       return self.__class__(self._fun, self._args.push(arg))


   def execute(self):

       return self._fun(*self._args)







"""


# --- Glue interface -------------------------------------------------------- #

class Glue(abc.ABC):

   @abc.abstractmethod
   def pack(self, funcall):
       pass




# --- Argument glue --------------------------------------------------------- #

class ArgGlue(Glue):

   def __init__(self, args, argfilter):

       self._args   = args
       self._filter = argfilter


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("args",   self._args)
                     .with_member("filter", self._filter)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._args   == other._args,
                   self._filter == other._filter,
                 ))


   def pack(self, funcall=None):

       if funcall is None:
          funcall = FunCall()

       vals  = self._filter.vals(self._args)
       nodes = self._filter.nodes(self._args)

       print(f"\nArgGlue.pack(): {self._args}, {nodes}, {vals}")

       funcall = funcall.add(*vals)

       if len(nodes) > 0: 
          return nodes[0].glue(*nodes[1:]).pack(funcall)

       return EmptyPack(funcall)




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, args=None):

       if args is None:
          args = tdutil.Stack()

       self._args = args

       print(f"\nFunCall(): {list(self._args.riter())}")


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("args", self._args)
                     .compile()
              )
       

   def __eq__(self, other):

       return self._args == other._args


   def add(self, *args):

       xs = self._args

       for arg in args:
           xs = xs.push(arg)

       return self.__class__(xs)


   def execute(self, fun):

       print(f"\nFunCall.execute: {fun}, {list(self._args.riter())}")

       return fun(*self._args.riter())

"""

"""

###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node sources ---------------------------------------------------------- #

class Sources: 

   def __init__(self, nodes, sources, layers):

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("nodes",   self._nodes)
                     .with_member("sources", self._sources)
                     .with_data("layers",    self._layers)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._nodes   == other._nodes,
                   self._sources == other._sources,
                   self._layers  == other._layers,
                 ))


   @tdutil.cacheable
   def layer(self):
       return max(self._layers)


   @tdutil.cacheable
   def adxs(self):
       return tuple(i for i, x in enumerate(self._layers) 
                                      if x == self.layer())

   @tdutil.cacheable
   def args(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return args




# --- Node glue ------------------------------------------------------------- #

class NodeGlue(Glue):

   def __init__(self, nodes, sources): 

       self._nodes   = nodes
       self._sources = sources


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("nodes",   self._nodes)
                     .with_member("sources", self._sources)
                     .compile()
              )


   def __eq__(self, other):

       return all((
                   self._nodes   == other._nodes,
                   self._sources == other._sources,
                 ))


   def pack(self, funcall):

       glue    = ArgGlue(self._sources.args(), ArgFilterByNode())
       source  = glue.pack(funcall)

       logic = tdnode.make_logic(
                                 self._sources.parents(),
                                 self._sources.adxs(), 
                                 source,
                                 *self._sources.args(), # FIXME: args only incl the last unpacked nodes, but not previous vals! 
                                )                       #        must pass vals here too!

       print(f"\nNodeGlue.pack(): {self._nodes}, {self._sources.adxs()}, {self._sources.args()}")

       return NodePack(source, self._sources.layer(), logic)




# --- Point glue ------------------------------------------------------------ #

class PointGlue(Glue):

   def __init__(self, sources):

       self._sources = sources


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("sources", self._sources)
                     .compile()
              )
       

   def __eq__(self, other):

       return self._sources == other._sources


   def pack(self, funcall):

       glue   = ArgGlue(self._sources.args(), ArgFilterByPoint())
       source = glue.pack(funcall)  

       return PointPack(source, self._sources.layer())


"""
"""

###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Pack interface -------------------------------------------------------- #

class Pack(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, fun):
       pass




# --- Node pack ------------------------------------------------------------- #

class NodePack(Pack):

   def __init__(self, source, layer, logic):

       self._source = source
       self._layer  = layer
       self._logic  = logic


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("source", self._source)
                     .with_data("layer",    self._layer)
                     .with_member("logic",  self._logic)
                     .compile()
              )
       

   def __eq__(self, other):

       return all((
                   self._source == other._source,
                   self._layer  == other._layer,
                   self._logic  == other._logic,
                 ))


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)

       return tdnode.make_node(
                               source, 
                               self._layer, 
                               fun.gate(self._logic) 
                              )




# --- Point pack ------------------------------------------------------------ #

class PointPack(Pack):

   def __init__(self, source):

       self._source = source


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("source", self._source)
                     .compile()
              )
       

   def __eq__(self, other):

       return self._source == other._source


   def pluginto(self, fun):

       return self._source.pluginto(fun)

      


# --- Empty pack ------------------------------------------------------------ #

class EmptyPack(Pack):

   def __init__(self, funcall):

       self._funcall = funcall


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("funcall", self._funcall)
                     .compile()
              )


   def __eq__(self, other):

       return self._funcall == other._funcall


   def pluginto(self, fun):

       return self._funcall.execute(fun)

"""






"""

###############################################################################
###                                                                         ###
###  Argument glue and packs: glueing arguments into a single pack          ###
###                           for function calls.                           ###
###                                                                         ###
###############################################################################


# --- Argument filter ------------------------------------------------------- #

class ArgFilter:

   def __init__(self, mask):

       self._mask = mask


   def __repr__(self):

       return (
               tdutil.StringRep(self)
                     .with_member("mask", self._mask)
                     .compile()
              )


   def __eq__(self, other):

       return self._mask == other._mask


   def _execute(self, args, mask):

       return tuple(arg for arg in args if mask(arg))


   def vals(self, args):

       return self._execute(args, lambda x: not self._mask(x))  


   def nodes(self, args):

       return self._execute(args, lambda x: self._mask(x))  




# --- Argument filter with a mapping capability ----------------------------- #

class ArgFilterWithMap:

   def __init__(self, argfilter, fun=None):

       if fun is None:
          def fun(x): return x

       self._filter = argfilter
       self._fun    = fun


   def vals(self, args):

       return self._filter.vals(args)


   def nodes(self, args):

       return tuple(self._fun(node) for node in self._filter.nodes(args))




# --- Specialized argument filters: filter by node -------------------------- #

class ArgFilterByNode(ArgFilter):

   def __init__(self):

       super().__init__(lambda x: isinstance(x, tdnode.Node))




# --- Specialized argument filters: filter by point ------------------------- #

class ArgFilterByPoint(ArgFilterWithMap):

   def __init__(self):

       super().__init__(ArgFilterByNode(), lambda x: x.disconnect())

"""



