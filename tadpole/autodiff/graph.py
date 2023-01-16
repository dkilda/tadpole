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

   _layer = -1 # FIXME put this constant in one place! MIN_LAYER = -1 --> e.g. MinLayer() class, with toint() method


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

   @abc.abstractmethod
   def layer(self):
       pass


   @abc.abstractmethod
   def adxs(self):
       pass


   @abc.abstractmethod
   def args(self):
       pass


   @abc.abstractmethod
   def parents(self):
       pass



# --- Packable interface ---------------------------------------------------- #

class Packable(abc.ABC):

   @abc.abstractmethod
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
 
       return self.__class__(self._nodes, self._meta.push((source, layer)))

       
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


"""

# --- Helper methods for glue ----------------------------------------------- #

def _nodify(x):

    if isinstance(x, tdnode.Node):
       return x

    return tdnode.Point(x)
"""




# --- Glue (without packing capability) ------------------------------------- #      

class Glue(Composite, Adhesive):

   def __init__(self, args):

       self._args = args


   @property
   @tdutil.cacheable
   def _engine(self):

       train = NodeTrain()

       for arg in self.iter():
           train = arg.attach(train)

       return train.concatenate()


   @tdutil.cacheable
   def iter(self):

       return iter(map(tdnode.nodify, self._args))


   def reduced(self):

       return self.__class__(self.args()) 


   def layer(self):

       return self._engine.layer()


   def adxs(self):

       return self._engine.adxs()


   def args(self):

       return self._engine.args()


   def parents(self):

       return self._engine.parents()




# --- Packing glue ---------------------------------------------------------- #

class PackingGlue(Composite, Adhesive, Packable):

   def __init__(self, glue):

       self._glue = glue


   def iter(self):

       return self._glue.iter()


   def reduced(self):

       return self.__class__(self._glue.reduced()) 


   def layer(self):

       return self._glue.layer()


   def adxs(self):

       return self._glue.adxs()


   def args(self):

       return self._glue.args()


   def parents(self):

       return self._glue.parents()




# --- Decorator with the default packing behavior --------------------------- #

def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.adxs():
           return PointPack(self.args()) # FIXME could add Point sources to PointPack directly!
                                         # So that PointPack consists of sources and not Points
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

class Passive(PackingGlue):

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

       return self._fun(*self._args.riter())




