#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.adjoint as tda
import tadpole.autodiff.util    as tdutil
import tadpole.autodiff.node    as tdnode








###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(abc.ABC):

   @abc.abstractmethod
   def layer(self):
       pass


   @abc.abstractmethod
   def adxs(self):
       pass


   @abc.abstractmethod
   def parents(self):
       pass


   @abc.abstractmethod
   def deshell(self):
       pass


   @abc.abstractmethod
   def deshelled(self):
       pass




# --- Concatenated arguments kernel ----------------------------------------- #

class ConcatArgs(Cohesive): 

   def __init__(self, nodes=None, sources=None, layers=None):

       if nodes   is None: nodes   = tdutil.Sequence()
       if sources is None: sources = tdutil.Sequence()
       if layers  is None: layers  = tdutil.Sequence()

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.val(self._nodes,   other._nodes)
       log.val(self._sources, other._sources)
       log.val(self._layers,  other._layers)

       return bool(log)


   def __hash__(self):

       return id(self)


   def add(self, node, source, layer):

       return self.__class__(
                             self._nodes.push(node), 
                             self._sources.push(source), 
                             self._layers.push(layer)
                            )

   @tdutil.cacheable
   def layer(self):

       return max(self._layers)


   @tdutil.cacheable
   def adxs(self):

       if self.layer() == minlayer():
          return tuple()

       return tuple(i for i, x in enumerate(self._layers) 
                                           if x == self.layer())

   @tdutil.cacheable
   def parents(self):

       nodes = [self._nodes[adx] for adx in self.adxs()]
       return tdnode.Parents(*nodes)


   @tdutil.cacheable
   def deshell(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return tuple(args)


   @tdutil.cacheable
   def deshelled(self):

       return Concat(self.deshell())




# --- Concatenation --------------------------------------------------------- #

class Concat(Cohesive):

   def __init__(self, args):

       self._args = args


   @tdutil.cacheable
   def execute(self):

       train = ConcatArgs() # FIXME rename ConcatArgs -> Args?

       for arg in self._args:
           train = arg.attach(train)

       return train


   def layer(self):

       return self.execute().layer()


   def adxs(self):

       return self.execute().adxs()


   def parents(self):

       return self.execute().parents()


   def deshell(self):

       return self.execute().deshell()


   def deshelled(self):

       return self.__class__(self.execute().deshell())





class Packable(abc.ABC):

   @abc.abstractmethod
   def apply(self, fun, out):
       pass





class Pack(Cohesive):

   def __init__(self, args):

       self._args = args

  
   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())

       
   def apply(self, fun, out): # FIXME input fun to ctor?

       if self.layer() == minlayer():
          return tdnode.Point(out)

       op   = tdnode.AdjointOp(fun, self.adxs(), out, self.deshell())
       gate = tdnode.make_gate(self.parents(), op) # FIXME should make_gate() be responsibility of Pack?

       return tdnode.Node(out, self.layer(), gate)




class NullPack(Cohesive):

   def __init__(self, args):

       self._args = args

  
   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())


   def apply(self, fun, out):

       if self.layer() == minlayer():
          return tdnode.Point(out)

       return out




def generate(first_, next_, stop_):

    x = first_

    for _ in itertools.count():

        yield x
        x = next_(x)

        if stop_(x):
           break  




class Envelope: # FIXME rename to Packs?

   def __init__(self, args, make_pack): 

       self._args      = args
       self._make_pack = make_pack


   def _generate(self):

       start = self._make_pack(self._args)

       return tdutil.generate(
                 start, lambda x: x.deshelled(), lambda x: x.adxs()))


   @tdutil.cacheable
   def _list(self):

       return list(self._generate())


   def others(self): 

       return iter(reversed(self._list()))


   def first(self, fun):

       args = self._list()[0].deshell()  
       out  = fun(*(arg.tovalue() for arg in args))

       return tdnode.Point(out)
           
           
        








class Fun:

   def __init__(self, fun, envelope):

       self._fun      = fun
       self._envelope = envelope


   def __call__(self, *args):

       envelope = self._envelope(args)
       out      = envelope.first(self._fun)

       for pack in envelope.others(): # FIXME rename to Packs impl Packs.__iter__?
           out = pack.apply(self, out)

       return out




def make_envelope(args, make_pack):

    return Envelope(Concat(args), make_pack)




def differentiable(fun):

    return Fun(fun, make_envelope(args, lambda x: Pack(x)))




def nondifferentiable(fun):

    return Fun(fun, make_envelope(args, lambda x: NullPack(x)))



















###############################################################################
###############################################################################
###############################################################################




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Define the minimum node layer ----------------------------------------- #

def minlayer():

    return -1




# --- Graph ----------------------------------------------------------------- #

class Graph:

   _layer = minlayer() 


   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, gate):

       start = tdnode.Node(self._x, type(self)._layer, gate) 

       return self._fun(start)


"""

###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Adjointable Function -------------------------------------------------- #

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




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       fun         = AdjointableFun(self, self._fun)
       concat_args = Active(ConcatArgs(args)) 

       return (
               concat_args.pack()
                          .pluginto(fun) 
              )




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       concat_args = Passive(ConcatArgs(args))

       return (
               concat_args.pack()
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


# --- Node train ------------------------------------------------------------ #

class NodeTrain:

   def __init__(self, nodes=None, meta=None):

       if nodes is None: nodes = tdutil.Sequence()       
       if meta  is None: meta  = tdutil.Sequence()

       self._nodes = nodes
       self._meta  = meta


   def __eq__(self, other):

       return all((
                   self._nodes == other._nodes,  
                   self._meta  == other._meta,
                 ))


   def __hash__(self):

       return id(self)


   def with_node(self, node):

       return self.__class__(self._nodes.push(node), self._meta)


   def with_meta(self, source, layer): 
 
       return self.__class__(self._nodes, self._meta.push((source, layer)))

       
   def concatenate(self):

       try:
          sources, layers = zip(*self._meta)
       except ValueError:
          sources, layers = tuple(), tuple()

       return ConcatArgsKernel(tuple(self._nodes), sources, layers)




# --- Node glue ------------------------------------------------------------- #

class NodeGlue:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return iter(map(tdnode.nodify, self._args))


   def concatenate(self):

       train = NodeTrain()

       for arg in self.iterate():
           train = arg.attach(train)

       return train.concatenate()




###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(abc.ABC):

   @abc.abstractmethod
   def layer(self):
       pass


   @abc.abstractmethod
   def adxs(self):
       pass


   @abc.abstractmethod
   def parents(self):
       pass


   @abc.abstractmethod
   def deshell(self):
       pass


   @abc.abstractmethod
   def deshelled(self):
       pass




# --- Packable interface ---------------------------------------------------- #

class Packable(abc.ABC):

   @abc.abstractmethod
   def pack(self):
       pass




# --- Concatenated arguments kernel ----------------------------------------- #

class ConcatArgsKernel(Cohesive): 

   def __init__(self, nodes, sources, layers):

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __eq__(self, other):

       return all((
                   self._nodes   == other._nodes,  
                   self._sources == other._sources, 
                   self._layers  == other._layers, 
                 ))


   def __hash__(self):

       return id(self)


   @tdutil.cacheable
   def layer(self):

       return max(self._layers)


   @tdutil.cacheable
   def adxs(self):

       if self.layer() == minlayer():
          return tuple()

       return tuple(i for i, x in enumerate(self._layers) 
                                           if x == self.layer())

   @tdutil.cacheable
   def parents(self):

       return tuple(self._nodes[adx] for adx in self.adxs())


   @tdutil.cacheable
   def deshell(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return tuple(args)


   @tdutil.cacheable
   def deshelled(self):

       glue = NodeGlue(self.deshell())

       return glue.concatenate()


###############################################################################
###############################################################################
###############################################################################





















class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       fun  = AdjointableFun(self, self._fun)
       pack = ActivePack(ConcatArgs(args)) # FIXME hidden dependency

       return pack.pluginto(fun) 
              



class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       pack = PassivePack(ConcatArgs(args)) # FIXME hidden dependency

       return pack.pluginto(self._fun)
              



class NodeGlue:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return iter(map(tdnode.nodify, self._args))


   def concatenate(self, train):

       for arg in self.iterate():
           train = arg.attach(train)

       return train.concatenate()



train = NodeTrain()
args  = NodeGlue(args).concatenate(train)



class ActivePack:

   def __init__(self, args):

       self._args = args


   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())


   def pluginto(self, fun):

       if not self.adxs():

          args = self.deshell()
          out  = fun(*(arg.tovalue() for arg in args))

          return Point(out)

       source = self.deshelled().pluginto(fun) # FIXME hidden dependency

       op   = AdjointOp(fun, self.adxs(), source, *self.deshell())
       gate = tdnode.make_gate(self.parents(), op) 

       return tdnode.Node(source, self.layer(), gate)
       


class PassivePack:

   def __init__(self, args):

       self._args = args


   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())


   def pluginto(self, fun):

       if not self.adxs():

          args = self.deshell()
          out  = fun(*(arg.tovalue() for arg in args))

          return Point(out)

       return self.deshelled().pluginto(fun) # FIXME hidden dependency






###############################################################################
###############################################################################
###############################################################################


# --- Concatenated arguments ------------------------------------------------ #

class ConcatArgs(Cohesive):

   def __init__(self, args):

       self._args = args


   def __eq__(self, other):

       return self._args == other._args


   def __hash__(self):

       return id(self)


   @property
   @tdutil.cacheable
   def _kernel(self):

       return NodeGlue(self._args).concatenate()


   def layer(self):

       return self._kernel.layer()


   def adxs(self):

       return self._kernel.adxs()


   def parents(self):

       return self._kernel.parents()


   def deshell(self):

       return self._kernel.deshell()


   def deshelled(self):

       return self.__class__(self._kernel.deshell())




# --- Packable concatenated arguments --------------------------------------- #

class PackableConcatArgs(Cohesive, Packable):

   def __init__(self, args):

       self._args = args


   def __eq__(self, other):

       return self._args == other._args


   def __hash__(self):

       return id(self)


   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())




# --- Decorator with the default packing behavior --------------------------- #

def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.adxs():
           return PointPack(self.deshell()) 
                                         
        source = self.deshelled().pack()

        return fun(self, source)

    return wrap




# --- Active concatenated arguments ----------------------------------------- #

class Active(PackableConcatArgs):

   @default_pack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.parents(),
                                 self.adxs(), 
                                 source,
                                 self.deshell() 
                                )  
 
       return ActivePack(source, self.layer(), logic)




# --- Passive concatenated arguments ---------------------------------------- #

class Passive(PackableConcatArgs):

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


   def __eq__(self, other):

       return all((
                   self._source == other._source,
                   self._layer  == other._layer,
                   self._logic  == other._logic,
                 ))


   def __hash__(self):

       return id(self)


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


   def __eq__(self, other):

       return self._source == other._source


   def __hash__(self):

       return id(self)


   def pluginto(self, fun): 

       return self._source.pluginto(fun)




# --- Point pack ------------------------------------------------------------ #

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.


class PointPack(Pack): 

   def __init__(self, nodes):

       self._nodes = nodes


   def __eq__(self, other):

       return self._nodes == other._nodes


   def __hash__(self):

       return id(self)


   def pluginto(self, fun):

       return fun(*(node.tovalue() for node in self._nodes))


"""

