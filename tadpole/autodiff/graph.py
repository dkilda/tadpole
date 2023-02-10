#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.adjoint as tda
import tadpole.autodiff.util    as tdutil
import tadpole.autodiff.node    as tdnode




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


   def __init__(self, root):

       self._root = root


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, fun, x):

       start = tdnode.Node(x, type(self)._layer, self._root) 

       return fun(start)




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable function wrap ------------------------------------------ #

class Differentiable:

   def __init__(self, fun, make_envelope):

       self._fun      = fun
       self._envelope = make_envelope


   def __call__(self, *args):

       return self._envelope(args).applywrap(self, self._fun)




# --- Non-differentiable function wrap -------------------------------------- #

class NonDifferentiable:

   def __init__(self, fun, make_envelope):

       self._fun      = fun
       self._envelope = make_envelope


   def __call__(self, *args):

       return self._envelope(args).apply(self._fun)




# --- Shorthand for a differentiable function wrap -------------------------- #

def differentiable(fun):

    return Differentiable(fun, lambda x: Envelope(x))




# --- Shorthand for a non-differentiable function wrap ---------------------- #

def nondifferentiable(fun):

    return NonDifferentiable(fun, lambda x: Envelope(x))




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- Compound interface ---------------------------------------------------- #

class Compound(abc.ABC):

   @abc.abstractmethod
   def concatenate(self):
       pass

   @abc.abstractmethod
   def pack(self):
       pass




# --- Helpers for Args ------------------------------------------------------- #

def nodify(x):

    if isinstance(x, tdnode.NodeLike):
       return x

    return tdnode.Point(x)




# --- Function arguments ---------------------------------------------------- #

class Args(tdutil.Tuple):

   @property
   def _args(self): 

       return self._xs


   def concat(self):

       concat = Concatenation() 
       args   = map(nodify, self._args)

       for arg in args:
           concat = arg.concat(concat)

       return concat

 
   def pack(self):

       return Pack(self.concat())




# --- Concatenable interface ------------------------------------------------ #

class Concatenable(abc.ABC):

   @abc.abstractmethod
   def attach(self, node, source, layer):
       pass




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




# --- Concatenation of nodes ------------------------------------------------ #

class Concatenation(Concatenable, Cohesive):

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


   def attach(self, node, source, layer):

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

       nodes = list(self._nodes)
       nodes = [nodes[adx] for adx in self.adxs()] 
       return tdnode.Parents(nodes)


   @tdutil.cacheable
   def deshell(self):

       args    = list(self._nodes)
       sources = list(self._sources)

       for adx in self.adxs():
           args[adx] = sources[adx] 

       return Args(args)




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Packed interface ------------------------------------------------------ #

class Packed(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

   @abc.abstractmethod
   def deshelled(self):
       pass

   @abc.abstractmethod
   def fold(self, funwrap, out):
       pass




# --- Argument pack (of concatenated nodes) --------------------------------- #

class Pack(Packed):

   def __init__(self, concat): 

       self._concat = concat


   @property
   def _layer(self):

       return self._concat.layer()


   @property
   def _adxs(self):

       return self._concat.adxs()


   @property
   def _args(self):

       return self._concat.deshell()


   @property
   def _parents(self):

       return self._concat.parents()


   def innermost(self):

       return self._layer == minlayer()


   def deshelled(self):

       return self.__class__(self._args.pack())

       
   def fold(self, funwrap, out): 

       if self.innermost(): 
          return tdnode.Point(out)

       op = tdnode.AdjointOp(funwrap, self._adxs, out, self._args)
       return self._parents.next(out, self._layer, op) 




# --- Enveloped interface --------------------------------------------------- #

class Enveloped(abc.ABC):

   @abc.abstractmethod
   def packs(self):
       pass

   @abc.abstractmethod
   def apply(self, fun):
       pass

   @abc.abstractmethod
   def applywrap(self, funwrap, out):
       pass




# --- Argument envelope ----------------------------------------------------- #

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.


class Envelope(Enveloped):

   def __init__(self, args):

       self._args = args

 
   def packs(self):

       return tdutil.Loop(
                          self._args.pack(),  
                          lambda x: x.deshelled(), 
                          lambda x: x.innermost()
                         )


   def apply(self, fun):

       args = self.packs().last().args()
       out  = fun(*(arg.tovalue() for arg in args))

       return tdnode.Point(out)     

       
   def applywrap(self, funwrap, fun):

       out = self.apply(fun)

       for pack in reversed(self.packs()):
           out = pack.fold(funwrap, out) 

       return out

















########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


def loop(first, next, stop):

    return tdutil.Loop(first, next, stop)



def packloop(args):

    start = Pack(Concat(args))
    return tdutil.Loop(start, lambda x: x.deshelled(), lambda x: x.adxs())) 








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



"""

Refactoring ConcatArgs (too many responsibilities?):

1) train = concat(args)
   cargs = ConcatArgs(train)
   layer = cargs.layer()


Train()
.add(node, src, layer)
.nodes()  { iter(nodes) }
.srcs()   { iter(srcs)  }
.layers() { iter(layers) }

could also let e.g.

.nodes(fun=None) { iter(map(fun, nodes)) }


2) train = Train()
   cargs = ConcatArgs(train) // has ctor(train=None) by default

   .add(node, src, layer) { 
                           train = self._train.add(node, src, layer)  
                           return self.__class__(train)
                          } // delegates to ._train! but creates an extra layer of immutable transforms... no good!



3) Since CArgs are only used to inject into Pack, we could just precalculate 
   (layer, adxs, args, parents) and inject them into Pack.

   Behavior like .deshell()/.deshelled() will be impl'ed by Pack.

   This way, CArgs would simply be a Pack factory. We could rename CArgs -> Concat, Pack -> CArgs?


Likewise, we can apply this to the Loop-Fun dilemma: make a factory which creates loop and 
diff/nondiff fun.




"""




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

       nodes = [self._nodes[adx] for adx in self.adxs()] # FIXME idx-access wont work with nodes = Sequence, use filter instead
       return tdnode.Parents(*nodes)


   @tdutil.cacheable
   def flow(self):

       flow, = set([parent.flow() for parent in self.parents()])
       return flow


   @tdutil.cacheable
   def deshell(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx] # FIXME idx-access wont work with sources = Sequence

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
       args  = map(tdnode.nodify, self._args)

       for arg in args:
           train = arg.attach(train)

       return train


   def layer(self):

       return self.execute().layer()


   def adxs(self):

       return self.execute().adxs()


   def parents(self):

       return self.execute().parents()


   def flow(self):

       return self.execute().flow()


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


   def flow(self):

       return self._args.flow()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())

       
   def apply(self, fun, out): # FIXME input fun to ctor?

       if self.layer() == minlayer():
          return tdnode.Point(out)

       op   = tdnode.AdjointOp(fun, self.adxs(), out, self.deshell())
       gate = self.flow().gate(self.parents(), op)

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




def generate(first_, next_, stop_): # FIXME could make class Generate, which also saves the endpt? 
                                    # Generate impls .__iter__(), .__reversed__(), and .end()
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


   @tdutil.cacheable
   def _first(self):

       return self._list()[0]


   def others(self): 

       return iter(reversed(self._list()))


   def first(self, fun):

       args = self._first().deshell()  
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












class Loop:

   def __init__(self, first, next, stop):

       self._first = first
       self._next  = next
       self._stop  = stop


   def _run(self):

       x = self._first

       for _ in itertools.count():

           yield x
           x = self._next(x)

           if self._stop(x):
              break


   @tdutil.cacheable
   def _list(self):

       return list(self._run())

       
   def __iter__(self):

       return self._run()


   def __reversed__(self):

       return iter(reversed(self._list()))


   def last(self):

       return next(reversed(self)) 




def loop(first, next, stop):

    return tdutil.Loop(first, next, stop)



def packloop(args):

    start = Pack(Concat(args))
    return tdutil.Loop(start, lambda x: x.deshelled(), lambda x: x.adxs())) 


       



"""
def generate(first_, next_, stop_): # FIXME could make class Generate, which also saves the endpt? 
                                    # Generate impls .__iter__(), .__reversed__(), and .end()
    x = first_

    for _ in itertools.count():

        yield x
        x = next_(x)

        if stop_(x):
           break  
"""







class Glue:

   def __init__(self, args):

       self._args = args


   def packs(self):

       start = Pack(Concat(self._args))
       return tdutil.Loop(start, lambda x: x.deshelled(), lambda x: x.adxs())) 


   def last(self):

       return self.packs().last()


   def pluginto(self, fun):

       args = self.packs().last().deshell()
       out  = fun(*(arg.tovalue() for arg in args))

       return tdnode.Point(out)



class Differentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       glue = Glue(args) # FIXME hidden dep! Should inject glue instead.

       out = glue.pluginto(self._fun)

       for pack in reversed(glue.packs()):
           out = pack.apply(self, out)
 
       return out



class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       glue = Glue(args) # FIXME hidden dep! Should inject glue instead.

       return glue.pluginto(self._fun)















class Fun:

   def __init__(self, fun):

       self._fun = fun


   def shells(self, *args):

       start = Pack(Concat(args))
       return tdutil.Loop(start, lambda x: x.deshelled(), lambda x: x.adxs())) 


   def __call__(self, *args):

       args = self.shells(*args).last().deshell()
       out  = self._fun(*(arg.tovalue() for arg in args))

       return tdnode.Point(out)     
        



class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       return self._fun(*args)



        
class Differentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       out = self._fun(*args)

       for shell in reversed(self._fun.shells(*args)):
           shell.apply(self, out)

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

