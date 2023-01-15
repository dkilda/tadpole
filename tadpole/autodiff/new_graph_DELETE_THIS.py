#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.util as tdutil
import tadpole.autodiff.node as tdnode


#############################################################################################


# --- (1) --- #


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

       
   def order(self):

       sources, layers = zip(*self._meta.riter())

       return InputOrder(self._nodes.tolist(), sources, layers)





class InputOrder(Ordered): 

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



class PassivePack(Pack):

   def __init__(self, source):

       self._source = source


   def pluginto(self, fun): 

       return self._source.pluginto(fun)




class PointPack(Pack):

   def __init__(self, nodes):

       self._nodes = nodes


   def pluginto(self, fun):

       funcall = FunCall(fun)

       for node in self._nodes:
           funcall = node.pluginto(funcall)

       return funcall.execute()  




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




#############################################################################################


# --- (2) --- #


class Bundled(abc.ABC):  

   @abc.abstractmethod
   def iter(self):
       pass


   @abc.abstractmethod
   def reduced(self):
       pass
  



class Ordered(abc.ABC):

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




class Packable(abc.ABC):

   @abs.abstractmethod
   def pack(self):
       pass




def _nodify(x):

    if isinstance(x, Node):
       return x

    return Point(x)




class Inputs(Bundled, Ordered):

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




def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.adxs():
           return PointPack(self.args())

        source = self.reduced().pack()

        return fun(self, source)

    return wrap




class BasicGlue(Bundled, Ordered, Packable):

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




class ActiveGlue(BasicGlue):

   @default_pack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.parents(),
                                 self.adxs(), 
                                 source,
                                 self.args() 
                                )  
 
       return ActivePack(source, self.layer(), logic)




class PassiveGlue(BasicGlue):

   @default_pack
   def pack(self, source):
    
       return PassivePack(source)




#############################################################################################


# --- (3) --- #

# TODO: NB we're scrapping the old Point class and replacing it with Var -> rename to Point


class Point(Node):

   def attach(self, train):

       return (
               train.with_node(self)
                    .with_meta(self._source, self._layer)
              )




class ReverseNode(Node, Reverse):

   def attach(self, train):

       return self._nodule.attach(train.with_node(self))
       



class Nodule:

   def attach(self, train):

       return train.with_meta(self._source, self._layer)




#############################################################################################



"""
##########################################













class ArgGlue:

   def __init__(self, args):

       self._args = args


   def new(self, args):

       return self.__class__(args)


   @tdutil.cacheable
   def iterate(self):

       return iter(map(_nodify, self._args))


   @tdutil.cacheable
   def train(self):

       concat = NodeConcatenation()

       for arg in self.iterate():
           concat = arg.attach(concat)

       return concat.train()




def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.train().adxs():
           return PointPack(self.train().args())

        args   = self.train().args()
        source = self.new(args).pack()

        return fun(self, source)

    return wrap




class Glue(abc.ABC):

   @abc.abstractmethod
   def new(self, args):
       pass


   @abc.abstractmethod
   def iterate(self):
       pass


   @abc.abstractmethod
   def train(self):
       pass


   @abc.abstractmethod
   def pack(self):
       pass




class Generic(Glue):

   def __init__(self, glue):

       self._glue = glue


   def new(self, args):

       return self.__class__(self._glue.new(args))


   def iterate(self):

       return self._glue.iterate()


   def train(self):

       return self._glue.train()




class Active(Generic):

   @default_pack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(source, self.train.layer(), logic)




class Passive(Generic):

   @default_pack
   def pack(self, source):
    
       return PassivePack(source)

















class Adhesive(abc.ABC):

   @abc.abstractmethod
   def new(self, args):
       pass


   @abc.abstractmethod
   def iterate(self):
       pass


   @abc.abstractmethod
   def train(self):
       pass


   @abc.abstractmethod
   def pack(self, source):
       pass
 



class GenericGlue(Adhesive):


   def __init__(self, args):

       self._args = args


   def new(self, args):

       return self.__class__(self._args.new(args))


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()







class ActiveArgs(GenericGlue):


   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(source, self.train.layer(), logic)




class PassiveArgs(GenericGlue):


   def pack(self, source):
    
       return PassivePack(source)




class ArgGlue:

   def __init__(self, glue):

       self._glue = glue


   def _new(self):

       args = self.train().args()

       return self.__class__(self._glue.new(args))  


   def iterate(self):

       return self._glue.iterate()


   def train(self):

       return self._glue.train()


   def pack(self):

       if not self.train().adxs():
          return PointPack(self.train().args())

       source = self._new().pack()

       return self._glue.pack(source)











class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def new(self, args):

       return self.__class__(Args(args))


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()
  

   @defaultpack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(source, self.train.layer(), logic)




















class DefaultArgs:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()


   def pack(self):

       if not self.train().adxs():
          return PointPack(self.train().args())

       new_source = self.__class__(self.train().args()).pack()

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 new_source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(new_source, self.train.layer(), logic)





class DefaultArgs:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()


   def pack(self):

       if not self.train().adxs():
          return PointPack(self.train().args())

       new_source = self.__class__(self.train().args()).pack()

       return self._args.pack(new_source)
   
       











class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def _new(self):

       return self.__class__(self.train().args())


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()


   def pack(self): # FIXME make all classes share this code!

       if not self.train().adxs():
          return PointPack(self.train().args())

       new_source = self._new().pack()

       return self._pack(new_source)


   def _pack(self, source):

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(source, self.train.layer(), logic)




class PassiveArgs:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return self._args.iterate()


   def train(self):

       return self._args.train()


   def pack(self):

       if not self.train().adxs():
          return PointPack(self.train().args())

       new_source = self.__class__(self.train().args()).pack()

       logic = tdnode.make_logic(
                                 self.train.parents(),
                                 self.train.adxs(), 
                                 new_source,
                                 *self.train.args(), 
                                )  
 
       return ActivePack(new_source, self.train.layer(), logic)




 






   

 




class Args:

   def __init__(self, args, action):

       self._args   = args
       self._action = action


   def _new(self):

       return self.__class__(self.train().args(), self._action)


   def iterate(self):

       return iter(map(_nodify, self._args))


   def train(self):

       glue = Glue()

       for arg in self.iterate():
           glue = arg.glue(glue)

       return glue.concatenate()


   def pack(self):

       if not self.train().adxs():
          return PointPack(self.concat().args())

       new_source = self._new().pack()
 
       return self._action.pack(self.train(), new_source)




class Active:

   def pack(self, train, new_source):

       logic = tdnode.make_logic(
                                 train.parents(),
                                 train.adxs(), 
                                 new_source,
                                 *train.args(), 
                                )  
 
       return ActivePack(new_source, train.layer(), logic)

       




class Passive:

   def pack(self, new_source):
    
       return PassivePack(new_source)









class Args:

   def __init__(self, args):

       self._args = args


   def iterate(self): 

       return iter(map(_nodify, self._args))


   def concatenate(self):

       glue = Glue()

       for arg in self.iterate():
           glue = arg.glue(glue)

       return glue.concatenate()


   def new_source(self): 

       return self.__class__(self.concatenate().args())



class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return self._args.iterate()


   def concatenate(self):

       return self._args.concatenate()


   def new_source(self):

       return self.__class__(self._args.new_source())


   def pack(self):

       new_source = self.__class__(self.sources.args()).pack()

       logic = tdnode.make_logic(
                                 self.sources().parents(),
                                 self.sources().adxs(), 
                                 new_source,
                                 *self.sources().args(), 
                                )  
 
       return ActivePack(new_source, self.sources().layer(), logic)

 








class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def pack(self):

       return        









class Args:

   def __init__(self, args):

       self._args = args


   def iterate(self): 

       return iter(map(_nodify, self._args))


   def sources(self):

       glue = Glue()

       for arg in self.iterate():
           glue = arg.glue(glue)

       return glue.sources()

       


def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.sources().adxs():
           return PointPack(self.sources().args())

        return fun(*args, **kwargs)

    return wrap



class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return self._args.iterate()


   def sources(self):

       return self._args.sources()


   @default_pack
   def pack(self):

       new_source = self.__class__(self.sources.args()).pack()

       logic = tdnode.make_logic(
                                 self.sources().parents(),
                                 self.sources().adxs(), 
                                 new_source,
                                 *self.sources().args(), 
                                )  
 
       return ActivePack(new_source, self.sources().layer(), logic)




class PassiveArgs:

   @default_pack
   def pack(self):

       new_source = self.__class__(self.sources.args()).pack()
    
       return PassivePack(new_source)





"""







































###############################################################################



class ActiveArgs:

   def __init__(self, args):

       self._args = args


   def glue(self):

       args = tuple(map(_nodify, self._args)) 

       return ArgGlue(args, ActiveAdhesive(args))




class ActiveAdhesive:

   def __init__(self, nodes):

       self._nodes = nodes


   def pack(self, sources):

       out = ActiveArgs(sources.args()).glue().pack()

       logic = tdnode.make_logic(
                                 self._nodes,
                                 sources.adxs(), 
                                 out,
                                 *sources.args(), 
                                ) 

       return ActivePack(out, sources.layer(), logic) 




class PassiveArgs:

   def __init__(self, args):

       self._args = args


   def glue(self):

       args = tuple(map(lambda x: x.disconnect(), map(_nodify, self._args))) 

       return ArgGlue(args, PassiveAdhesive())



class PassiveAdhesive:

   def pack(self, sources):

       new_source = PassiveArgs(sources.args()).glue().pack()

       return PassivePack(new_source, sources.layer())









class ArgGlue(Glue):

   def __init__(self, args, adhesive): # FIXME we expect mapped args

       self._args     = args
       self._adhesive = adhesive


   def pack(self):

       if all(isinstance(arg, ValueNode) for arg in self._args):
          return ValuePack(self._args)

       glue = NodeGlue(self._adhesive)

       for arg in self._args:
           glue = arg.glue(glue)

       return glue.pack()

       

    
class NodeGlue(Glue):

   def __init__(self, nodes=None, data=None):

       if nodes is None: nodes = tdutil.Stack()       
       if data  is None: data  = tdutil.Stack()

       self._nodes = nodes
       self._data  = data


   def attach_node(self, node):

       return self.__class__(self._nodes.push(node), self._data)


   def attach_data(self, source, layer):

       return self.__class__(self._nodes, self._data.push((source, layer)))


   def nodes(self):

       return self._nodes.tolist()


   def sources(self):

       return Sources(self.nodes(), *zip(*self._data.riter()))


   def pack(self):

       return self._adhesive.pack(self.sources())




class Sources: 

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



class PassivePack(Pack):

   def __init__(self, source):

       self._source = source


   def pluginto(self, fun): 

       return self._source.pluginto(fun)




class ValuePack(Pack):

   def __init__(self, nodes):

       self._nodes = nodes


   def pluginto(self, fun):

       funcall = FunCall(fun)

       for node in self._nodes:
           funcall = node.pluginto(funcall)

       return funcall.execute()  




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












##################



class NodePackEngine: # pack, map

   def __init__(self):


   def map(self, args):

       return args


   def pack(self, nodes, sources): 

    # out   = NodeArgGlue(sources.args()).pack()


       new_source = ArgGlue(sources.args(), ).pack()

       logic = tdnode.make_logic(
                                 nodes,
                                 sources.adxs(), 
                                 new_source,
                                 *sources.args(), 
                                )  

       return NodePack(new_source, sources.layer(), logic)







class NodeGlue(Glue): # Same for all nodes (incl Var, Point, Rev/Fw Nodes) # Rename to Adhesive?

   def __init__(self, nodes=None, data=None):

       if nodes is None: nodes = tdutil.Stack()       
       if data  is None: data  = tdutil.Stack()

       self._nodes = nodes
       self._data  = data


   def attach_node(self, node):

       return self.__class__(self._nodes.push(node), self._data)


   def attach_data(self, source, layer):

       return self.__class__(self._nodes, self._data.push((source, layer)))


   def sources(self):

       sources, layers = zip(*self._data.riter())

       return Sources(self.nodes(), sources, layers)


   def nodes(self):

       return self._nodes.tolist()


####




   def pack(self):

       nodes   = self._nodes.tolist()
       sources = Sources(nodes, *zip(*self._data.riter()))

       out   = NodeArgGlue(sources.args()).pack()
       logic = tdnode.make_logic(
                                 nodes,
                                 sources.adxs(), 
                                 out,
                                 *sources.args(), 
                                )  

       return NodePack(out, sources.layer(), logic)












###############################################################################


def _nodify(x):

    if isinstance(x, Node):
       return x

    return Var(x)




class ArgMap:

   def __init__(self, fun=None):

       if fun is None:
          def fun(x): return x

       self._fun = fun


   def apply(self, args):

       return tuple(map(self._fun, map(_nodify, args)))




class Adhesive:

   def __init__(self, make_glue):

       self._make_glue = make_glue


   def _glue(self, args):

       if all(isinstance(arg, Var) for arg in args):
          return VarGlue()

       return self._make_glue()


   def glue(self, args):

       glue = self._glue(args)

       for arg in args:
           glue = arg.glue(glue)

       return glue




class ArgGlue(Glue):

   def __init__(self, adhesive, args, argmap):

       self._adhesive = adhesive

       self._args   = args
       self._argmap = argmap


   def pack(self):

       args = self._argmap.apply(self._args)
       glue = self._adhesive.glue(args)

       return glue.pack()




def node_arg_glue(args):

    adhesive = Adhesive(lambda: NodeGlue())
    argmap   = ArgMap()

    return ArgGlue(adhesive, args, argmap)




def point_arg_glue(args):

    adhesive = Adhesive(lambda: PointGlue())
    argmap   = ArgMap(lambda x: x.disconnect())

    return ArgGlue(adhesive, args, argmap)  




class NodeArgGlue(ArgGlue):

   def __init__(self, args):

       adhesive = Adhesive(lambda: NodeGlue())
       argmap   = ArgMap()

       super().__init__(adhesive, args, argmap)




class PointArgGlue(ArgGlue):

   def __init__(self, args):

       adhesive = Adhesive(lambda: PointGlue())
       argmap   = ArgMap(lambda x: x.disconnect())

       super().__init__(adhesive, args, argmap)    




###############################################################################      



class VarGlue(Glue):

   def __init__(self, sources=None):

       if sources is None: 
          sources = tdutil.Stack()

       self._sources = sources


   def attach(self, source):

       





class GenericGlue(Glue):

   def __init__(self, nodes=None, data=None):

       if nodes is None: nodes = tdutil.Stack()       
       if data  is None: data  = tdutil.Stack()

       self._nodes = nodes
       self._data  = data


   def attach_node(self, node):

       return self.__class__(self._nodes.push(node), self._data)


   def attach_data(self, source, layer):

       return self.__class__(self._nodes, self._data.push((source, layer)))


   def pack(self):

       nodes   = self._nodes.tolist()
       sources = Sources(nodes, *zip(*self._data.riter()))

       out   = NodeArgGlue(sources.args()).pack()
       logic = tdnode.make_logic(
                                 nodes,
                                 sources.adxs(), 
                                 out,
                                 *sources.args(), 
                                )  

       return NodePack(out, sources.layer(), logic)



class NodeGlueEngine:

   def __init__(self):


   def pack(self, nodes, sources):



def make_node_pack(nodes, sources):

    out   = NodeArgGlue(sources.args()).pack()
    logic = tdnode.make_logic(
                              nodes,
                              sources.adxs(), 
                              out,
                              *sources.args(), 
                             )  

    return NodePack(out, sources.layer(), logic)




##################














class NodeGlueEngine(GlueEngine):


   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self, new_source):

       logic = tdnode.make_logic(
                                 self._nodes,
                                 self._sources.adxs(), 
                                 new_source,
                                 *self._sources.args(), 
                                )   

       return NodePack(new_source, self._sources.layer(), logic)










###############################################################################



class Var(Node):

   def glue(self, glue):

       return glue.attach(self._source)



class ReverseNode(Node, Reverse):

   def glue(self, glue):

       glue = self._nodule.glue(glue)

       return glue.attach_node(self)
       



class Nodule:

   def glue(self, glue):

       return glue.attach_data(self._source, self._layer)




###############################################################################      



"""
class ArgGlue(Glue):

   def __init__(self, args, argmap, make_glue):

       self._args = args
       self._map  = argmap

       self._make_glue = make_glue 


   def _glue(self, args):

       if all(isinstance(arg, Var) for arg in args):
          return VarGlue()

       return self._make_glue()


   def pack(self):

       args = self._adhesive.args(self._args)
       glue = self._adhesive.glue(args)

       args = self._argmap.apply(self._args)
       glue = self._glue(args)

       for arg in args:
           glue = arg.glue(glue)

       return glue.pack()




class ArgGlue(Glue):

   def __init__(self, args, fun=None):

       if fun is None:
          def fun(x): return x 

       self._args = args
       self._fun  = fun


   def _all(self, cond):

       return all(cond(arg) for arg self._args):


   def _glue(self):

       if self._all(lambda x: isinstance(x, Var)):
          return VarGlue()

       if self._all(lambda x: isinstance(x, (ReverseNode, ForwardNode, Var))):
          return NodeGlue()
 
       if self._all(lambda x: isinstance(x, (Point, Var)):
          return PointGlue()

       raise ValueError(
          f"\nArgGlue: invalid set of argument types "
          f"{''.join(str(type(arg)) for arg in self._args)}")


   def pack(self):

       if self._all(lambda x: not isinstance(x, Node)):  
          return VarPack(self._args)

       args = tuple(map(self._fun, map(_nodify, self._args))) 
       glue = self._glue()

       for arg in args:
           glue = arg.glue(glue)

       return glue.pack()


"""





















###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       args = [Var(arg) for arg in args if not isinstance(arg, Node)]
       fun  = FunWithGate(self, self._fun)

       return (
               make_glue(args).pack()
                              .pluginto(fun) 
              )




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       args = tuple(map(lambda x: x.disconnect(), args)) # FIXME but we must do this recursively!

       return (
               make_glue(args).pack()
                              .pluginto(self._fun) 
              )






#####################################################################


class NodeGlue(Glue):

   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self, ):






def make_glue(nodes):
 
    return self._nodes[0].glue(*self._nodes[1:])





class ArgGlue(Glue):

   def __init__(self, sources, engine):

       self._nodes   = nodes
       self._sources = sources
       self._engine  = engine


   def pack(self): # FIXME input the mapped nodes here (FIXME IDEA: SourcesWithMap(Sources(...), map) ?

       if not self._sources.adxs():
          return VarPack(self._sources.args()) # FIXME but this must stay at the top

       new_source = make_glue(self._sources.args()).pack()

       return self._engine.pack(new_source)




class NodeGlue(ArgGlue):

   def __init__(self, nodes, sources):

       super().__init__(sources, NodeGlueEngine(nodes, sources))




class PointGlue(ArgGlue):

   def __init__(self, nodes, sources):

       super().__init__(sources, PointGlueEngine(sources))




class NodeGlueEngine(GlueEngine):

   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self, new_source):

       logic = tdnode.make_logic(
                                 self._nodes,
                                 self._sources.adxs(), 
                                 new_source,
                                 *self._sources.args(), 
                                )   

       return NodePack(new_source, self._sources.layer(), logic)




class PointGlueEngine(GlueEngine):

   def __init__(self, sources):

       self._sources = sources


   def pack(self, new_source):

       return PointPack(new_source, self._sources.layer())




#####################################################################
       

class NodePack(Pack):

   def __init__(self, source, layer, logic):

       self._source = source
       self._layer  = layer
       self._logic  = logic


   def pluginto(self, fun): 

       out = self._source.pluginto(funcall)

       return tdnode.make_node(
                               out, 
                               self._layer, 
                               fun.gate(self._logic) 
                              )



class PointPack(Pack):

   def __init__(self, source):

       self._source = source


   def pluginto(self, fun):

       return self._source.pluginto(fun)




class VarPack(Pack):

   def __init__(self, nodes): 

       self._nodes = nodes


   def pluginto(self, fun):

       funcall = FunCall(fun)

       for node in self._nodes:
           funcall = node.pluginto(funcall)

       return funcall.execute()  




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




#####################################################################



class SourcesWithMap:

   def __init__(self, sources, fun): # FIXME this only does mapping after reduction, but the initial Glue created 
                                     #       will still be NodeGlue... 

       self._sources = sources
       self._fun     = fun


   def layer(self):

       return self._sources.layer()


   def adxs(self):

       return self._sources.adxs()


   def args(self):

       return self._fun(self._sources.args())




class Sources: 

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




# FIXME reimplement NodeGlue as a Media/Engine object



class Var(Node): # FIXME impl like Point but insert self into Glue

   def __init__(self, source):

       self._source = source


   def glue(self, *others):

       




#####################################################################










class ArgGlue(Glue):

   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self):

       if not self._sources.adxs():
          return VarPack(self._sources.args()) # FIXME this must be at the top!

       return make_glue(self._nodes).pack()
 


class NodeGlue(Glue):

   def __init__(self, nodes, sources):

       self._core    = ArgGlue(nodes, sources)
       self._nodes   = nodes
       self._sources = sources


   def pack(self):

       out   = self._core.pack()
       logic = tdnode.make_logic(
                                 self._nodes,
                                 self._sources.adxs(), 
                                 out,
                                 *self._sources.args(), 
                                )   

       return NodePack(out, self._sources.layer(), logic)





 
       



class NodeGlue(Glue):

   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self):

       if not self._sources.adxs():
          return VarPack(self._sources.args()) # FIXME we want this to contain the unpacked values! 

       out   = make_glue(self._nodes).pack()  
       logic = tdnode.make_logic(
                                 self._nodes,
                                 self._sources.adxs(), 
                                 out,
                                 *self._sources.args(), 
                                )    

       return NodePack(out, self._sources.layer(), logic)




class PointGlue(Glue):

   def __init__(self, nodes, sources):

       self._nodes   = nodes
       self._sources = sources


   def pack(self):

       if not self._sources.adxs():
          return VarPack(self._sources.args())

       out = make_glue(self._nodes).pack()

       return PointPack(out, self._sources.layer())

       




class NodePack(Pack):

   def __init__(self, source, layer, logic):

       self._source = source
       self._layer  = layer
       self._logic  = logic


   def pluginto(self, fun): 

       out = self._source.pluginto(fun)

       return tdnode.make_node(
                               out, 
                               self._layer, 
                               fun.gate(self._logic) # FIXME rename GatedFun -> FunCall (with .compute() and .gate() methods)?
                              )





class PointPack(Pack):

   def __init__(self, source):

       self._source = source


   def pluginto(self, fun):

       return self._source.pluginto(fun)




class VarPack(Pack):

   def __init__(self, nodes): # FIXME we want this to contain the unpacked values!

       self._nodes = nodes


   def pluginto(self, fun):

       return fun(*self._nodes)  









class VarNodule(Nodule):

   def __init__(self, source):

       super().__init__(source, -1)


   def tovalue(self):

       return 

































































































