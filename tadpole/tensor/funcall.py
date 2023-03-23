#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.util        as util
import tadpole.tensor.core as core




###############################################################################
###                                                                         ###
###  Arguments for tensor function calls                                    ###
###                                                                         ###
###############################################################################


# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return all(x == y for x, y in zip(self._args, other._args))    

       return False


   def __len__(self):

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __reversed__(self):

       return iter(list(reversed(self._args)))


   def __getitem__(self, idx):

       return self._args[idx]


   def pluginto(self, funcall):

       for arg in self._args:
           funcall = arg.pluginto(funcall)

       return funcall.execute()




###############################################################################
###                                                                         ###
###  Engine for tensor function calls                                       ###
###                                                                         ###
###############################################################################


# --- EngineLike interface -------------------------------------------------- #

class EngineLike(abc.ABC):

   @abc.abstractmethod
   def nargs(self):
       pass

   @abc.abstractmethod
   def datas(self):
       pass

   @abc.abstractmethod
   def inds(self):
       pass

   @abc.abstractmethod
   def attach(self, backend, data, inds):
       pass

   @abc.abstractmethod
   def execute(self, *args, **kwargs):
       pass




# --- Engine ---------------------------------------------------------------- #

class Engine(EngineLike):

   def __init__(self, fun, datas=None, inds=None):
   
       if datas is None: datas = util.Sequence()
       if inds  is None: inds  = util.Sequence()

       self._fun   = fun
       self._datas = datas
       self._inds  = inds


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):

          log.val(self._fun, other._fun)

       if bool(log):

          log.val(self._inds, other._inds)

          if bool(log):
             return all(
                op.allequal(x, y) for x, y in zip(self._datas, other._datas)
             )

       return False


   def nargs(self):

       return len(self._datas)


   def datas(self):

       return iter(self._datas)


   def inds(self):

       return iter(self._inds)

  
   def attach(self, data, inds):

       return self.__class__(
                             self._fun,
                             self._datas.push(data),
                             self._inds.push(inds)
                            ) 


   def execute(self, *args, **kwargs):

       return self._fun(*args, **kwargs)




###############################################################################
###                                                                         ###
###  Generic tensor function calls                                          ###
###                                                                         ###
###############################################################################


# --- Function call interface ----------------------------------------------- #

class FunCall(abc.ABC):

   @abc.abstractmethod
   def attach(self, data, inds):
       pass

   @abc.abstractmethod
   def execute(self):
       pass




# --- Extract call ---------------------------------------------------------- #

class ExtractCall(FunCall): 

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       return self._engine.execute(*self._engine.datas())




# --- Transform call -------------------------------------------------------- #

class TransformCall(FunCall): 

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()

       return self._engine.execute(data, inds)




###############################################################################
###                                                                         ###
### Standard algebra function calls                                         ###
###                                                                         ###
###############################################################################


# --- Elementwise call ------------------------------------------------------ #

class Elemwise(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       assert len(set(self._engine.inds())) == 1,
          f"\n{type(self).__name__}.execute(): "
          f"an elementwise operation cannot be performed for tensors "
          f"with non-matching indices {tuple(self._engine.inds())}" 

       outinds = next(self._engine.inds())
       outdata = self._engine.execute(*self._engine.datas())

       return core.Tensor(outdata, outinds)




# --- Reduce call ----------------------------------------------------------- #

class Reduce(FunCall):

   def __init__(self, engine, inds=None):

       if inds is None:
          inds = tuple()

       if not isinstance(inds, (tuple, util.TupleLike)):
          inds = (inds,)

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine
       self._inds   = inds


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()
       
       outinds = inds.remove(*self._inds)
       outdata = self._engine.execute(data, inds.axes(self._inds))

       return core.Tensor(outdata, outinds)




###############################################################################
###                                                                         ###
###  Reindexing and reshaping function calls                                ###
###                                                                         ###
###############################################################################


# --- Reindex call ---------------------------------------------------------- #

class Reindex(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()

       outinds = self._engine.execute(inds)
       
       assert outinds.shape == inds.shape, (
          f"\n{type(self).__name__}.execute(): "
          f"the output shape {outinds.shape} is not compatible "
          f"with the original shape {inds.shape}."
       )
       
       return core.Tensor(data, outinds)




# --- Reshape call ---------------------------------------------------------- #

class Reshape(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))  


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()

       outinds = self._engine.execute(inds)
       outdata = op.reshape(data, outinds.shape)
       
       assert outinds.shape == inds.shape, (
          f"\n{type(self).__name__}.execute(): "
          f"the output shape {outinds.shape} is not compatible "
          f"with the original shape {inds.shape}."
       )
       
       return core.Tensor(outdata, outinds)




