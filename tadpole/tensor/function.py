#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util       as util
import tadpole.tensor.core as core
 



###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Comparison of iterables of tensors ------------------------------------ #

def allallequal(xs, ys):

    return all(core.allequal(x, y) for x, y in zip(xs, ys))


def allallclose(xs, ys, **opts):

    return all(core.allclose(x, y, **opts) for x, y in zip(xs, ys))




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
   def backends(self):
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

   def __init__(self, fun, backends=None, datas=None, inds=None):
   
       if backends is None: backends = util.Sequence()
       if datas    is None: datas    = util.Sequence()
       if inds     is None: inds     = util.Sequence()

       self._fun      = fun
       self._backends = backends
       self._datas    = datas
       self._inds     = inds


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):

          log.val(self._fun, other._fun)

       if bool(log):

          log.val(self._backends, other._backends)
          log.val(self._inds,     other._inds)

          if bool(log):
             return allallequal(self._datas, other._datas)

       return False


   def nargs(self):

       return len(self._datas)


   def backends(self):

       return iter(self._backends)


   def datas(self):

       return iter(self._datas)


   def inds(self):

       return iter(self._inds)

  
   def attach(self, backend, data, inds):

       return self.__class__(
                             self._fun,
                             self._backends.push(backend),
                             self._datas.push(data),
                             self._inds.push(inds)
                            ) 


   def execute(self, *args, **kwargs):

       return self._fun(*args, **kwargs)




###############################################################################
###                                                                         ###
###  Tensor function calls: various call types                              ###
###                                                                         ###
###############################################################################


# --- Function call interface ----------------------------------------------- #

class FunCall(abc.ABC):

   @abc.abstractmethod
   def attach(self, backend, data, inds):
       pass

   @abc.abstractmethod
   def execute(self):
       pass




# --- Reindexing call ------------------------------------------------------- #

class ReindexCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, EngineLike):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend, = self._engine.backends()
       data,    = self._engine.datas()
       inds,    = self._engine.inds()

       outinds = self._engine.execute(inds) 

       if outinds.shape == inds.shape: 
          return core.Tensor(backend, data, outinds)

       try:
          outdata = backend.reshape(data, outinds.shape)
       except ValueError:
          print((
             f"\nReindexCall.execute(): "
             f"output indices shape {outinds.shape} "
             f"is not compatible with data shape {data.shape}."
          ))
          raise 

       return core.Tensor(backend, outdata, outinds)  
       



# --- Transform call -------------------------------------------------------- #

class TransformCall(FunCall): # TODO NB identical to a generic DecompCall

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend, = self._engine.backends()
       data,    = self._engine.datas()
       inds,    = self._engine.inds()

       return self._engine.execute(backend, data, inds)




# --- Visit call ------------------------------------------------------------ #

class VisitCall(FunCall): 

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend = next(self._engine.backends())

       return self._engine.execute(backend, *self._engine.datas())




# --- Unary elementwise call ------------------------------------------------ #

class UnaryElemWiseCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend, = self._engine.backends()
       data,    = self._engine.datas()
       inds,    = self._engine.inds()

       outdata = self._engine.execute(backend, data, inds)

       return core.Tensor(backend, outdata, inds)




# --- Binary elementwise call ----------------------------------------------- #

class BinaryElemWiseCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend = next(self._engine.backends())

       dataA, dataB = self._engine.datas()
       indsA, indsB = self._engine.inds()

       assert indsA == indsB, 
          f"BinaryElemWiseCall.execute(): an elementwise operation "
          f"cannot be performed for tensors "
          f"with non-matching indices {indsA} != {indsB}"

       outdata = self._engine.execute(backend, dataA, dataB)

       return core.Tensor(backend, outdata, indsA)




# --- Dot call -------------------------------------------------------------- #

class DotCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend = next(self._engine.backends())

       dataA, dataB = self._engine.datas()
       indsA, indsB = self._engine.inds()

       outinds = contract.make_output_inds((indsA, indsB))
       outdata = self._engine.execute(backend, dataA, dataB)

       return core.Tensor(backend, outdata, outinds)




# --- Einsum call ----------------------------------------------------------- #

class EinsumCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend = next(self._engine.backends())
       datas   = self._engine.datas()
       inds    = self._engine.inds()

       outinds  = contract.make_output_inds(inds) 
       equation = contract.make_einsum_equation(inds, outinds)

       outdata = self._engine.execute(backend, equation, *datas)

       return core.Tensor(backend, outdata, outinds)  




# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return allallequal(self._args, other._args)    

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










"""

# --- Function call interface ----------------------------------------------- #

class Call(abc.ABC):

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def attach(self, backend, data, inds):
       pass

   @abc.abstractmethod
   def execute(self):
       pass




# --- Visit call ------------------------------------------------------------ #

class VisitCall(FunCall):

   def __init__(self, fun, content=None):

       if not isinstance(content, Content):
          content = Content(content)

       self._fun     = fun
       self._content = content


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self,          other)
       log.val(self._fun,     other._fun)
       log.val(self._content, other._content)

       return bool(log)


   def __iter__(self):

       return iter(self._content)


   def __len__(self):

       return len(self._content)


   def attach(self, backend, data, inds):

       return self.__class__(
                             self._fun, 
                             self._content.attach(backend, data, inds)
                            ) 

   def execute(self):

       backends, datas, inds = zip(*self._content)

       return self._fun(backends[0], *datas)




# --- Transform call -------------------------------------------------------- #

class TransformCall(FunCall):

   def __init__(self, fun, content=None):

       if not isinstance(content, Content):
          content = Content(content)

       self._fun     = fun
       self._content = content


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self,          other)
       log.val(self._fun,     other._fun)
       log.val(self._content, other._content)

       return bool(log)


   def __iter__(self):

       return iter(self._content)


   def __len__(self):

       return len(self._content)


   def attach(self, backend, data):

       return self.__class__(
                             self._fun, 
                             self._content.attach(backend, data, inds)
                            ) 

   def execute(self):

       backends, datas, inds = zip(*self._content)

       out, inds = self._fun(backends[0], *datas, *inds)
       return core.Tensor(backends[0], out, inds)




# --- Split call ------------------------------------------------------------ #

class SplitCall(FunCall):

   def __init__(self, fun, content=None):

       if not isinstance(content, Content):
          content = Content(content)

       self._fun     = fun
       self._content = content


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self,          other)
       log.val(self._fun,     other._fun)
       log.val(self._content, other._content)

       return bool(log)


   def __iter__(self):

       return iter(self._content)


   def __len__(self):

       return len(self._content)


   def attach(self, backend, data):

       return self.__class__(
                             self._fun, 
                             self._content.attach(backend, data)
                            ) 

   def execute(self):

       backends, datas, inds = zip(*self._content)

       outputs = self._fun(backends[0], *datas)
       outputs = (core.Tensor(backends[0], out) for out in outputs)
       
       return tuple(outputs) 

"""




