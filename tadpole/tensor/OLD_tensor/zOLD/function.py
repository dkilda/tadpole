#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util        as util
import tadpole.tensor.core as core
 



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
             return all(bn.allequal(x, y) 
                for bn, x, y in zip(self._backends, self._datas, other._datas)
             )

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




# --- Transform call -------------------------------------------------------- #

class TransformCall(FunCall): 

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




# --- Extract call ---------------------------------------------------------- #

class ExtractCall(FunCall): 

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend = next(self._engine.backends())

       return self._engine.execute(backend, *self._engine.datas())




# --- Reduce call ----------------------------------------------------------- #

class ReduceCall(FunCall):

   def __init__(self, engine, inds=None):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       if inds is None:
          inds = tuple()

       if not isinstance(inds, (tuple, util.TupleLike)):
          inds = (inds,)

       self._engine = engine
       self._inds   = inds


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       backend, = self._engine.backends()
       data,    = self._engine.datas()
       inds,    = self._engine.inds()
       
       outinds = inds.remove(*self._inds)
       outdata = self._engine.execute(backend, data, self._axes)

       return core.Tensor(backend, outdata, outinds)


   @property
   def _axes(self):

       inds, = self._engine.inds()

       if len(self._inds) == 0:
          return None

       if len(self._inds) == 1:
          return inds.axis(self._inds)

       return inds.axes(self._inds)




# --- Elementwise call ------------------------------------------------------ #

class ElemwiseCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   def execute(self):

       assert len(set(self._engine.inds())) == 1,
          f"ElemwiseCall.execute(): "
          f"an elementwise operation cannot be performed for tensors "
          f"with non-matching indices {tuple(self._engine.inds())}" 

       backend = next(self._engine.backends())
       outinds = next(self._engine.inds())
       outdata = self._engine.execute(backend, *self._engine.datas())

       return core.Tensor(backend, outdata, outinds)




# --- Reindex call ---------------------------------------------------------- #

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
       
       assert outinds.shape == inds.shape, (
          f"\nReindexCall.execute(): "
          f"the output shape {outinds.shape} is not compatible "
          f"with the original shape {inds.shape}."
       )
       
       return core.Tensor(backend, data, outinds)




# --- Reshape call ---------------------------------------------------------- #

class ReshapeCall(FunCall):

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
       outdata = backend.reshape(data, outinds.shape)
       
       assert outinds.shape == inds.shape, (
          f"\nReindexCall.execute(): "
          f"the output shape {outinds.shape} is not compatible "
          f"with the original shape {inds.shape}."
       )
       
       return core.Tensor(backend, data, outinds)




# --- Dot product call ------------------------------------------------------ #

class DotCall(FunCall):

   def __init__(self, engine):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine = engine


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   @util.cacheable
   def outinds(self):

       return contract.make_output_inds(self._engine.inds())


   def execute(self):

       backend = next(self._engine.backends())
       outdata = self._engine.execute(backend, *self._engine.datas())

       return core.Tensor(backend, outdata, self.outinds())




# --- Einsum call ----------------------------------------------------------- #

class EinsumCall(FunCall):

   def __init__(self, engine, outinds=None):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine  = engine
       self._outinds = outinds


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   @util.cacheable
   def outinds(self):

       if self._outinds is None: 
          return contract.make_output_inds(inds)

       return self._outinds


   def execute(self):

       backend = next(self._engine.backends())
       datas   = self._engine.datas()
       inds    = self._engine.inds()

       equation = contract.make_einsum_equation(inds, self.outinds())
       outdata  = self._engine.execute(backend, equation, *datas)

       return core.Tensor(backend, outdata, self.outinds()  




# --- Decomposition call ---------------------------------------------------- #

class DecompCall(FunCall):

   def __init__(self, engine, outinds=None):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine  = engine
       self._outinds = outinds


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   @util.cacheable
   def outinds(self):

       if self._outinds is None: 
          return contract.make_output_inds(inds)

       return self._outinds


   def execute(self):

       backend = next(self._engine.backends())
       datas   = self._engine.datas()
       inds    = self._engine.inds()

       equation = contract.make_einsum_equation(inds, self.outinds())
       outdata  = self._engine.execute(backend, equation, *datas)

       return core.Tensor(backend, outdata, self.outinds() 










"""

###############################################################################
###############################################################################
###############################################################################


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
       
"""



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




