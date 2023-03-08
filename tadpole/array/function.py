#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util             as util
import tadpole.array.core       as core
import tadpole.array.operations as op

from tadpole.array.types import (
   ContentLike,
   FunCall,
)



###############################################################################
###                                                                         ###
###  General framework for array function calls.                            ###
###                                                                         ###
###############################################################################


# --- Content --------------------------------------------------------------- #

class Content:

   def __init__(self, content=None):
   
       if content is None:
          content = util.Sequence()

       self._content = content


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):

          arrays1, datas1 = zip(*self._content)
          arrays2, datas2 = zip(*other._content)

          return ((arrays1 == arrays2) 
                  and util.allallequal(datas1, datas2))

       return False


   def __iter__(self):

       return iter(self._content)


   def __len__(self):

       return len(self._content)

         
   def attach(self, backend, data):

       return self.__class__(self._content.push((backend, data)))




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


   def attach(self, backend, data):

       return self.__class__(
                             self._fun, 
                             self._content.attach(backend, data)
                            ) 

   def execute(self):

       backends, datas = zip(*self._content)

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
                             self._content.attach(backend, data)
                            ) 

   def execute(self):

       backends, datas = zip(*self._content)

       out = self._fun(backends[0], *datas)
       return core.Array(backends[0], out)




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

       backends, datas = zip(*self._content)

       outputs = self._fun(backends[0], *datas)
       outputs = (core.Array(backends[0], out) for out in outputs)
       
       return tuple(outputs) 

 


# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def allclose(self, other, **opts):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return op.allallclose(self._args, other._args, **opts)    

       return False


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return op.allallequal(self._args, other._args)    

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




