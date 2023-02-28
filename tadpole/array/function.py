#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util




###############################################################################
###                                                                         ###
###  General framework for array function calls.                            ###
###                                                                         ###
###############################################################################


# --- Content --------------------------------------------------------------- #

class Content:

   def __init__(self, content=util.Sequence()):

       self._content = content


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._content, other._content)

       return bool(log)


   def __iter__(self):

       return iter(self._content)


   def __len__(self):

       return len(self._content)

         
   def attach(self, array, data):

       return self.__class__(self._content.push((array, data)))


# --- Visit ----------------------------------------------------------------- #

class Visit:

   def __init__(self, fun, content=Content()):

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


   def attach(self, array, data):

       return self.__class__(self._fun, self._content.attach(array, data)) 


   def execute(self):

       arrays, datas = zip(*self._content)
       space         = arrays[0].space() 

       return space.visit(self._fun, *datas) 




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, fun, content=Content()):

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


   def attach(self, array, data):

       return self.__class__(self._fun, self._content.attach(array, data)) 


   def execute(self):

       arrays, datas = zip(*self._content)
       space         = arrays[0].space() 

       return space.apply(self._fun, *datas) 




# --- Args ------------------------------------------------------------------ #

class Args:

   def __init__(self, *args):

       self._args = args


   def allclose(self, other, **opts):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return core.allallclose(self._args, other._args, **opts)    

       return False


   def __eq__(self, other):

       if not type(self) == type(other):
          return False

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          return core.allallequal(self._args, other._args)    

       return False


   def __len__(self):

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __getitem__(self, idx):

       return self._args[idx]


   def pluginto(self, funcall):

       for arg in self._args:
           funcall = arg.pluginto(funcall)

       return funcall.execute()




