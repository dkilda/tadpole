#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.util as tdutil

from tests.common.fakes        import NULL, fakeit
from tests.autodiff.fakes.misc import Fun, FunReturn, Map






###############################################################################
###                                                                         ###
### Sequence data structure (quasi-immutable)                               ###
###                                                                         ###
###############################################################################


# --- Sequence -------------------------------------------------------------- #

class Sequence:

   def __init__(self, push=NULL, pop=NULL, top=NULL, 
                      size=NULL, empty=NULL, contains=NULL, iterate=NULL):

       self._push = push
       self._pop  = pop
       self._top  = top
       self._size = size

       self._contains = contains
       self._iterate  = iterate


   @fakeit
   def push(self, x):

       return self._push[x]


   @fakeit
   def pop(self):

       return self._pop


   @fakeit
   def top(self):

       return self._top


   @fakeit
   def size(self):

       return self._size


   @fakeit
   def empty(self):

       return self._empty


   @fakeit
   def contains(self, x):

       return self._contains[x]


   @fakeit
   def iterate(self):

       return self._iterate


   def __len__(self):

       return self.size()


   def __contains__(self, x):

       return self.contains(x)


   def __iter__(self):

       return self.iterate()


   def __reversed__(self):

      return reversed(self.iterate())








































