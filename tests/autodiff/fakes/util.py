#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import collections

import tests.common              as common
import tests.autodiff.fakes.util as util

import tadpole.autodiff.util as tdutil




###############################################################################
###                                                                         ###
###  Basic building blocks for fake types:                                  ###
###  map, operation set, value, function, function map, operator.           ###
###                                                                         ###
###############################################################################


# --- Map ------------------------------------------------------------------- #

class Map:

   def __init__(self, out):

       if not isinstance(out, dict):
          out = collections.defaultdict(lambda: out)

       self._out = out


   def __getitem__(self, key):

       try:
          return self._out[key]

       except KeyError:

          keys = list(self._out.keys())
          return self._out[keys[keys.index(key)]]  




# --- Operation set: add, multiply ------------------------------------------ #

class Set:

   def __init__(self, items):

       if not isinstance(items, frozenset):
          items = frozenset([items])

       self._items = items


   def __eq__(self, other):

       return all((
                   type(self)  == type(other),
                   self._items == other._items,
                 ))


   def __hash__(self):

       return hash(self._items)


   def __add__(self, other):

       return AddSet(self._items | other._items)


   def __mul__(self, other):

       return MulSet(self._items | other._items)




class AddSet(Set):
   pass


class MulSet(Set):
   pass




# --- Value unit ------------------------------------------------------------ #

class ValueUnit:
   pass
 



# --- Value ----------------------------------------------------------------- #

class Value:

   def __init__(self, val=None):  

       if val is None:
          val = ValueUnit() 

       if not isinstance(val, Set):
          val = AddSet(val)

       self._val = val


   def __eq__(self, other):

       return self._val == other._val


   def __hash__(self):

       return hash(self._val)


   def __iter__(self):

       yield self


   def __len__(self):

       return 1


   def __contains__(self, x):

       return x == self


   def __getitem__(self, idx):

       return tuple(self)[idx]


   def __add__(self, other):

       if not other:
          return self

       return self.__class__(self._val + other._val)  


   def __radd__(self, other):

       return self.__add__(other)
 

   def __mul__(self, other):

       if not other:
          return self

       return self.__class__(self._val * other._val)


   def __rmul__(self, other):

       return self.__mul__(other)




# --- Fun ------------------------------------------------------------------- #

class Fun:

   def __init__(self, out, *args):

       self._out  = out
       self._args = args


   def update_args(self, *args): # TODO NB, THIS SOLVES PROBLEMS WITH CIRCULAR REFS!

       self._args = args
       return self


   def __call__(self, *args):

       if self._out is None:
          return Value()

       if not self._args:
          return self._out

       if isinstance(self._out, Map):
          return self._out[args]

       assert self._args == args, (
           f"Fun: no output for the args {args} provided. "
           f"Fun accepts args {self._args}."
       )

       return self._out




# --- FunMap ---------------------------------------------------------------- #

class FunMap:

   def __init__(self, **data):

       self._data = data


   def __getitem__(self, key): 

       try:
          name, default = key 
       except ValueError:
          name, default = key, None
       
       return self._data.get(name, lambda *args, **kwargs: default)




# --- Operator -------------------------------------------------------------- #

class Op:

   def __init__(self, transform=None):

       self._transform = transform


   def __call__(self, fun, *args):

       if self._tranform is not None:
          args = self._transform(*args)

       return fun(*args)




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy -------------------------------------------------------- #

class ArgProxy(tdutil.ArgProxy):

   def __init__(self, **data):  

       self._data = data


   def _get(self, name, default=None):
       
       return self._data.get(name, default)


   def insert(self, args, x):

       default = Fun(Value())

       return self._get("insert", default)(args, x)


   def extract(self, args):

       default = Fun(Value())

       return self._get("extract", default)(args)


 

