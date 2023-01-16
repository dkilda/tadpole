#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc


###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Decorator for cacheable methods -------------------------------------- #

def cacheable(fun):

    name = f"_{fun.__name__}_cached"

    def wrap(self):

        try:
           return getattr(self, name) 

        except AttributeError:
           out = fun(self)
           setattr(self, name, out)
           return out

    return wrap




###############################################################################
###                                                                         ###
### Sequence data structure (quasi-immutable)                               ###
###                                                                         ###
###############################################################################


# --- Sequence ----------------------------------------------------------------- #

class Sequence: 

   def __init__(self, xs=None, end=0):

       if xs is None:
          xs = []

       self._xs  = xs
       self._end = end


   def push(self, x):

       self._xs.append(x)
       return self.__class__(self._xs, self._end + 1)


   def pop(self):

       return self.__class__(self._xs, self._end - 1)


   def top(self):

       return self._xs[self._end - 1] 


   def size(self):

       return self._end


   def empty(self):

       return self.size() == 0


   def contains(self, x):

       return x in self._xs[:self._end]


   def iterate(self):

       for i in range(self.size()):
           yield self._xs[i]


   def __len__(self):

       return self.size()


   def __contains__(self, x):

       return self.contains(x)


   def __iter__(self):

       return self.iterate()


   def __reversed__(self):

      return reversed(self.iterate())




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy interface ---------------------------------------------- #

class ArgProxy(abc.ABC):

   @abc.abstractmethod
   def insert(self, args, x):
       pass

   @abc.abstractmethod
   def extract(self, args):
       pass




# --- Singular argument proxy (represents a single variable in args) -------- #

class SingularArgProxy(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def _str(self):

       out = StringRep(self)
       out = out.with_data("adx", self._adx)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._adx == other._adx


   def insert(self, args, x):
        
       out            = list(args)
       out[self._adx] = x

       return out


   def extract(self, args):

       return args[self._adx]




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class PluralArgProxy(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def _str(self):

       out = StringRep(self)
       out = out.with_data("adx", self._adx)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._adx == other._adx


   def insert(self, args, x):

       out = list(args)

       for i, v in zip(self._adx, x):
           out[i] = v  

       return out


   def extract(self, args):

       return tuple(args[i] for i in self._adx)  




# --- Unary function argument proxy ----------------------------------------- # 

def make_argproxy(adx):

    if adx is None:
       adx = 0

    if isinstance(adx, int):
       return SingularArgProxy(adx)

    return PluralArgProxy(adx)




###############################################################################
###                                                                         ###
###  Printing utility: string representation of an object.                  ###
###                                                                         ###
###############################################################################


# --- Helper methods -------------------------------------------------------- #

def _format(x):

    return f"type = {str(type(x))[8:-2]}, id = {id(x)} "


def _str(name, x):

    if isinstance(x, (list, tuple)):
       return "\n{}: [\n{}\n]".format(name, '\n'.join(_format(v) for v in x))

    return f"\n{name}: {_format(x)}"  




# --- String representation ------------------------------------------------- #

class StringRep:

   def __init__(self, obj, members=None, data=None):

       if members is None: members = Sequence()
       if data    is None: data    = Sequence()

       self._obj     = obj
       self._members = members
       self._data    = data


   def with_member(self, name, x):

       return self.__class__(
                             self._obj, 
                             self._members.push((name, x)), 
                             self._data
                            )


   def with_data(self, name, x):

       return self.__class__(
                             self._obj, 
                             self._members, 
                             self._data.push((name, x))
                            )


   def _obj_str(self):
 
       return f"\n\n{type(self._obj).__name__}. "


   def _member_str(self):

       return "".join(_str(name, x) for name, x in self._members)


   def _data_str(self):

       return "".join(f"\n{name}: {x}" for name, x in self._data)


   def compile(self):

       return f"{self._obj_str()}"  \
              f"{self._data_str()}" \
              f"{self._member_str()}\n\n"



















































