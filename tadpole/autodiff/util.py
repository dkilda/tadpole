#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc


###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Decorator for cacheable methods --------------------------------------- #

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


# --- Sequence -------------------------------------------------------------- #

class Sequence: 

   def __init__(self, xs=None, end=0):

       if xs is None:
          xs = []

       self._xs  = xs
       self._end = end


   @property
   def _sequence(self):

       return self._xs[:self._end] 


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("items", self._sequence)

       return str(rep)


   def __eq__(self, other):

       if self._end != other._end:
          return False

       if self._end > 0:
          return self._sequence == other._sequence

       return True


   def __hash__(self):

       return id(self)


   def __len__(self):

       return self._end


   def __contains__(self, x):

       return x in self._sequence


   def __iter__(self):

       for i in range(self.size()):
           yield self._xs[i]


   def __reversed__(self):

      return reversed(self.__iter__())


   def push(self, x):

       self._xs.append(x)
       return self.__class__(self._xs, self._end + 1)


   def pop(self):

       return self.__class__(self._xs, self._end - 1)




###############################################################################
###                                                                         ###
### Tuple data structure (immutable):                                       ###
### a base class for customized tuple-like types.                           ###
###                                                                         ###
###############################################################################


# --- Tuple ----------------------------------------------------------------- #

class Tuple:

   def __init__(self, xs):

       self._xs = xs


   def __repr__(self):

       rep = ReprChain()

       rep.typ(self)
       rep.ref("items", self._xs)

       return str(rep)


   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.ref(self._xs, other._xs)

       return bool(log)


   def __hash__(self):

       return hash(self._xs)


   def __len__(self):

       return len(self._xs)


   def __contains__(self, x):

       return x in self._xs


   def __iter__(self):

       return iter(self._xs)


   def __getitem__(self, idx):

       return self._xs[idx]




###############################################################################
###                                                                         ###
### Customized loop iterator.                                               ###
### Defined by the first item and the next and stop functions, instead of   ###
### a range. Can be traversed in forward and reverse directions, keeps      ###
### track of the last item of the loop.                                     ###
###                                                                         ###
###############################################################################


# --- Loop iterator --------------------------------------------------------- #

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

def argproxy(adx):

    if adx is None:
       adx = 0

    if isinstance(adx, int):
       return SingularArgProxy(adx)

    return PluralArgProxy(adx)




###############################################################################
###                                                                         ###
###  Logical chain: used for equality comparisons and other assertions.     ###
###                                                                         ###
###############################################################################


# --- Helper methods -------------------------------------------------------- #

def refid(x):

    if isinstance(x, (tuple, list)):
       return tuple(map(id, x))

    return id(x)




# --- Logical chain --------------------------------------------------------- #

class LogicalChain:

   def __init__(self):

       self._chain = []


   def _add(self, cond):

       self._chain.append(cond)
       return self


   def typ(self, x, y):

       return self._add(type(x) == type(y))
 

   def ref(self, x, y):

       return self._add(refid(x) == refid(y))
        

   def val(self, x, y):

       return self._add(x == y)


   def __bool__(self):

       return all(self._chain)


   def __repr__(self):

       rep = ReprChain()
       rep.typ(self)
       return str(rep)




###############################################################################
###                                                                         ###
###  Representation chain: constructs a string representation (repr)        ###
###  of an object.                                                          ###
###                                                                         ###
###############################################################################


# --- Helper methods (for formatting) --------------------------------------- #

def format_type(x):

    return f"{str(type(x))[8:-2]}"


def format_obj(x):

    return f"{format_type(x)}, id = {id(x)}, contains:"


def format_val(name, x):

    return f"{format_type(x)} {name} = {x}"


def format_ref(name, x):

    if isinstance(x, (tuple, list)):
       return format_refs(name, x)

    return f"{format_type(x)} {name} ID = {id(x)}"


def format_refs(name, xs):

    xids = ', '.join(map(lambda v: f"{format_type(v)} ID = {id(v)}", xs))
    return f"{format_type(xs)} {name} : {xids}"


def vjoin(args):

    return '\n  . '.join(filter(None, args))




# --- Representation chain -------------------------------------------------- #

class ReprChain:

   def __init__(self):

       self._str = []


   def _add(self, xstr):

       self._str.append(xstr)
       return self

       
   def typ(self, x):

       return self._add(format_obj(x))

  
   def ref(self, name, x):

       return self._add(format_ref(name, x))
 

   def val(self, name, x):
 
       return self._add(format_val(name, x))   


   def __repr__(self):

       return f"{format_obj(self)}"


   def __str__(self):

       return f"\n{vjoin(self._str)}\n"




