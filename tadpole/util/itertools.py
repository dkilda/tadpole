#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import itertools
import cytoolz

import tadpole.util as util




###############################################################################
###                                                                         ###
###  General content manipulation tools                                     ###
###                                                                         ###
###############################################################################


# --- Frequencies of items in an iterable ----------------------------------- #

def frequencies(xs):

    return cytoolz.frequencies(xs)




# --- Concatenate an iterable of iterables ---------------------------------- #

def concat(xs):

    return cytoolz.concat(xs)




# --- Select unique items from an iterable ---------------------------------- #

def unique(xs):

    xs = list(xs)
    return relsort(set(xs), xs)




# --- Relative complement of xs wrt ys -------------------------------------- #

def complement(xs, ys):

    xs = list(xs)
    return relsort(set(xs) - set(ys), xs)  




# --- Create range from slice ----------------------------------------------- #

def range_from_slice(x):

    return range(*x.indices(x.stop)) 




###############################################################################
###                                                                         ###
###  Sorting tools                                                          ###
###                                                                         ###
###############################################################################


# --- Get a list of indices that sort an iterable  -------------------------- #

def argsort(xs):

    return sorted(range(len(xs)), key=xs.__getitem__)




# --- Relative sort of xs wrt ys -------------------------------------------- #

def relsort(xs, ys):

    return list(sorted(xs, key=ys.index))




###############################################################################
###                                                                         ###
###  Mapping tools                                                          ###
###                                                                         ###
###############################################################################


# --- Identity dictionary (each key equals its corresponding value) --------- #

def identity_dict(dct):

    return len(set(zip(*dct.items()))) == 1




# --- Inverted dictionary --------------------------------------------------- #

def inverted_dict(dct):

    inverted_dct = {}

    for k, v in dct.items():
        inverted_dct[v] = inverted_dct.get(v, tuple()) + (k,)

    return inverted_dct




# --- Unpacked dictionary --------------------------------------------------- #

def unpacked_dict(dct):

    def container(x):

        if isinstance(x, (tuple, list, util.Container)):
           return x

        return (x,)

    return {key: val for keys, val in dct.items() for key in container(keys)}




# --- Convert to list-of-dicts ---------------------------------------------- #

def listofdicts(dct):

    return [dict(zip(dct, v)) for v in zip(*dct.values())] 



         
###############################################################################
###                                                                         ###
###  Customized loop iterator.                                              ###
###  Defined by the first item and the next and stop functions, instead of  ###
###  a range. Can be traversed in forward and reverse directions, keeps     ###
###  track of the last item of the loop.                                    ###
###                                                                         ###
###############################################################################


# --- Loop iterator --------------------------------------------------------- #

class Loop:

   def __init__(self, first, next, stop):

       self._first = first
       self._next  = next
       self._stop  = stop


   @util.cacheable
   def _items(self):

       return list(self)


   @util.cacheable
   def _reversed(self):

       return reversed(self._items())

       
   def __iter__(self):

       x = self._first

       for _ in itertools.count():

           yield x

           if self._stop(x):
              break

           x = self._next(x)


   def __reversed__(self):

       return iter(self._reversed())


   def first(self):

       return self._first  


   def last(self):
 
       return next(reversed(self)) 


   def size(self):

       return len(self._items())

       
   def once(self):

       return self.size() == 1




###############################################################################
###                                                                         ###
###  Sequence data structure                                                ###
###                                                                         ###
###############################################################################


# --- List with history ----------------------------------------------------- #

class List:

   def __init__(self, origin):

       if not isinstance(origin, list):
          origin = list(origin)

       self._origin  = origin
       self._history = tuple()


   def __iter__(self):

       return iter(self._origin)


   def apply(self, tasks):

       self.restore()

       for task in tasks:
           task.execute(self._origin) 

       self._history = tuple(tasks)
       return self
 

   def restore(self):

       for task in reversed(self._history):
           task.undo(self._origin)

       self._history = tuple()
       return self




# --- Sequence -------------------------------------------------------------- #

class Sequence:

   def __init__(self, origin=None, tasks=tuple()):

       if not origin:
          origin = []

       if not isinstance(origin, List):
          origin = List(origin)

       self._origin = origin
       self._tasks  = tasks


   @property
   @util.cacheable
   def _list(self):

       self._origin.apply(self._tasks)

       return list(self._origin)


   def _apply(self, task):

       return self.__class__(self._origin, (*self._tasks, task))


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("items", self._list)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._list, other._list)

       return bool(log)


   def __hash__(self):

       return id(self)

 
   def __iter__(self):

       return iter(self._list)


   def __reversed__(self):

       return reversed(self._list)


   def __len__(self):

       return len(self._list)


   def __contains__(self, item):

       return item in self._list


   def __getitem__(self, idx):

       return self._list[idx]


   def push(self, item):

       return self._apply(Push(item))


   def pop(self):

       return self._apply(Pop())




###############################################################################
###                                                                         ###
###  Sequence tasks operating on the internal list of a Sequence            ###
###                                                                         ###
###############################################################################


# --- Task interface -------------------------------------------------------- #

class Task(abc.ABC):

   @abc.abstractmethod
   def execute(self, lst):
       pass

   @abc.abstractmethod
   def undo(self, lst):
       pass




# --- Push task ------------------------------------------------------------- #

class Push(Task):

   def __init__(self, item):

       self._item = item


   def execute(self, lst):

       lst.append(self._item)


   def undo(self, lst):

       lst.remove(self._item)




# --- Pop task -------------------------------------------------------------- #

class Pop(Task):      
       
   def __init__(self):

       self._item = None 


   def execute(self, lst):
 
       self._item = lst.pop()


   def undo(self, lst):

       lst.append(self._item)




