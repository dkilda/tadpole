#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.autodiff.util as tdutil




###############################################################################
###                                                                         ###
###  Sequence data structure                                                ###
###                                                                         ###
###############################################################################


# --- List with history ----------------------------------------------------- #

class List:

   def __init__(self, origin):

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
   @tdutil.cacheable
   def _list(self):

       self._origin.apply(self._tasks)
       return list(self._origin)


   def _apply(self, task):

       return self.__class__(self._origin, (*self._tasks, task))

 
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





