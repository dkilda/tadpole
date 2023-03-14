#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.tensor.identifiers as identifiers
import tadpole.util as util




# --- IndexLike interface --------------------------------------------------- #

class IndexLike(abc.ABC):

   # --- Equality and hashing --- #

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass


   # --- String representation --- #

   @abc.abstractmethod
   def __repr__(self):
       pass

   @abc.abstractmethod
   def __str__(self):
       pass


   # --- Index space --- #

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __reversed__(self):
       pass


   # --- Out of place modifiers --- #

   @abc.abstractmethod
   def resized(self, size, **opts):
       pass

   @abc.abstractmethod
   def renamed(self, name, **opts):
       pass




# --- Index ----------------------------------------------------------------- #

# TODO impl ConnectedIndex decorator with all the partner/connectivity info?

class Index(IndexLike): 

   # --- Construction --- #

   def __init__(self, name, size=1, uuid=None):

       if uuid is None:
          uuid = identifiers.new_uuid()

       self._name = name
       self._size = size
       self._uuid = uuid


   # --- Copying (forbidden to enforce uniqueness) --- #

   def __copy__(self):

       return self


   def __deepcopy__(self, memdict={}):

       return self


   # --- Equality and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._uuid, other._uuid)

       if bool(log):
          assert self._size == other._size,
             f"Index.__eq__(): indices {self} and {other} are equal but "
             f"their sizes {self._size} != {other._size} do not match!"
          
       return bool(log) 


   def __hash__(self):

       return hash(self._uuid)  


   # --- String representation --- #

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)

       rep.val("name", self._name)
       rep.val("size", self._size)
       rep.val("uuid", self._uuid)

       return str(rep)


   def __str__(self):

       return self._name


   # --- Index space --- #

   def __len__(self):
 
       return self._size


   def __iter__(self):

       return iter(range(self._size))


   def __reversed__(self):

       return reversed(iter(self))


   # --- Out of place modifiers --- #

   def resized(self, size, **opts):

       return self.__class__(self._name, size, **opts)


   def renamed(self, name, **opts):

       return self.__class__(name, self._size, **opts)




# --- ShapeFromInds --------------------------------------------------------- #

class ShapeFromInds(util.TupleLike):

   def __init__(self, *inds):

       self._inds = inds


   @property
   @util.cacheable
   def _sizes(self):

       return tuple(map(len, self._inds)) 


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
          log.val(self._sizes, other._sizes)

       return bool(log)


   def __hash__(self):

       return hash(self._sizes)


   def __len__(self):

       return len(self._sizes)


   def __contains__(self, x):

       return x in self._sizes


   def __iter__(self):

       return iter(self._sizes)  


   def __reversed__(self):

       return reversed(self._sizes)  


   def __getitem__(self, idx):

       return self._sizes[idx]





# --- Indices --------------------------------------------------------------- #

class Indices(util.TupleLike):

   def __init__(self, *inds):

       self._inds = inds
       self._axis_by_name = {}










"""
class ShapeFromInds(util.TupleLike):

   def __init__(self, *inds):

       self._inds = inds


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._inds, other._inds)

       return bool(log)


   def __hash__(self):

       return hash(self._inds)


   def __len__(self):

       return len(self._inds)


   def __contains__(self, x):

       return x in self._inds


   def __iter__(self):

       return iter(self._inds)  


   def __reversed__(self):

       return reversed(self._inds)  


   def __getitem__(self, idx):

       return self._inds[idx]
"""

































