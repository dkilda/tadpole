#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import tadpole.util        as util
import tadpole.index.uuids as uuids


from tadpole.index.types import (
   Index,
)




###############################################################################
###                                                                         ###
###  Tensor index                                                           ###
###                                                                         ###
###############################################################################


# --- General Index --------------------------------------------------------- #

class IndexGen(Index): 

   # --- Construction --- #

   def __init__(self, tags, size=1, uuid=None):

       if uuid is None:
          uuid = uuids.next_uuid()

       self._tags = tags
       self._size = size
       self._uuid = uuid


   # --- Copying (forbidden to enforce uniqueness) --- #

   def __copy__(self):

       return self


   def __deepcopy__(self, memdict={}):

       return self


   # --- String representation --- #

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)

       rep.val("tags", self._tags)
       rep.val("size", self._size)
       rep.val("uuid", self._uuid)

       return str(rep)


   # --- Equality, hashing, size --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._uuid, other._uuid)

       if bool(log):
          assert self._size == other._size, (
             f"{type(self).__name__}.__eq__: "
             f"indices {self} and {other} are equal but their "
             f"sizes {self._size} != {other._size} do not match!"
          )
          
       return bool(log) 


   def __hash__(self):

       return hash(self._uuid)  


   def __len__(self):
 
       return self._size


   # --- General methods --- #

   def all(self, *tags):

       return len(set(tags) - set(self._tags)) == 0


   def any(self, *tags):

       return len(set(tags) & set(self._tags)) > 0


   def resized(self, start, end):

       uuid = self._uuid

       if "_@start" in uuid:
          uuid = uuid[: uuid.index("_@start")]

       uuid = f"{self._uuid}_@start_{start}_@end_{end}"

       return self.__class__(self._tags, end - start, uuid=uuid)


   def retagged(self, tags):

       return self.__class__(tags, self._size)




# --- Literal Index --------------------------------------------------------- #

class IndexLit(Index): 

   # --- Construction --- #

   def __init__(self, name, size=1):

       self._name   = name
       self._origin = IndexGen(name, size, uuid=name)


   # --- Copying (forbidden to enforce uniqueness) --- #

   def __copy__(self):

       return self


   def __deepcopy__(self, memdict={}):

       return self


   # --- String representation --- #

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)

       rep.val("name", self._name)
       rep.val("size", len(self))
       rep.val("uuid", self._name)

       return str(rep)


   # --- Equality, hashing, size --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._origin, other._origin)

       return bool(log) 


   def __hash__(self):

       return hash(self._origin)  


   def __len__(self):
 
       return len(self._origin)


   # --- General methods --- #

   def all(self, *tags):

       return self._origin.all(*tags)


   def any(self, *tags):

       return self._origin.any(*tags)


   def resized(self, start, end):

       return self._origin.resized(start, end)


   def retagged(self, tags):

       return self.__class__(tags, len(self))




# --- Singleton Index ------------------------------------------------------- #

class IndexOne(Index):

   # --- Copying (forbidden to enforce uniqueness) --- #

   def __copy__(self):

       return self


   def __deepcopy__(self, memdict={}):

       return self


   # --- String representation --- #

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)

       return str(rep)


   # --- Equality, hashing, size --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)
          
       return bool(log) 


   def __hash__(self):

       return 1 


   def __len__(self):
 
       return 1


   # --- General methods --- #

   def all(self, *tags):

       return False


   def any(self, *tags):

       return False


   def resized(self, start, end):

       return IndexLit(f"@ONE_@start_{start}_@end_{end}", end - start)


   def retagged(self, tags):

       return IndexGen(tags)




###############################################################################
###                                                                         ###
###  Collection of tensor indices with extra functionality                  ###
###  (operations acting on groups of Index objects).                        ###
###                                                                         ###
###############################################################################


# --- Indices --------------------------------------------------------------- #

class Indices(util.Container):

   # --- Construction --- #

   def __init__(self, *inds):

       self._inds = inds


   # --- Copying --- #

   def copy(self):

       return self.__class__(*self._inds)


   __copy__ = copy


   # --- String representation --- #

   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       rep.val("inds", self._inds)

       return str(rep) 


   # --- Container-like methods --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
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


   # --- Properties --- #

   @property
   def size(self):
       return int(np.prod(self.shape))

   @property
   def ndim(self):
       return len(self._inds)

   @property
   def shape(self):
       return tuple(map(len, self._inds))


   # --- Index container behavior --- #

   def all(self, *tags):

       return tuple(filter(lambda x: x.all(*tags), self._inds))  


   def any(self, *tags):

       return tuple(filter(lambda x: x.any(*tags), self._inds))


   def _map(self, tags):

       if isinstance(tags, Index):
          return (tags, )

       return self.all(*tags)


   def map(self, *tags):

       out = []

       for xtags in tags:
           out.extend(self._map(xtags))

       return tuple(out)


   def axes(self, *inds):

       return tuple(self._inds.index(ind) for ind in self.map(*inds))


   # --- Out-of-place modifications --- #

   def remove(self, *inds):

       return self.__class__(*(ind for ind in self if ind not in inds))


   def add(self, *inds, axis=0):

       newinds = list(self)

       for ind in reversed(inds):
           newinds.insert(axis, ind)

       return self.__class__(*newinds)


   def push(self, *inds):

       return self.add(*inds, axis=len(self))


   # --- Set arithmetic --- #

   def __and__(self, other):

       return self.remove(*self.remove(*other))


   def __or__(self, other):

       return self.push(*other)


   def __xor__(self, other):

       return self.remove(*other) 




###############################################################################
###                                                                         ###
###  Index operations                                                       ###
###                                                                         ###
###############################################################################


# --- Basic index info ------------------------------------------------------ #

def shapeof(*inds):

    return Indices(*inds).shape



def sizeof(*inds):

    if len(inds) == 1:
       return len(inds[0])

    return Indices(*inds).size




