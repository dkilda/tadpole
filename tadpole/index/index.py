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


   # --- Equality and hashing --- #

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


   # --- Index space --- #

   def __len__(self):
 
       return self._size


   def __iter__(self):

       return iter(range(self._size))


   def __reversed__(self):

       return reversed(range(self._size))


   # --- General methods --- #

   def matches_all(self, *tags):

       return len(set(tags) - set(self._tags)) == 0


   def matches_any(self, *tags):

       return len(set(tags) & set(self._tags)) > 0


   def resized(self, size, **opts):

       return self.__class__(self._tags, size, **opts)


   def retagged(self, tags, **opts):

       return self.__class__(tags, self._size, **opts)




###############################################################################
###                                                                         ###
###  Collection of tensor indices with extra functionality                  ###
###  (operations acting on groups of Index objects).                        ###
###                                                                         ###
###############################################################################


# --- Axis map -------------------------------------------------------------- #

class AxisMap:

   def __init__(self, source):

       self._source = source

  
   @util.cacheable
   def _axismap(self):

       ind_by_axis = dict(zip(range(len(self._source)), self._source))
       axes_by_ind = util.inverted_dict(ind_by_axis)    

       def _iter(x):

           if isinstance(x, Index):
              return itertools.repeat(x)

           return iter(x)

       return {ind: _iter(axes) for ind, axes in axes_by_ind.items()}


   def __getitem__(self, ind):

       return next(self._axismap()[ind])




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
       return np.prod(self.shape)

   @property
   def ndim(self):
       return len(self._inds)

   @property
   def shape(self):
       return tuple(map(len, self._inds))


   # --- Index container behavior --- #

   def all(self, *tags):

       return tuple(filter(lambda x: x.matches_all(*tags), self._inds))  


   def any(self, *tags):

       return tuple(filter(lambda x: x.matches_any(*tags), self._inds))


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

       axismap = AxisMap(self._inds)

       return tuple(axismap[ind] for ind in self.map(*inds))


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




