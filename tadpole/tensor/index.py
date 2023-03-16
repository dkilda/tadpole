#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util         as util
import tadpole.tensor.uuids as uuids




###############################################################################
###                                                                         ###
###  Tensor index                                                           ###
###                                                                         ###
###############################################################################


# --- IndexLike interface --------------------------------------------------- #

class IndexLike(abc.ABC):

   # --- String representation --- #

   @abc.abstractmethod
   def __repr__(self):
       pass


   # --- Equality and hashing --- #

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
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


   # --- General methods --- #

   @abc.abstractmethod
   def answers(self, name):
       pass

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
          uuid = uuids.next()

       self._name = name
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

       rep.val("name", self._name)
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
          assert self._size == other._size,
             f"Index.__eq__(): indices {self} and {other} are equal but "
             f"their sizes {self._size} != {other._size} do not match!"
          
       return bool(log) 


   def __hash__(self):

       return hash(self._uuid)  


   # --- Index space --- #

   def __len__(self):
 
       return self._size


   def __iter__(self):

       return iter(range(self._size))


   def __reversed__(self):

       return reversed(iter(self))


   # --- General methods --- #

   def answers(self, name):

       return self._name == name


   def resized(self, size, **opts):

       return self.__class__(self._name, size, **opts)


   def renamed(self, name, **opts):

       return self.__class__(name, self._size, **opts)




###############################################################################
###                                                                         ###
###  Collection of tensor indices with extra functionality                  ###
###  (operations acting on groups of Index objects).                        ###
###                                                                         ###
###############################################################################


# --- Indices --------------------------------------------------------------- #

class Indices(util.TupleLike):

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


   # --- TupleLike methods --- #

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

   def inds(self, name):

       return tuple(filter(lambda x: x.answers(name), self._inds))  


   def ind(self, name):

       return self.inds(name)[0]


   def axis(self, ind):

       return self._inds.index(ind)


   # --- Out-of-place modifications --- #

   def remove(self, *inds):

       return self.__class__(*(for ind in self if ind not in inds))


   def add(self, *inds, axis=0):

       newinds = list(self)
       for ind in reversed(inds):
           newinds.insert(axis, ind)

       return self.__class__(*newinds)


   def push(self, *inds):

       return self.add(*inds, axis=len(self))




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




# --- Index transformations (preserving the number of inds) ----------------- #

def reindex(inds, indmap):

    newinds = list(inds)

    for i, ind in enumerate(inds):

        try:
            newinds[i] = indmap[ind]
        except KeyError:
            pass

    return type(inds)(*newinds) 




def transpose(inds, *order):

    assert set(inds) == set(order),
       f"index.transpose(): input and output must contain the same "
       f"set of indices, but input {inds} does not match output {order}."

    return type(inds)(*order)




# --- Index transformations: (changing the number of inds) ------------------ #

def fuse(inds, fusemap):

    if isinstance(fusemap, dict):
       fusemap = fusemap.items()

    for inp, out in fusemap:

        if not isinstance(inp, Index):
           inp = Index(inp, sizeof(*inp))

        assert sizeof(*inp) == sizeof(out), (
           f"index.fuse(): input {inp} and output {out} must have "
           f"matching sizes, but output size {sizeof(out)} != net "
           f"input size {sizeof(*inp)} = prod({shapeof(*inp)})." 
        )

        inds = inds.remove(*inp).add(out)

    return inds

       

 
def split(inds, splitmap):

    if isinstance(splitmap, dict):
       splitmap = splitmap.items()

    for inp, out in splitmap:

        assert sizeof(inp) == sizeof(*out), (
           f"index.split(): input {inp} and output {out} must have "
           f"matching sizes, but output size {sizeof(inp)} != net input "
           f"size {sizeof(*out)} = prod({shapeof(*out)})." 
        )

        axis = inds.axis(inp)
        inds = inds.remove(inp).add(*out, axis=axis)

    return inds




def squeeze(inds):

    return inds.remove(*filter(lambda x: len(x) == 1, inds))




def unsqueeze(inds, names):

    return inds.add(*(Index(name) for name in names))




