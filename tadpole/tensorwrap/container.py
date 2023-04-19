#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid




###############################################################################
###                                                                         ###
###  Special container types for gradients                                  ###
###                                                                         ###
###############################################################################


# --- Null gradient --------------------------------------------------------- #

class NullGrad(util.Container, tn.Grad):

   # --- Construction --- #

   def __init__(self, size):

       self._size = size


   # --- Grad methods --- #

   def todense(self):

       return zeros(self._size)


   def addto(self, other):

       if isinstance(other, self.__class__):
          return self

       return other.addto(self)


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._size, other._size)

       return bool(log)


   def __hash__(self):

       return hash(self._size)


   def __bool__(self):

       return False


   # --- Container methods --- #

   def __len__(self):

       return self._size


   def __contains__(self, x):

       return x in self.todense()


   def __iter__(self):

       return iter(self.todense())


   def __getitem__(self, pos):

       return self.todense()[pos]




# --- Sparse gradient ------------------------------------------------------- #

class SparseGrad(util.Container, tn.Grad):

   # --- Construction --- #

   def __init__(self, size, pos, vals):

       self._size = size
       self._pos  = pos
       self._vals = vals


   # --- Grad methods --- #

   def todense(self):

       dense  = zeros(self._size)
       source = put(dense._source, self._pos, self._vals) 

       return ContainerGen(source)


   def addto(self, other):

       if not other:
          other = zeros(self._size)

       if isinstance(other, self.__class__):
          other = other.todense()

       assert len(self) == len(other), (
          f"{type(self).__name__}.addto: "
          f"gradient accumulation cannot be performed for "
          f"containers with different sizes {len(self)} != {len(other)}."
       )

       source = put(other._source, self._pos, self._vals, accumulate=True) 
           
       return type(other)(source)  


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._size, other._size)
          log.val(self._pos,  other._pos)
          log.val(self._vals, other._vals)

       return bool(log)


   def __hash__(self):

       return hash((self._size, self._pos, self._vals))


   # --- Container methods --- #

   def __len__(self):

       return self._size


   def __contains__(self, x):

       return x in self.todense()


   def __iter__(self):

       return iter(self.todense())


   def __getitem__(self, pos):

       return self.todense()[pos]




###############################################################################
###                                                                         ###
###  General container and associated functions                             ###
###                                                                         ###
###############################################################################


# --- Basic functionality --------------------------------------------------- #

@ad.nondifferentiable
def size(x):

    return len(x)


@ad.nondifferentiable
def contains(x, item):

    return item in x


@ad.nondifferentiable
def todense(x):

    return x.todense()


@ad.nondifferentiable
def tonull(x):

    return x.tonull()



# --- Iteration ------------------------------------------------------------- #

def iterate(x):

    for pos in range(len(x)):
        yield x[pos]




# --- Element access -------------------------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    try:
       return x.item(pos)
    except (AttributeError, TypeError):
       return x[pos]


@ad.differentiable
def sparsegrad(x, pos, vals):
	
    if isinstance(pos, int):
       pos  = (pos,  )
       vals = (vals, )

    if isinstance(pos, slice):
       pos = tuple(util.range_from_slice(pos))

    return SparseGrad(len(x), pos, vals)

 


# --- Element access VJPs --------------------------------------------------- #

ad.makevjp(getitem,    
              lambda g, out, x, pos: sparsegrad(x, pos, g)
)
 
ad.makevjp(sparsegrad, 
              lambda g, out, x, pos: g[pos]
)




# --- Element access JVPs --------------------------------------------------- #

ad.makejvp(getitem,    "linear")
ad.makejvp(sparsegrad, "linear")




# --- Put data into a container --------------------------------------------- #

def put(data, pos, vals, accumulate=False):

    posvals = zip(pos, vals)
    out     = list(data)
       
    if   accumulate:
         for p, v in posvals:
             out[p].addto(v)
    else:
         for p, v in posvals:
             out[p] = v

    return type(data)(out)




# --- Container of zero tensors --------------------------------------------- #

def zeros(size):

    items = [tn.zeros(tid.Indices()) for i in range(size)]

    return ContainerGen(items)




# --- General container ----------------------------------------------------- #

class ContainerGen(util.Container, tn.Grad):

   # --- Construction --- #

   def __init__(self, source):

       self._source = source


   # --- Grad methods --- #

   def tonull(self):

       return zeros(len(self))


   def todense(self):

       return self


   def addto(self, other):

       if not other:
          return self

       if isinstance(other, SparseGrad):
          return other.addto(self)

       source = type(other._source)(*self._source, *other._source)

       return self.__class__(source)

 
   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._source, other._source)

       return bool(log)


   def __hash__(self):

       return hash(self._source)


   # --- Element access --- #

   def item(self, pos):

       return self._source[pos]


   # --- Container methods --- #

   def __len__(self):

       return len(self._source)


   def __contains__(self, x):

       return x in self._source


   def __iter__(self):

       return iterate(self)


   def __getitem__(self, pos):

       return getitem(self, pos)




