#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.array    as ar
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid

from tadpole.tensorwrap.types import (
   Container,
   TensorContainer,
)




###############################################################################
###                                                                         ###
###  Container space                                                        ###
###                                                                         ###
###############################################################################


# --- ContainerSpace -------------------------------------------------------- #

class ContainerSpace(Container, tn.Space):

   # --- Construction --- #

   def __init__(self, spaces):

       if not isinstance(spaces, tuple):
          spaces = tuple(spaces)

       self._spaces = spaces
 

   # --- Private helpers --- #

   def _map(self, fun):

       def funwrap(x, i):
           try:
              return fun(x, i)
           except TypeError:
              return fun(x)

       return tuple(
          funwrap(space, i) for i, space in enumerate(self._spaces)
       )


   def _transform(self, fun):

       return self.__class__(self._map(fun)) 


   def _apply(self, fun):

       return ContainerGen(self._map(fun)) 


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._spaces, other._spaces)

       return bool(log)


   def __hash__(self):

       return hash(self._spaces)


   # --- Fill space with data --- #

   def fillwith(self, data):

       return self._apply(
          lambda x, i: x.fillwith(data[i])
       ) 


   # --- Reshape space --- #

   def reshape(self, inds):

       return self._transform(
          lambda x, i: x.reshape(inds[i])
       ) 


   # --- Gradient factories --- #

   def sparsegrad(self, pos, vals):

       return SparseGrad(self, pos, vals)
       

   def nullgrad(self):

       return NullGrad(self)


   # --- Container factories --- #

   def zeros(self):

       return self._apply(
          lambda x: x.zeros()
       )


   def ones(self):

       return self._apply(
          lambda x: x.ones()
       )


   def unit(self, pos, **opts):

       return self._apply(
          lambda x: x.unit(pos[i], **util.listofdicts(opts)[i])
       )


   def eye(self, lind=None, rind=None):

       if lind is None: lind = [None]*len(self)
       if rind is None: rind = [None]*len(self)

       return self._apply(
          lambda x: x.eye(lind[i], rind[i])
       )


   def rand(self, **opts):

       return self._apply(
          lambda x: x.rand(**util.listofdicts(opts)[i])
       )


   def randn(self, **opts):

       return self._apply(
          lambda x: x.randn(**util.listofdicts(opts)[i])
       )


   def randuniform(self, boundaries, **opts):

       return self._apply(
          lambda x: x.randuniform(boundaries[i], **util.listofdicts(opts)[i])
       )


   def units(self, **opts):

       return self._apply(
          lambda x: x.units(**util.listofdicts(opts)[i])
       )


   def basis(self, **opts):

       return self._apply(
          lambda x: x.basis(**util.listofdicts(opts)[i])
       )


   # --- Space properties --- #

   @property
   def dtype(self):
       return self._map(lambda x: x.dtype)

   @property
   def size(self):
       return self._map(lambda x: x.size)

   @property 
   def ndim(self):
       return self._map(lambda x: x.ndim)

   @property
   def shape(self):
       return self._map(lambda x: x.shape)


   # --- Container methods --- #

   def __len__(self):

       return len(self._spaces)


   def __contains__(self, x):

       return x in self._spaces


   def __iter__(self):

       return iter(self._spaces)


   def __getitem__(self, pos):

       return self._spaces[pos]




###############################################################################
###                                                                         ###
###  Special container types for gradients                                  ###
###                                                                         ###
###############################################################################


# --- Null gradient --------------------------------------------------------- #

class NullGrad(TensorContainer, tn.Grad):

   # --- Construction --- #

   def __init__(self, space):

       self._space = space


   # --- Grad methods --- #

   def addto(self, other):

       if not other:
          return self

       return other.addto(self)


   def todense(self):

       return self._space.zeros()


   def tonull(self):

       return self


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       return bool(log)


   def __hash__(self):

       return hash(self._space)


   def __bool__(self):

       return False


   # --- Container methods --- #

   def copy(self, **opts):

       return self.__class__(self._space) 


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return self._space 


   def item(self, pos):

       return self.todense().item(pos)


   def __len__(self):

       return len(self._space)


   def __contains__(self, x):

       return x in self.todense()


   def __iter__(self):

       return iter(self.todense())


   def __getitem__(self, pos):

       return self.todense()[pos]




# --- Sparse gradient ------------------------------------------------------- #

class SparseGrad(TensorContainer, tn.Grad):

   # --- Construction --- #

   def __init__(self, space, pos, vals):

       self._space = space
       self._pos   = pos
       self._vals  = vals


   # --- Grad methods --- #

   def addto(self, other):

       if not other:
          other = self.space().zeros()

       if isinstance(other, self.__class__):
          other = other.todense()

       assert len(self) == len(other), (
          f"{type(self).__name__}.addto: "
          f"gradient accumulation cannot be performed for containers "
          f"with non-matching spaces {self.space()} != {other.space()}"
       )

       data = put(other._data, self._pos, self._vals, accumulate=True) 
           
       return self.__class__(data)


   def todense(self):

       return put(self.space().zeros(), self._pos, self._vals) 


   def tonull(self):

       return NullGrad(self.space())


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._space, other._space)
          log.val(self._pos,   other._pos)

       if bool(log):
          log.val(self._vals,  other._vals)

       return bool(log)


   def __hash__(self):

       return hash((self._space, self._pos, self._vals))


   # --- Container methods --- #

   def copy(self, **opts):

       return self.__class__(self._space, self._pos, self._vals) 


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return self._space 


   def item(self, pos):

       return self.todense().item(pos)


   def __len__(self):

       return len(self._space)


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


# --- General container ----------------------------------------------------- #

class ContainerGen(TensorContainer, tn.Grad):

   # --- Construction --- #

   def __init__(self, data):

       if not isinstance(data, tuple):
          data = tuple(data)

       self._data = data


   # --- Grad methods --- #

   def addto(self, other):

       if not other:
          return self

       if isinstance(other, SparseGrad):
          return other.addto(self)

       data = tuple(y.addto(x) for y, x in zip(other._data, self._data))
           
       return self.__class__(data)  


   def todense(self):

       return self


   def tonull(self):

       return NullGrad(self.space())


   # --- Comparison and hashing --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._data, other._data)

       return bool(log)


   def __hash__(self):

       return hash(self._data)


   # --- Container methods --- #

   def copy(self, **opts):

       return self.__class__(tuple(x.copy(**opts) for x in self._data))


   def withdata(self, data):

       return self.space().fillwith(data)


   def space(self):

       return ContainerSpace(tuple(x.space() for x in self._data))


   def item(self, pos):

       return self._data[pos]


   def __len__(self):

       return len(self._data)


   def __contains__(self, x):

       return x in self._data


   def __iter__(self):

       return iterate(self)


   def __getitem__(self, pos):

       return getitem(self, pos)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Container methods                ###
###                                                                         ###
###############################################################################


# --- Gradient functions ---------------------------------------------------- #

@ad.differentiable
def addgrads(x, y):

    return y.addto(x)


@ad.nondifferentiable
def todense(x):

    return x.todense()


@ad.nondifferentiable
def tonull(x):

    return x.tonull()




# --- Container functions --------------------------------------------------- #

@ad.nondifferentiable
def copy(x, **opts):

    return x.copy(**opts)


@ad.nondifferentiable
def withdata(x, data):

    return x.withdata(data) 


@ad.nondifferentiable
def space(x):

    return x.space()


@ad.nondifferentiable
def item(x, pos):

    return x.item(pos)


@ad.nondifferentiable
def size(x):

    return len(x)


@ad.nondifferentiable
def contains(x, item):

    return item in x


def iterate(x):

    for pos in range(len(x)):
        yield x[pos]


@ad.differentiable
def getitem(x, pos):

    try:
       return x.item(pos)
    except (AttributeError, TypeError):
       return x[pos]




# --- Put data into a container --------------------------------------------- #

def put(data, pos, vals, accumulate=False):

    posvals = zip(pos, vals)
    out     = list(data)
       
    if   accumulate:
         for p, v in posvals:
             out[p] = out[p].addto(v)
    else:
         for p, v in posvals:
             out[p] = v

    return type(data)(out)




# --- Apply unary function -------------------------------------------------- #

def apply_unary(fun, x):

    return ContainerGen(tuple(map(fun, x)))




# --- Apply binary function ------------------------------------------------- #

def apply_binary(fun, x, y):

    return ContainerGen(tuple(map(fun, zip(x, y))))




###############################################################################
###                                                                         ###
###  Container VJPs and JVPs                                                ###
###                                                                         ###
###############################################################################

 
# --- Element access VJPs --------------------------------------------------- #

@ad.differentiable
def sparsegrad(x, pos, space):
	
    if isinstance(pos, int):
       pos = [pos]
       x   = [x]

    if isinstance(pos, slice):
       pos = tuple(util.range_from_slice(pos))

    return space.sparsegrad(pos, x) 




ad.makevjp(getitem,    
              lambda g, out, x, pos: sparsegrad(g, pos, len(x))
)

 
ad.makevjp(sparsegrad, 
              lambda g, out, x, pos, size: g[pos]
)




# --- Element access JVPs --------------------------------------------------- #

ad.makejvp(getitem,    "linear")
ad.makejvp(sparsegrad, "linear")




