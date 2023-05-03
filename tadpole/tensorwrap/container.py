#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.array    as ar
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid




###############################################################################
###                                                                         ###
###  Container space                                                        ###
###                                                                         ###
###############################################################################


# --- ContainerSpace -------------------------------------------------------- #

class ContainerSpace(util.Container, tn.Space):

   # --- Construction --- #

   def __init__(self, spaces):

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
          lambda x, i: x.fillwith(inds[i])
       ) 


   # --- Gradient factories --- #

   def sparsegrad(self, pos, vals):

       return self._apply(
          lambda x, i: x.sparsegrad(pos[i], vals[i])
       )


   def nullgrad(self):

       return self._apply(
          lambda x: x.nullgrad()
       )


   # --- Tensor factories --- #

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

class NullGrad(util.Container, tn.Grad):

   # --- Construction --- #

   def __init__(self, size):

       self._size = size


   # --- Grad methods --- #

   def tonull(self):

       return self


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

   def tonull(self):

       return zeros(len(self))


   def todense(self):

       source = put(zeros(self._size), self._pos, self._vals) 

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



# --- Gradient accumulation ------------------------------------------------- #

@ad.differentiable
def addgrads(x, y):

    return y.addto(x)




# --- Iteration ------------------------------------------------------------- #

def iterate(x):

    for pos in range(len(x)):
        yield x[pos]




# --- Element access -------------------------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    print("GETITEM: ", x.item(pos), pos)

    try:
       return x.item(pos)
    except (AttributeError, TypeError):
       return x[pos]


@ad.differentiable
def sparsegrad(x, pos, size):
	
    if isinstance(pos, int):
       pos = (pos,  )
       x   = (x,    )

    if isinstance(pos, slice):
       pos = tuple(util.range_from_slice(pos))

    print("SPARSEGRAD: ", x, pos, size)

    return SparseGrad(size, pos, x)

 


# --- Element access VJPs --------------------------------------------------- #

ad.makevjp(getitem,    
              lambda g, out, x, pos: sparsegrad(g, pos, len(x))
)

 
ad.makevjp(sparsegrad, 
              lambda g, out, x, pos, size: g[pos]
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
             out[p] = out[p].addto(v)
    else:
         for p, v in posvals:
             out[p] = v

    return type(data)(out)




# --- Container of zero tensors --------------------------------------------- #

def zeros(size):

    return ContainerGen(tuple(tn.NullGrad() for _ in range(size)))




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

       source = (y.addto(x) for y, x in zip(other._source, self._source))
           
       return self.__class__(type(other._source)(source))

 
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




