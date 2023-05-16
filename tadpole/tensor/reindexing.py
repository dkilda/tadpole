#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core           as core
import tadpole.tensor.elemwise_unary as unary


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.engine import (
   EngineUnary,
   EngineElemwise,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.index import (
   Index,
   IndexGen, 
   IndexLit,
   Indices,
)




###############################################################################
###                                                                         ###
###  Helpers                                                                ###
###                                                                         ###
###############################################################################


# --- Reindexing map -------------------------------------------------------- #

class ReindexMap:

   def __init__(self, source):

       self._source = source

  
   @util.cacheable
   def _indmap(self):

       def _iter(x):

           if isinstance(x, Index):
              return itertools.repeat(x)

           return iter(x)

       return {inp: _iter(out) for inp, out in self._source.items()}


   def __getitem__(self, inp):

       return next(self._indmap()[inp])
       
       


###############################################################################
###                                                                         ###
###  Tensor reindexing engine and operator                                  ###
###                                                                         ###
###############################################################################


# --- Tensor reindexing factory --------------------------------------------- #

def tensor_reindex(x):

    engine = x.pluginto(EngineReindex())
    return engine.operator()




# --- Tensor reindexing engine ---------------------------------------------- #

class EngineReindex(Engine):

   def __init__(self, source=None):

       if source is None:
          source = EngineUnary(TensorReindex)

       self._source = source


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._source, other._source)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._source.attach(data, inds))


   def operator(self):

       return self._source.operator()




# --- Tensor reindexing operator -------------------------------------------- #

class TensorReindex:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _map(self, *inds):

       return self._inds.map(*inds)


   def _new(self, data, inds):

       if not isinstance(inds, Indices):
          inds = Indices(*inds)

       return core.TensorGen(data, inds)


   # --- Reindexing and reshaping methods --- #

   def reindex(self, indmap):

       indmap      = ReindexMap(indmap) # FIXME remove ReindexMap?
       output_inds = list(self._inds)

       for i, ind in enumerate(self._inds):
           try:
               output_inds[i] = indmap[ind] 
           except KeyError:
               pass

       return self._new(self._data, output_inds) 


   def transpose(self, *output_inds):

       if len(output_inds) == 0:
          output_inds = tuple(reversed(self._inds))

       output_inds = self._map(*output_inds)

       assert set(self._inds) == set(output_inds), (
          f"{type(self).__name__}.transpose: "
          f"the destination indices {output_inds} are not "
          f"compatible with the source indices {self._inds}."
       )

       data = ar.transpose(self._data, self._inds.axes(*output_inds))
       return self._new(data, output_inds)


   def fuse(self, fusemap):

       fused   = self._inds 
       unfused = self._inds

       for inp, out in reversed(fusemap.items()):
           
           inp = self._map(*inp)

           if not isinstance(out, Index):
              out = IndexGen(out, tid.sizeof(*inp))

           assert tid.sizeof(*inp) == tid.sizeof(out), (
               f"{type(self).__name__}.fuse: "
               f"sizes of input indices {inp} and output index {out} must "
               f"match, but the input size {tid.sizeof(*inp)} != "
               f"the output size {tid.sizeof(out)}."
           )

           fused   = fused.remove(*inp).add(out)
           unfused = unfused.remove(*inp).add(*inp)

       data = ar.transpose(self._data, self._inds.axes(*unfused)) 
       data = ar.reshape(data, fused.shape)

       return self._new(data, fused)


   def split(self, splitmap):

       inds = self._inds

       for inp, out in splitmap.items():                   
       
           inp, = self._map(inp)

           assert tid.sizeof(inp) == tid.sizeof(*out), (
              f"{type(self).__name__}.split: "
              f"sizes of input index {inp} and output indices {out} must "
              f"match, but the input size {tid.sizeof(inp)} != "
              f"the output size {tid.sizeof(*out)}."
           )

           axis, = inds.axes(inp)
           inds  = inds.remove(inp).add(*out, axis=axis)

       data = ar.reshape(self._data, inds.shape)
       return self._new(data, inds)


   def squeeze(self, inds=None):

       if   inds is None:
            singletons = [ind for ind in self._inds if len(ind) == 1]   
  
       else:
            singletons = self._inds.map(*inds)

            assert all(len(ind) == 1 for ind in singletons), (
               f"{type(self).__name__}.squeeze: "
               f"Cannot squeeze an input index with size != 1!"
               f"Sizes of all input indices: {tid.shapeof(*singletons)}."
            ) 

       inds = self._inds.remove(*singletons)
       data = ar.squeeze(self._data, self._inds.axes(*singletons))

       return self._new(data, inds)


   def unsqueeze(self, inds):

       singletons = self._map(*inds)

       output_inds = self._inds.add(*singletons)
       output_data = ar.unsqueeze(self._data, output_inds.axes(*singletons))

       return self._new(output_data, output_inds)  


   def expand(self, inds):    

       output_inds = self._inds.add(*inds) 
       output_data = ar.broadcast_to(self._data, output_inds.shape)

       return self._new(output_data, output_inds) 


   def flatten(self, ind=None):

       if ind is None:
          ind = "flat"    

       return self.fuse({self._inds: ind})


   def diag(self, ind=None): # TODO move diag to linalg!

       if ind is None:
          ind = self._inds[0]

       if not isinstance(ind, Index):
          ind = IndexGen(ind, len(self._inds[0]))

       if self._inds.ndim == 1:
          return core.TensorGen(ar.diag(self._data), (ind, *self._inds))

       if self._inds.ndim == 2:
          return core.TensorGen(ar.diag(self._data), (ind,))

       raise ValueError(
          f"TensorReindex.diag: "
          f"diag is only supported for tensors with "
          f"ndim = 1 or 2, but ndim != {self._inds.ndim}."
       )




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorReindex methods            ###
###                                                                         ###
###############################################################################


# --- Reindexing and reshaping methods -------------------------------------- #

@ad.differentiable
def reindex(x, indmap):

    op = tensor_reindex(x)
    return op.reindex(indmap)


@ad.differentiable
def transpose(x, *output_inds):

    op = tensor_reindex(x)
    return op.transpose(*output_inds)


def htranspose(x, *output_inds):

    return transpose(unary.conj(x), *output_inds)


@ad.differentiable
def fuse(x, fusemap):

    op = tensor_reindex(x)
    return op.fuse(fusemap)


@ad.differentiable
def split(x, splitmap):

    op = tensor_reindex(x)
    return op.split(splitmap)


@ad.differentiable
def squeeze(x, inds=None):

    op = tensor_reindex(x)
    return op.squeeze(inds)


@ad.differentiable
def unsqueeze(x, inds):

    op = tensor_reindex(x)
    return op.unsqueeze(inds)


@ad.differentiable
def expand(x, inds):    

    op = tensor_reindex(x)
    return op.expand(inds)


@ad.differentiable     
def flatten(x, ind=None):    

    op = tensor_reindex(x)
    return op.flatten(ind)


@ad.differentiable
def diag(x, ind=None):

    op = tensor_reindex(x)
    return op.diag(ind)




