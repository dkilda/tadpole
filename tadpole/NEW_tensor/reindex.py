#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar


from tadpole.tensor.train import (
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)




###############################################################################
###                                                                         ###
###  Tensor reindexing operations                                           ###
###                                                                         ###
###############################################################################


# --- Reindexing engine: creates TensorReindex ------------------------------ #

class EngineReindex(Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def attach(self, data, inds):

       if self._train.size() == 1:
          raise TooManyArgsError(self, 1)

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       data, = self._train.data()
       inds, = self._train.inds()

       return TensorReindex(data, inds)




# --- Factory: creates TensorReindex ---------------------------------------- #

def tensor_reindex(x):

    engine = x.pluginto(EngineReindex())
    return engine.operator()




# --- TensorReindex operator ------------------------------------------------ #

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


   # --- Main methods --- #

   def reindex(self, indmap):

       output_inds = list(self._inds)

       for i, ind in enumerate(self._inds):
           try:
               output_inds[i] = indmap[ind]
           except KeyError:
               pass

       return self._new(data, output_inds) 


   def transpose(self, *output_inds):

       output_inds = self._map(*output_inds)

       assert set(self._inds) == set(output_inds),
          f"{type(self).__name__}.transpose: "
          f"the destination indices {output_inds} are not "
          f"compatible with the source indices {self._inds}."

       data = ar.transpose(data, self._inds.axes(*output_inds))
       return self._new(data, output_inds)


   def fuse(self, fusemap):

       inds = self._inds 

       for inp, out in fusemap.items():
           
           inp = self._map(*inp)

           if not isinstance(out, Index):
              out = Index(out, sizeof(*inp))

           assert sizeof(*inp) == sizeof(out), (
               f"{type(self).__name__}.fuse: "
               f"sizes of input indices {inp} and output index {out} must "
               f"match, but the input size {sizeof(*inp)} != "
               f"the output size {sizeof(out)}."
           )

           inds = inds.remove(*inp).add(out)

       data = ar.reshape(data, inds.shape)
       return self._new(data, inds)


   def split(self, splitmap):

       inds = self._inds

       for inp, out in splitmap.items():                   
       
           inp, = self._map(inp)

           assert sizeof(*inp) == sizeof(out), (
              f"{type(self).__name__}.split: "
              f"sizes of input index {inp} and output indices {out} must "
              f"match, but the input size {sizeof(inp)} != "
              f"the output size {sizeof(*out)}."
           )

           axis, = inds.axes(inp)
           inds  = inds.remove(inp).add(*out, axis=axis)

       data = ar.reshape(data, inds.shape)
       return self._new(data, inds)


   def squeeze(self):

       singletons = (ind for ind in self._inds if len(ind) == 1)

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
 



###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorReindex methods            ###
###                                                                         ###
###############################################################################


# --- Main methods ---------------------------------------------------------- #

@ad.differentiable
def reindex(x, indmap):

    op = tensor_reindex(x)
    return op.reindex(indmap)


@ad.differentiable
def transpose(x, *output_inds):

    op = tensor_reindex(x)
    return op.transpose(*output_inds)


@ad.differentiable
def fuse(x, fusemap):

    op = tensor_reindex(x)
    return op.fuse(fusemap)


@ad.differentiable
def split(x, splitmap):

    op = tensor_reindex(x)
    return op.split(splitmap)


@ad.differentiable
def squeeze(x):

    op = tensor_reindex(x)
    return op.squeeze()


@ad.differentiable
def unsqueeze(x, inds):

    op = tensor_reindex(x)
    return op.unsqueeze(inds)


@ad.differentiable
def expand(x, inds):    

    op = tensor_reindex(x)
    return op.expand(inds)




