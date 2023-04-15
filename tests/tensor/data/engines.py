#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.tensor.engine as tne 

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




# --- Train data ---------------------------------------------------------- #

TrainData = collections.namedtuple("TrainData", [
               "train", 
               "tensors", "arrays", "datas", "inds",
               "indnames", "shapes", "backend", "opts",
            ])




def train_dat(backend, indnames, shapes, **opts):

    ws = []
    for i in range(len(shapes)):

        w = data.tensor_dat(data.randn)(
               backend, indnames[i], shapes[i], seed=i+1, **opts
            )
        ws.append(w)

    tensors = [w.tensor for w in ws]
    arrays  = [w.array  for w in ws]
    datas   = [w.data   for w in ws]
    inds    = [w.inds   for w in ws]

    arrays = util.Sequence(arrays)
    inds   = util.Sequence(inds)
    train  = tne.TrainTensorData(arrays, inds)

    return TrainData(
       train, 
       tensors, arrays, datas, inds, 
       indnames, shapes, ws[0].backend, opts
    )   




