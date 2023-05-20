#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.array     as ar
import tadpole.container as tc
import tadpole.tensor    as tn
import tadpole.index     as tid

import tadpole.array.backends as backends

import tests.tensor.fakes as fake
import tests.tensor.data  as data

from tadpole.container import (
   NullGrad,
   SparseGrad,
   ContainerGen,
   ContainerSpace,
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




# --- Null grad data ------------------------------------------------------ #

NullContainerData = collections.namedtuple("NullContainerData", [
                         "container", "dense", "tensorspaces", "space",
                         "tensors", "arrays", "datas",  
                         "inds", "shapes", "backend", "opts",
                      ])




def null_container_dat(datafun):

    def wrap(backend, indnames, shapes, **opts):

        w = data.ntensor_dat(datafun)(
               backend, indnames, shapes, **opts
            )

        tspaces = [t.space() for t in w.tensors]

        space = ContainerSpace(tspaces)
        dense = ContainerGen(
                   [tspaces[i].zeros() for i in range(len(tspaces))]
                )

        x = NullGrad(space)

        return NullContainerData(
           x, dense, tspaces, space, 
           w.tensors, w.arrays, w.datas, 
           w.inds, w.shapes, w.backend, w.opts
        )

    return wrap




# --- Sparse grad data ------------------------------------------------------ #

SparseContainerData = collections.namedtuple("SparseContainerData", [
                         "container", "dense", "pos", "vals", 
                         "tensorspaces", "space",
                         "tensors", "arrays", "datas",  
                         "inds", "shapes", "backend", "opts",
                      ])




def sparse_container_dat(datafun):

    def wrap(backend, indnames, shapes, pos, **opts):

        w = data.ntensor_dat(datafun)(
               backend, indnames, shapes, **opts
            )

        tensors = [w.tensors[p] for p in pos]
        tspaces = [t.space()    for t in w.tensors]

        space = ContainerSpace(tspaces)
        dense = ContainerGen(
                   [w.tensors[i] if  i in pos else tspaces[i].zeros() 
                                 for i in range(len(tspaces))]
                )
 
        x = SparseGrad(space, pos, tensors)

        return SparseContainerData(
           x, dense, pos, tensors, tspaces, space,
           w.tensors, w.arrays, w.datas, 
           w.inds, w.shapes, w.backend, w.opts
        )

    return wrap




# --- General container data ------------------------------------------------ #

ContainerData = collections.namedtuple("ContainerData", [
                   "container", "tensorspaces", "space",
                   "tensors", "arrays", "datas",  
                   "inds", "shapes", "backend", "opts",
                ])




def container_dat(datafun):

    def wrap(backend, indnames, shapes, **opts):

        w = data.ntensor_dat(datafun)(
               backend, indnames, shapes, **opts
            )

        tspaces = [t.space() for t in w.tensors]
        space   = ContainerSpace(tspaces)
        x       = ContainerGen(w.tensors)

        return ContainerData(
           x, tspaces, space, 
           w.tensors, w.arrays, w.datas, 
           w.inds, w.shapes, w.backend, w.opts
        )

    return wrap




