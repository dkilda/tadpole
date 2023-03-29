#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tests.tensor.fakes as fake
import tests.tensor.data  as data
import tests.array.data   as ardata


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




def train_dat(enginetype, backend, indnames, shapes, **opts):

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

    arrays = util.Sequence(*arrays)
    inds   = util.Sequence(*inds)
    train  = TrainTensorData(arrays, inds)

    return TrainData(
       train, 
       tensors, arrays, datas, inds, 
       indnames, shapes, ws[0].backend, opts
    )   














"""
# --- Engine data --------------------------------------------------------- #

EngineData = collections.namedtuple("EngineData", [
                "engine",  "train", 
                "tensors", "arrays", "datas", "inds",
                "opts",
             ])



'''
def engine_dat(enginetype, backend, indnames, shapes, **opts):

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

    arrays = util.Sequence(*arrays)
    inds   = util.Sequence(*inds)

    train  = TrainTensorData(arrays, inds)
    engine = enginetype(train)

    return EngineData(
       engine, train, 
       tensors, arrays, datas, inds, 
       opts
    )   
'''



def engine_dat(which):

    def wrap(backend, indnames, shapes, **opts):

        enginetype = {
           "contract":    EngineContract, 
           "dot":         EngineDot,
           "decomp":      EngineDecomp,
           "reduction":   EngineReduce,
           "reindexing":  EngineReindex,
           "interaction": EngineInteraction,
           "unary":       EngineElemwiseUnary,
           "binary":      EngineElemwiseBinary,
           "ternary":     EngineElemwiseTernary,
        }[which]

        opdatfun = {
           "contract":    contract_dat, 
           "decomp":      decomp_dat,
           "reduction":   reduction_dat,
           "reindexing":  reindexing_dat,
           "interaction": interaction_dat,
           "unary":       elemwise_unary_dat,
           "binary":      elemwise_binary_dat,
           "ternary":     elemwise_ternary_dat,
        }[which]

        train    = 
        engine   = enginetype()
        operator = opdatfun(backend, indnames, shapes, **opts)

        return EngineData()
    
    return wrap




# --- Operator data --------------------------------------------------------- #

OperatorData = collections.namedtuple("OperatorData", [
                  "operator", 
                  "tensors", "arrays", "datas",  
                  "inds", "shapes", "backend", "opts", "info"
               ])




def contract_dat(datafun):

    def wrap(backend, indnames, shapes, **opts): 

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

        operator = tn.contraction.TensorContract(arrays, inds, product) # TODO

        return OperatorData(
                  operator, 
                  tensors, arrays, datas,  
                  inds, shapes, ws[0].backend, opts, {"product": product}
               )        
       
    return wrap




def decomp_dat(datafun):

    def wrap(backend, indnames, shape, **opts): 

        w = data.tensor_dat(data.randn)(
               backend, indnames, shape, **opts
            )

        operator = tn.decomp.TensorDecomp(w.array, partition)  # TODO

        return OperatorData(
                  operator, 
                  w.tensor, w.array, w.data,  
                  w.inds, w.shape, w.backend, opts, {"partition": partition}
               )        

    return wrap





def reduction_dat(datafun):

    def wrap(backend, indnames, shape, **opts): 

        w = data.tensor_dat(data.randn)(
               backend, indnames, shape, **opts
            )

        operator = tn.reduction.TensorReduce(w.array, w.inds) 

        return OperatorData(
                  operator, 
                  w.tensor, w.array, w.data,  
                  w.inds, w.shape, w.backend, opts, []
               )        

    return wrap



def reindexing_dat(datafun):

    def wrap(backend, indnames, shape, **opts): 

        w = data.tensor_dat(data.randn)(
               backend, indnames, shape, **opts
            )

        operator = tn.reduction.TensorReindex(w.array, w.inds) 

        return OperatorData(
                  operator, 
                  w.tensor, w.array, w.data,  
                  w.inds, w.shape, w.backend, opts, {}
               )        

    return wrap




def elemwise_dat(n, datafun):

    def wrap(backend, indnames, shape, **opts): 

        ws = []
        for i in range(n):

            w = data.tensor_dat(data.randn)(
                   backend, indnames, shape, seed=i+1, **opts
                )
            ws.append(w)

        tensors = [w.tensor for w in ws]
        arrays  = [w.array  for w in ws]
        datas   = [w.data   for w in ws]

        operator = {
           1: tn.elemwise_unary.TensorElemwiseUnary,
           2: tn.elemwise_binary.TensorElemwiseBinary,
           3: tn.elemwise_ternary.TensorElemwiseTernary,
        }[n](*arrays, inds)

        return OperatorData(
                  operator, 
                  tensors, arrays, datas,  
                  ws[0].inds, ws[0].shape, ws[0].backend, opts, {}
               )        

    return wrap




def interaction_dat(datafun):

    def wrap(backend, indnames, shapes, **opts): 

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

        operator = tn.interaction.TensorInteraction(arrays, inds)

        return OperatorData(
                  operator, 
                  tensors, arrays, datas,  
                  inds, shapes, ws[0].backend, opts, {}
               )        

    return wrap

"""




