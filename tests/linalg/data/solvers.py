#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid
import tadpole.linalg   as la

import tadpole.array.backends as backends

import tests.linalg.fakes as fake
import tests.linalg.data  as data


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




# --- Solver input data ----------------------------------------------------- #

SolverInputData = collections.namedtuple("SolverInputData", [
                     "indsA", "shapeA",
                     "indsB", "shapeB",
                     "indsX", "shapeX",
                     "indsI", "shapeI", "sizeI", "axesIA", "axesIB",
                     "indsJ", "shapeJ", "sizeJ", "axesJA", "axesJX",
                     "indsK", "shapeK", "sizeK", "axesKB", "axesKX", 
                  ])




def solver_input_000():

    return SolverInputData(

       indsA  = "mno",     
       shapeA = (2,8,4),

       indsB  = "mop",
       shapeB = (2,4,1),

       indsX  = "np",
       shapeX = (8,1),

       indsI  = "mo", 
       shapeI = (2,4),
       sizeI  = 8,
       axesIA = (0,2),
       axesIB = (0,1),

       indsJ  = "n", 
       shapeJ = (8,),
       sizeJ  = 8,
       axesJA = (1,),
       axesJX = (0,), 

       indsK  = "p", 
       shapeK = (1,),
       sizeK  = 1,
       axesKB = (1,),
       axesKX = (1,),
    )




def solver_input_001():

    return SolverInputData(

       indsA  = "mno",     
       shapeA = (2,8,4),

       indsB  = "mopq",
       shapeB = (2,4,5,6),

       indsX  = "npq",
       shapeX = (8,5,6),

       indsI  = "mo", 
       shapeI = (2,4),
       sizeI  = 8,
       axesIA = (0,2),
       axesIB = (0,1),

       indsJ  = "n", 
       shapeJ = (8,),
       sizeJ  = 8,
       axesJA = (1,),
       axesJX = (0,), 

       indsK  = "pq", 
       shapeK = (5,6),
       sizeK  = 30,
       axesKB = (2,3),
       axesKX = (1,2),
    )




def solver_input_002():

    return SolverInputData(

       indsA  = "lmno",     
       shapeA = (2,3,6,4),

       indsB  = "mopq",
       shapeB = (3,4,5,7),

       indsX  = "lnpq",
       shapeX = (2,6,5,7),

       indsI  = "mo", 
       shapeI = (3,4),
       sizeI  = 12,
       axesIA = (1,3),
       axesIB = (0,1),

       indsJ  = "ln", 
       shapeJ = (2,6),
       sizeJ  = 12,
       axesJA = (0,2),
       axesJX = (0,1), 

       indsK  = "pq", 
       shapeK = (5,7),
       sizeK  = 35,
       axesKB = (2,3),
       axesKX = (2,3),
    )




# --- Solver data ----------------------------------------------------------- #

SolverData = collections.namedtuple("SolverData", [
                "tensorA",  "tensorB",  "tensorX",
                "arrayA",   "arrayB",   "arrayX",
                "matrixA",  "matrixB",  "matrixX",
                "indsA",    "indsB",    "indsX",
                "shapeA",   "shapeB",   "shapeX",
                "indsI",    "indsJ",    "indsK",
                "shapeI",   "shapeJ",   "shapeK",
                "sizeI",    "sizeJ",    "sizeK",
                "backend", 
             ])




def solver_data(datafun):

    def argsort(*xs):

        return tuple(util.argsort(xs))


    def wrap(method, backend, solver_input, **opts):

        # --- Input --- #

        w = solver_input 
        v = data.nindices_dat([w.indsA, w.indsB], [w.shapeA, w.shapeB])  

        indsA = v.inds.map(*w.indsA)
        indsB = v.inds.map(*w.indsB)
        indsX = v.inds.map(*w.indsX)

        a = data.array_dat(datafun)(backend, (w.sizeI, w.sizeJ), **opts)  
        b = data.array_dat(datafun)(backend, (w.sizeI, w.sizeK), **opts)   

        # --- Run solver on matrices --- #

        matrixA = a.array
        arrayA  = ar.reshape(matrixA,  (*w.shapeI, *w.shapeJ)) 
        arrayA  = ar.transpose(arrayA, argsort(*w.axesIA, *w.axesJA)) 

        matrixB = b.array
        arrayB  = ar.reshape(matrixB,  (*w.shapeI, *w.shapeK))  
        arrayB  = ar.transpose(arrayB, argsort(*w.axesIB, *w.axesKB))

        matrixX = {
                   "solve":          ar.solve,
                   "trisolve_upper": lambda x, y: ar.trisolve(x, y, which="upper"),
                   "trisolve_lower": lambda x, y: ar.trisolve(x, y, which="lower"),
                  }[method](matrixA, matrixB)

        # --- Create input and output tensors --- #

        arrayX = ar.reshape(matrixX,  (*w.shapeJ, *w.shapeK))
        arrayX = ar.transpose(arrayX, argsort(*w.axesJX, *w.axesKX))

        tensorA = tn.TensorGen(arrayA, v.inds.map(*w.indsA))
        tensorB = tn.TensorGen(arrayB, v.inds.map(*w.indsB))
        tensorX = tn.TensorGen(arrayX, v.inds.map(*w.indsX))

        return SolverData(
                tensorA,  tensorB,  tensorX,
                arrayA,   arrayB,   arrayX,
                matrixA,  matrixB,  matrixX,
                w.indsA,  w.indsB,  w.indsX,
                w.shapeA, w.shapeB, w.shapeX,
                w.indsI,  w.indsJ,  w.indsK,
                w.shapeI, w.shapeJ, w.shapeK,
                w.sizeI,  w.sizeJ,  w.sizeK,
                a.backend, 
               )

    return wrap




# --- Solve data ------------------------------------------------------------ #

def solve_linalg_dat(solver_input):

    def wrap(datafun, backend, **opts):
        return solver_data(datafun)(
                  "solve", backend, solver_input(), **opts
               )
    return wrap




# --- Triangular solve (upper) data ----------------------------------------- #

def trisolve_upper_linalg_dat(solver_input):

    def wrap(datafun, backend, **opts):
        return solver_data(datafun)(
                  "trisolve_upper", backend, solver_input(), **opts
               )
    return wrap




# --- Triangular solve (lower) data ----------------------------------------- #

def trisolve_lower_linalg_dat(solver_input):

    def wrap(datafun, backend, **opts):
        return solver_data(datafun)(
                  "trisolve_lower", backend, solver_input(), **opts
               )
    return wrap




