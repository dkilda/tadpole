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
                     "indsI", "shapeI", "sizeI", "axesI",
                     "indsJ", "shapeJ", "sizeJ", "axesJA", "axesJB",
                     "indsK", "shapeK", "sizeK", "axesK",  
                  ])




def solver_input_000():

    return SolverInputData(

       indsA  = "mno",     
       shapeA = (2,3,4),

       indsB  = "mop"
       shapeB = (2,4,1)

       indsX  = "np"
       shapeX = (3,1)

       indsI  = "mo" 
       shapeI = (2,4)
       sizeI  = 8
       axesI  = (0,2)

       indsJ  = "n" 
       shapeJ = (3,)
       sizeJ  = 3
       axesJA = (1,)
       axesJB = (0,) 

       indsK  = "p" 
       shapeK = (1,)
       sizeK  = 1
       axesK  = (1,)
    )




def solver_input_001():

    return SolverInputData(

       indsA  = "mno",     
       shapeA = (2,3,4),

       indsB  = "mopq"
       shapeB = (2,4,5,6)

       indsX  = "npq"
       shapeX = (3,5,6)

       indsI  = "mo" 
       shapeI = (2,4)
       sizeI  = 8
       axesI  = (0,2)

       indsJ  = "n" 
       shapeJ = (3,)
       sizeJ  = 3
       axesJA = (1,)
       axesJB = (0,) 

       indsK  = "pq" 
       shapeK = (5,6)
       sizeK  = 30
       axesK  = (1,2)
    )




def solver_input_002():

    return SolverInputData(

       indsA  = "lmno",     
       shapeA = (2,3,4,5),

       indsB  = "mopq"
       shapeB = (2,5,6,7)

       indsX  = "nlpq"
       shapeX = (4,2,6,7)

       indsI  = "mo" 
       shapeI = (3,5)
       sizeI  = 15
       axesI  = (1,3)

       indsJ  = "nl" 
       shapeJ = (4,2)
       sizeJ  = 8
       axesJA = (2,0)
       axesJB = (0,1) 

       indsK  = "pq" 
       shapeK = (6,7)
       sizeK  = 42
       axesK  = (2,3)
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
                "inds",
                "backend", 
             ])




def solver_data(datafun):

    def wrap(method, backend, solver_input, **opts):

        # --- Input --- #

        w = solver_input 
        v = data.indices_dat(w.indsA + w.indsB, w.shapeA + w.shapeB)  
        a = data.array_dat(datafun)(backend, (w.sizeI, w.sizeJ), **opts)  
        b = data.array_dat(datafun)(backend, (w.sizeI, w.sizeK), **opts)   

        # --- Run solver on matrices --- #

        matrixA = a.array
        arrayA  = ar.reshape(matrixA,  (*w.shapeI, *w.shapeJ))  
        arrayA  = ar.transpose(arrayA, v.inds.axes(*w.indsA)) 

        matrixB = b.array
        arrayB  = ar.reshape(matrixB,  (*w.shapeI, *w.shapeK))  
        arrayB  = ar.transpose(arrayB, v.inds.axes(*w.indsB)) 

        matrixX = {
                   "solve":          ar.solve,
                   "trisolve_upper": lambda x, y: ar.trisolve(x, y, which="upper"),
                   "trisolve_lower": lambda x, y: ar.trisolve(x, y, which="lower"),
                  }[method](matrixA, matrixB)

        # --- Create input and output tensors --- #

        arrayX = ar.reshape(matrixX,  (*w.shapeJ, *w.shapeK))
        arrayX = ar.transpose(arrayX, v.inds.axes(*w.indsX))

        indsA = v.inds.map(*w.indsA)
        indsB = v.inds.map(*w.indsB)
        indsX = v.inds.map(*w.indsX)

        tensorA = tn.TensorGen(arrayA, indsA)
        tensorB = tn.TensorGen(arrayB, indsB)
        tensorX = tn.TensorGen(arrayX, indsX)

        return SolverData(
                tensorA,  tensorB,  tensorX,
                arrayA,   arrayB,   arrayX,
                matrixA,  matrixB,  matrixX,
                w.indsA,  w.indsB,  w.indsX,
                w.shapeA, w.shapeB, w.shapeX,
                w.indsI,  w.indsJ,  w.indsK,
                w.shapeI, w.shapeJ, w.shapeK,
                w.sizeI,  w.sizeJ,  w.sizeK,
                v.inds,
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




