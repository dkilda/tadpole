#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.linalg.decomp     as lad
import tadpole.linalg.properties as lap
import tadpole.linalg.solvers    as las
import tadpole.linalg.transform  as lat

from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Helpers                                                                ### 
###                                                                         ###
###############################################################################


# --- Index alignment logic (by left/right indices) ------------------------- #

def aligned(fun):

    def wrap(x, *args, linds=None, rinds=None, **kwargs):

        inds = Indices(*tn.union_inds(x))

        if linds is not None:

           linds = Indices(*inds.map(*linds))
           rinds = Indices(*inds.remove(*linds)) 

           return fun(x, *args, linds=linds, rinds=rinds, **kwargs)

        if rinds is not None:

           rinds = Indices(*inds.map(*rinds))
           linds = Indices(*inds.remove(*rinds)) 

           return fun(x, *args, linds=linds, rinds=rinds, **kwargs) 

        raise ValueError(
           f"align: must provide at least one of linds and rinds, "
           f"but linds = {linds}, rinds = {rinds}"
        )

    return wrap 




###############################################################################
###                                                                         ###
###  Tensor linalg decompositions                                           ###
###                                                                         ###
###############################################################################


# --- Generic decomposition operator ---------------------------------------- #

class DecompOp:

   def __init__(self, fun, linds, rinds):

       self._fun   = fun
       self._linds = linds
       self._rinds = rinds


   def reshape(self, x):

       x = tn.fuse(x, {self._linds: "l", self._rinds: "r"})

       return tn.transpose(x, "l", "r")


   def apply(self, x, *args, **kwargs):

       return self._fun(x, *args, **kwargs)


   def unreshape(self, L, R):

       L = tn.split(L, {"l": self._linds})  
       R = tn.split(R, {"r": self._rinds})

       return L, R




# --- Explicit-rank decomposition functor ----------------------------------- #

class DecompExplicit:

   def __init__(self, op, *args, **kwargs):

       if not isinstance(op, DecompOp):
          op = DecompOp(op, *args, **kwargs)

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x               = self._op.reshape(x)
       U, S, VH, error = self._op.apply(x, *args, **kwargs)
       U, VH           = self._op.unreshape(U, VH)

       return U, S, VH, error




# --- Hidden-rank decomposition functor ------------------------------------- #

class DecompHidden:

   def __init__(self, op, *args, **kwargs):

       if not isinstance(op, DecompOp):
          op = DecompOp(op, *args, **kwargs)

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x    = self._op.reshape(x)
       L, R = self._op.apply(x, *args, **kwargs)
       L, R = self._op.unreshape(L, R)

       return L, R




# --- Explicit-rank decompositions ------------------------------------------ #

@aligned
def svd(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(lad.svd, linds, rinds)
    return decomp(x, *args, **kwargs)


@aligned
def eig(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(lad.eig, linds, rinds) 
    return decomp(x, *args, **kwargs)


@aligned
def eigh(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(lad.eigh, linds, rinds) 
    return decomp(x, *args, **kwargs)




# --- Hidden-rank decompositions -------------------------------------------- #

@aligned
def qr(x, linds, rinds, *args, **kwargs):

    decomp = DecompHidden(lad.qr, linds, rinds) 
    return decomp(x, *args, **kwargs)


@aligned
def lq(x, linds, rinds, *args, **kwargs):

    decomp = DecompHidden(lad.lq, linds, rinds) 
    return decomp(x, *args, **kwargs)




###############################################################################
###                                                                         ###
###  Tensor linalg properties                                               ###
###                                                                         ###
###############################################################################


# --- Linalg unary operator ------------------------------------------------- #

class LinalgUnaryOp:

   def __init__(self, fun, linds, rinds):

       self._fun   = fun
       self._linds = linds
       self._rinds = rinds


   def reshape(self, x):

       x = tn.fuse(x, {self._linds: "l", self._rinds: "r"})
       return tn.transpose(x, "l", "r")


   def apply(self, x, *args, **kwargs):

       return self._fun(x, *args, **kwargs)


   def unreshape(self, x, y):

       out = tn.split(x, {"l": self._linds, "r": self._rinds})
       return tn.transpose_like(out, y)




# --- Linalg property functor ----------------------------------------------- #

class LinalgProperty:

   def __init__(self, op, *args, **kwargs):

       if not isinstance(op, LinalgUnaryOp):
          op = LinalgUnaryOp(op, *args, **kwargs) 

       self._op = op


   def __call__(self, x, *args, **kwargs):
 
       out = self._op.reshape(x)
       out = self._op.apply(out, *args, **kwargs)

       return out




# --- Linalg matrix functor ------------------------------------------------- #

class LinalgMatrix:

   def __init__(self, op, *args, **kwargs):

       if not isinstance(op, LinalgUnaryOp):
          op = LinalgUnaryOp(op, *args, **kwargs) 

       self._op = op


   def __call__(self, x, *args, **kwargs):
 
       out = self._op.reshape(x)
       out = self._op.apply(out, *args, **kwargs)
       out = self._op.unreshape(out, x)

       return out




# --- Linear algebra properties --------------------------------------------- #

@aligned
def norm(x, *args, linds, rinds, **kwargs):

    unary = LinalgProperty(lap.norm, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def trace(x, *args, linds, rinds, **kwargs):

    unary = LinalgProperty(lap.trace, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def det(x, *args, linds, rinds, **kwargs):

    unary = LinalgProperty(lap.det, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def inv(x, *args, linds, rinds, **kwargs):

    unary = LinalgMatrix(lap.inv, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def tril(x, *args, linds, rinds, **kwargs):

    unary = LinalgMatrix(lap.tril, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def triu(x, *args, linds, rinds, **kwargs):

    unary = LinalgMatrix(lap.triu, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def diag(x, *args, linds, rinds, **kwargs):

    unary = LinalgProperty(lap.diag, linds, rinds)
    return unary(x, *args, **kwargs) 




###############################################################################
###                                                                         ###
###  Tensor linalg solvers                                                  ###
###                                                                         ###
###############################################################################


# --- Linalg solver functor ------------------------------------------------- #

class LinalgSolver:

   def __init__(self, fun, linds, rinds):

       self._fun   = fun
       self._linds = linds
       self._rinds = rinds


   def __call__(self, a, b, *args, **kwargs):

       indsI = self._linds
       indsJ = self._rinds
       indsK = tuple(tn.complement_inds(b, a))
 
       i = IndexGen("i", tid.sizeof(*indsI))
       j = IndexGen("j", tid.sizeof(*indsJ))
       k = IndexGen("k", tid.sizeof(*indsK))

       a = tn.transpose(tn.fuse(a, {indsI: i, indsJ: j}), i, j)
       b = tn.transpose(tn.fuse(b, {indsI: i, indsK: k}), i, k)

       x = self._fun(a, b, *args, **kwargs) 
       x = tn.split(x, {j: indsJ, k: indsK}) 
       x = tn.transpose(x, *indsJ, *indsK)   

       return x




# --- Linear algebra solvers ------------------------------------------------ #

@aligned
def solve(a, b, linds, rinds):

    solver = LinalgSolver(las.solve, linds, rinds)
    return solver(a, b)   


@aligned
def trisolve(a, b, linds, rinds, *args, **kwargs):   

    solver = LinalgSolver(las.trisolve, linds, rinds)
    return solver(a, b, *args, **kwargs) 




