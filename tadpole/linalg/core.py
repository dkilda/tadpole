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

           return fun(x, *args, linds, rinds, **kwargs) 

        if rinds is not None:

           rinds = Indices(*inds.map(*rinds))
           linds = Indices(*inds.remove(*rinds)) 

           return fun(x, *args, linds, rinds, **kwargs) 

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

   def __init__(self, op):

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x               = self._op.reshape(x)
       U, S, VH, error = self._op.apply(x, *args, **kwargs)
       U, VH           = self._op.unreshape(U, VH)

       return U, S, VH, error




# --- Hidden-rank decomposition functor ------------------------------------- #

class DecompHidden:

   def __init__(self, op):

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x    = self._op.reshape(x)
       L, R = self._op.apply(x, *args, **kwargs)
       L, R = self._op.unreshape(L, R)

       return L, R




# --- Explicit-rank decompositions ------------------------------------------ #

@aligned
def svd(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(DecompOp(lad.svd, linds, rinds))

    return decomp(x, *args, **kwargs)


@aligned
def eig(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(DecompOp(lad.eig, linds, rinds)) 

    return decomp(x, *args, **kwargs)


@aligned
def eigh(x, linds, rinds, *args, **kwargs):

    decomp = DecompExplicit(DecompOp(lad.eigh, linds, rinds)) 

    return decomp(x, *args, **kwargs)




# --- Hidden-rank decompositions -------------------------------------------- #

@aligned
def qr(x, linds, rinds, *args, **kwargs):

    decomp = DecompHidden(DecompOp(lad.qr, linds, rinds)) 

    return decomp(x, *args, **kwargs)


@aligned
def lq(x, linds, rinds, *args, **kwargs):

    decomp = DecompHidden(DecompOp(lad.lq, linds, rinds)) 

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
       return tn.transpose_like(x, y)




# --- Linalg unary functor -------------------------------------------------- #

class LinalgUnary:

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
def norm(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.norm, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def trace(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.trace, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def det(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.det, linds, rinds)
    return unary(x, *args, **kwargs)  


@aligned
def inv(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.inv, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def tril(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.tril, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def triu(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.triu, linds, rinds)
    return unary(x, *args, **kwargs) 


@aligned
def diag(x, linds, rinds, *args, **kwargs):

    unary = LinalgUnary(lap.diag, linds, rinds)
    return unary(x, *args, **kwargs) 




###############################################################################
###                                                                         ###
###  Tensor linalg solvers                                                  ###
###                                                                         ###
###############################################################################


# --- Linalg solver functor ------------------------------------------------- #

class LinalgSolver:

   def __init__(self, fun, indsI, indsJ, indsK):

       self._fun   = fun
       self._indsI = indsI
       self._indsJ = indsJ
       self._indsK = indsK


   def __call__(self, a, b, *args, **kwargs):
 
       i = IndexGen("i", tid.sizeof(*self._indsI))
       j = IndexGen("j", tid.sizeof(*self._indsJ))
       k = IndexGen("k", tid.sizeof(*self._indsK))

       a = tn.transpose(tn.fuse(a, {self._indsI: i, self._indsJ: j}), i, j)
       b = tn.transpose(tn.fuse(b, {self._indsI: i, self._indsK: k}), j, k)

       x = self._fun(a, b, *args, **kwargs) 
       x = tn.split(tn.transpose(x, j, k), {self._indsJ: j, self._indsK: k})
    
       return x




# --- Linear algebra solvers ------------------------------------------------ #

def solve(a, b, indsI, indsJ, indsK):

    solver = LinalgSolver(las.solve, indsI, indsJ, indsK)
    return solver(a, b)   


def trisolve(a, b, indsI, indsJ, indsK, *args, **kwargs):   

    solver = LinalgSolver(las.trisolve, indsI, indsJ, indsK)
    return solver(a, b, *args, **kwargs) 




