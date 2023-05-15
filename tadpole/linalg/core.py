#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.linalg.decomp as lad

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
###  Tensor decomposition                                                   ###
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


   def decomp(self, x, *args, **kwargs):

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
       U, S, VH, error = self._op.decomp(x, *args, **kwargs)
       U, VH           = self._op.unreshape(U, VH)

       return U, S, VH, error




# --- Hidden-rank decomposition functor ------------------------------------- #

class DecompHidden:

   def __init__(self, op):

       if not isinstance(op, DecompOp):
          op = DecompOp(op, *args, **kwargs)

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x    = self._op.reshape(x)
       L, R = self._op.decomp(x, *args, **kwargs)
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
    



