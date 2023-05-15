#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util  as util
import tadpole.index as tid

from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)

import tadpole.tensor.decomp as decomp




# --- Left alignment -------------------------------------------------------- #

class AlignmentLeft(Alignment):

   def __init__(self, partial_inds):

       self._partial_inds = partial_inds


   def _mapto(self, inds):

       return inds.map(*self._partial_inds)


   def linds(self, inds):

       return Indices(*self._mapto(inds))
       

   def rinds(self, inds):

       return Indices(*inds.remove(*self._mapto(inds))) 




# --- Right alignment -------------------------------------------------------- #

class AlignmentRight(Alignment):

   def __init__(self, partial_inds):

       self._partial_inds = partial_inds


   def _mapto(self, inds):

       return inds.map(*self._partial_inds)


   def linds(self, inds):

       return Indices(*inds.remove(*self._mapto(inds))) 
       

   def rinds(self, inds):

       return Indices(*self._mapto(inds))




def alignment(linds=None, rinds=None):

    if linds is not None:
       return AlignmentLeft(linds)

    if rinds is not None:
       return AlignmentRight(rinds)

    raise ValueError(
       f"alignment: must provide at least one of linds and rinds, 
       f"but linds = {linds}, rinds = {rinds}"
    )






class DecompOp:

   def __init__(self, fun, align):

       self._fun   = fun
       self._align = align


   def reshape(x):

       x = tn.fuse(x, {self._align.linds(): "l", self._align.rinds(): "r"})

       return tn.transpose(x, "l", "r")


   def decomp(self, x, *args, **kwargs):

       return self._fun(x, *args, **kwargs)


   def unreshape(self, L, R):

       L = tn.split(L, {"l": self._align.linds()})  
       R = tn.split(R, {"r": self._align.rinds()})

       return L, R



class DecompExplicit:

   def __init__(self, op):

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x               = self._op.reshape(x)
       U, S, VH, error = self._op.decomp(x, *args, **kwargs)
       U, VH           = self._op.unreshape(U, VH)

       return U, S, VH, error




class DecompHidden:

   def __init__(self, op):

       self._op = op


   def __call__(self, x, *args, **kwargs):

       x    = self._op.reshape(x)
       L, R = self._op.decomp(x, *args, **kwargs)
       L, R = self._op.unreshape(L, R)

       return L, R



 



"""
class DecompExplicit(Decomp):

   def __init__(self, fun, align):

       self._fun   = fun
       self._align = align


   def __call__(self, x, *args, **kwargs):

       x = tn.fuse(x, {self._align.linds(): "l", self._align.rinds(): "r"})
       x = tn.transpose(x, "l", "r")

       U, S, VH, error = self._fun(x, *args, **kwargs)

       if 


       U  = tn.split(U,  {"l": self._align.linds()})  
       VH = tn.split(VH, {"r": self._align.rinds()})

       return U, S, VH, error






class DecompHidden(Decomp):

   def __init__(self, fun, align):

       self._fun   = fun
       self._align = align


   def __call__(self, x, *args, **kwargs):

       x = tn.fuse(x, {self._align.linds(): "l", self._align.rinds(): "r"})
       x = tn.transpose(x, "l", "r")

       L, R = self._fun(x, *args, **kwargs)

       L = tn.split(L, {"l": self._align.linds()})  
       R = tn.split(R, {"r": self._align.rinds()})

       return L, R



class DecompOutput:

   def left(self, out):

       return out[0]


   def right(self, out):

       return out[2]


   def assemble(out, left, right):
 
       return left, out[1], right, out[3]




   def output(self, L, R):

       return L
"""



def decompfun(decomptype):

    def _decompfun(fun, linds=None, rinds=None):

        return decomptype(DecompOp(fun, alignment(linds, rinds)))

    return _decompfun



def svd(x, linds=None, rinds=None, *args, **kwargs):

    decomp = decompfun(DecompExplicit)(ladecomp.svd, linds, rinds)

    return decomp(x, *args, **kwargs)


def eig(x, linds=None, rinds=None, *args, **kwargs):

    decomp = decompfun(DecompExplicit)(ladecomp.eig, linds, rinds)

    return decomp(x, *args, **kwargs)


def eigh(x, linds=None, rinds=None, *args, **kwargs):

    decomp = decompfun(DecompExplicit)(ladecomp.eigh, linds, rinds)

    return decomp(x, *args, **kwargs)


def qr(x, linds=None, rinds=None, *args, **kwargs):

    decomp = decompfun(DecompHidden)(ladecomp.qr, linds, rinds)

    return decomp(x, *args, **kwargs)


def lq(x, linds=None, rinds=None, *args, **kwargs):

    decomp = decompfun(DecompHidden)(ladecomp.lq, linds, rinds)

    return decomp(x, *args, **kwargs)
    



"""

def svd(x, linds=None, rinds=None, *args, **kwargs):

    align  = alignment(linds, rinds)
    decomp = DecompExplicit(DecompOp(ladecomp.svd, align))

    return decomp(x, *args, **kwargs)



def eig(x, linds=None, rinds=None, *args, **kwargs):

    align  = alignment(linds, rinds)
    decomp = DecompExplicit(ladecomp.eig, align)

    return decomp(x, *args, **kwargs)



def eigh(x, linds=None, rinds=None, *args, **kwargs):

    align  = alignment(linds, rinds)
    decomp = DecompExplicit(ladecomp.eig, align)

    return decomp(x, *args, **kwargs)



def qr(x, linds=None, rinds=None, *args, **kwargs):

    align  = alignment(linds, rinds)
    decomp = DecompHidden(ladecomp.qr, align)

    return decomp(x, *args, **kwargs)



def lq(x, linds=None, rinds=None, *args, **kwargs):

    align  = alignment(linds, rinds)
    decomp = DecompHidden(ladecomp.lq, align)

    return decomp(x, *args, **kwargs)



"""
       


"""

def svd(x, linds=None, rinds=None, *args, **kwargs):

    align = alignment(linds, rinds)    

    x = tn.fuse(x, {align.linds(): "l", align.rinds(): "r"})
    x = tn.transpose(x, "l", "r")

    U, S, VH, error = decomp.svd(x, *args, **kwargs)

    U  = tn.split(U,  {"l": align.linds()})  
    VH = tn.split(VH, {"r": align.rinds()})

    return U, S, VH, error




def decomp_explicit(fun, linds=None, rinds=None, sind=None, trunc=None):

    



    def wrap(x, linds=None, rinds=None, sind=None, trunc=None): 

        align = alignment(linds, rinds) 

        out = fun(x, )


        return U, S, VH, error


     
    return wrap

"""


















