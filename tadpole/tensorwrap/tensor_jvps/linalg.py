#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn


from tadpole.tensorwrap.tensor_vjps.linalg import (
   fmatrix,
   tri,
)




###############################################################################
###                                                                         ###
###  Decompositions                                                         ###
###                                                                         ###
###############################################################################


# --- SVD ------------------------------------------------------------------- #

def jvp_svd(g, out, x):

    """
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://arxiv.org/pdf/1909.02659.pdf

    """





# --- Eigendecomposition (general) ------------------------------------------ #

def jvp_eig(g, out, x): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """

    v, s = out
    f    = fmatrix(s)

    dv = v @ (f * (tn.inv(v) @ g @ v))
    ds = tn.space(s).eye() * (tn.inv(v) @ g @ v)

    return (dv, ds)

    


# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def jvp_eigh(g, out, x): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """




# --- QR decomposition ------------------------------------------------------ #

def jvp_qr(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """





# --- LQ decomposition ------------------------------------------------------ #

def jvp_lq(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """




# --- Record decomp JVPs to JVP map ----------------------------------------- # 

ad.makejvp(tn.svd,  jvp_svd)
ad.makejvp(tn.eig,  jvp_eig)
ad.makejvp(tn.eigh, jvp_eigh)
ad.makejvp(tn.qr,   jvp_qr)
ad.makejvp(tn.lq,   jvp_lq)




###############################################################################
###                                                                         ###
###  Standard matrix properties and transformations                         ###
###                                                                         ###
###############################################################################


# --- Norm ------------------------------------------------------------------ #

def jvp_norm(g, out, x, order=None):

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm

    """

    if order in (None, 'fro'):

       return tn.sum(x * g) / out


    if order == 'nuc':

       U, S, VH = tn.svd(x)

       return tn.sum(g * (U @ VH))


    if isinstance(order, int):

       if not (x.ldim == 1 or x.rdim == 1):
          raise ValueError(
             f"jvp_norm: an integer norm order {order} is only valid for " 
             f"vectors, but x has dimensions ({x.ldim}, {x.rdim})."
          )

       return tn.sum(g * x * tn.abs(x)**(order-2)) / out**(order-1) 


    raise ValueError(
       f"jvp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc', or an integer vector norm order."
    )




# --- Inverse --------------------------------------------------------------- #

def jvp_inv(g, out, x):

    return - out @ g @ out




# --- Determinant ----------------------------------------------------------- #

def jvp_det(g, out, x):

    return out * tn.trace(tn.inv(x) * g)  




# --- Trace ----------------------------------------------------------------- #

def jvp_trace(g, out, x):

    return tn.trace(g)




# --- Stack matrices -------------------------------------------------------- #

def jvp_stack(g, adx, out, xs, ind):

    def grad(idx):

        if idx == adx:
           return g

        return tn.space(xs[idx]).zeros()

    gs = tuple(grad(idx) for idx in range(len(xs)))
      
    return tn.stack(gs, ind)




# --- Record standard linalg JVPs ------------------------------------------- #

ad.makejvp(tn.norm,  jvp_norm)
ad.makejvp(tn.inv,   jvp_inv)
ad.makejvp(tn.det,   jvp_det)
ad.makejvp(tn.trace, jvp_trace)
ad.makejvp(tn.diag,  "linear")

ad.makejvp(tn.tril,  lambda g, out, x, which=0: tn.tril(g, which=which))
ad.makejvp(tn.triu,  lambda g, out, x, which=0: tn.triu(g, which=which))

ad.makejvp_combo(tn.stack, jvp_stack)
ad.makejvp_combo(tn.dot,   "linear")




###############################################################################
###                                                                         ###
###  Linear algebra solvers                                                 ###
###                                                                         ###
###############################################################################


# --- Solve the equation ax = b --------------------------------------------- #

def jvpA_solve(g, out, a, b):

    return - tn.solve(a, g @ out)


def jvpB_solve(g, out, a, b):

    return tn.solve(a, g)




# --- Solve the equation ax = b, assuming a is a triangular matrix ---------- #

def jvpA_trisolve(g, out, a, b, which="upper"):

    return - tri(which)(tn.trisolve(a, g @ out, which))


def jvpB_trisolve(g, out, a, b, which="upper"):

    return tn.trisolve(a, g, which)




# --- Record linalg solver JVPs --------------------------------------------- #

ad.makejvp(tn.solve,    jvpA_solve,    jvpB_solve)
ad.makejvp(tn.trisolve, jvpA_trisolve, jvpB_trisolve)




