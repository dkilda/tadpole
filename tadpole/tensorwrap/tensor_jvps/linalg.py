#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util          as util
import tadpole.autodiff      as ad
import tadpole.tensor        as tn
import tadpole.linalg.matrix as la

from tadpole.tensorwrap.tensor_vjps.linalg import (
   eye,
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
    https://arxiv.org/pdf/1909.02659.pdf
    https://j-towns.github.io/papers/svd-derivative.pdf

    Eq. 16-18

    """
    lind, rind = tn.union_inds(x)
    u, s, v    = out[0], out[1], out[2].H   

    f = fmatrix(s**2)

    g1 = u.H("sl") @ dx("lr")   @ v("rz")
    g2 = v.H("sr") @ dx.H("rl") @ u("lz") 

    ds = 0.5 * tn.space(s).eye() * (g1 + g2)
    du = u @ (f * (g1  * s  + s.T * g2))
    dv = v @ (f * (s.T * g1 + g2  * s))


    if x.ldim < x.rdim:

       vvH = v @ v.H
       dv  = dv + (tn.space(vvH).eye() - vvH) @ dx.H @ (u / s)


    if x.ldim > x.rdim:

       uuH = u @ u.H
       du  = du + (tn.space(uuH).eye() - uuH) @ dx @ (v / s) 


    return du, ds, dv.H




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

    v, s = out
    f    = fmatrix(s)

    dv = v @ (f * (v.H @ g @ v))
    ds = tn.space(s).eye() * (v.H @ g @ v)

    return (dv, ds)




# --- QR decomposition ------------------------------------------------------ #

def jvp_qr(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    p.3, p.7, p.17 (Variations)

    """

    def trisolve(r, a):
    
        return tn.trisolve(r.H, a.H, "lower").H


    def trisym(m):

        space = tn.space(m)
        E     = 2 * space.tril() + space.eye()

        return 0.5 * (m + m.H) * E.H


    def kernel(q, r, dx):

        c = trisolve(r, q.H @ dx) 

        dr = trisym(c) @ r
        dq = trisolve(r, dx - q @ dr)

        return dq, dr


    q, r = out

    if x.ldim >= x.rdim:
       return kernel(q, r, g)

    dx1, dx2 = g[:, : x.ldim], g[:, x.ldim :]
    x1,  x2  = x[:, : x.ldim], x[:, x.ldim :]
    r1,  r2  = r[:, : x.ldim], r[:, x.ldim :]

    dq, dr1 = kernel(q, r1, dx1)
    dr2     = q.H @ (dx2 - dq @ r2) 

    dr = tn.stack((dr1, dr2), "right")

    return dq, dr




# --- LQ decomposition ------------------------------------------------------ #

def jvp_lq(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    p. 16

    """

    def trisolve(l, a):
    
        return tn.trisolve(l, a, "lower")


    def trisym(m):

        space = tn.space(m)
        E     = 2 * space.tril() + space.eye()

        return 0.5 * (m + m.H) * E


    def kernel(l, q, dx):

        c = trisolve(l, dx @ q.H) 

        dl = l @ trisym(c)
        dq = trisolve(l, dx - dl @ q)

        return dl, dq


    l, q = out

    if x.ldim <= x.rdim:
       return kernel(l, q, g)

    dx1, dx2 = g[: x.rdim, :], g[: x.rdim, :]
    x1,  x2  = x[: x.rdim, :], x[: x.rdim, :]
    l1,  l2  = l[: x.rdim, :], l[: x.rdim, :]

    dl1, dq = kernel(l1, q, dx1)
    dl2     = (dx2 - l2 @ dq) @ q.H 

    dl = tn.stack((dl1, dl2), "left")

    return dl, dq




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


    raise ValueError(
       f"jvp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc'."
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




