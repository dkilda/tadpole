#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn

import tadpole.linalg.unwrapped as la

from tadpole.index import (
   Index,
   IndexGen, 
   IndexLit,
   Indices,
)




###############################################################################
###                                                                         ###
###  VJP's of tensor decompositions                                         ###
###                                                                         ###
###############################################################################


# --- Helpers: identity matrix ---------------------------------------------- #

def eye(x, inds=None): 

    if inds is None:
       return tn.space(x).eye()

    sind  = IndexLit(inds[0], x.shape[0])
    sind1 = IndexLit(inds[1], x.shape[-1])

    return tn.space(x).eye(sind, sind1)




# --- Helpers: F-matrix ----------------------------------------------------- #

def fmatrix(s): 

    seye = eye(s,"ij")

    return 1. / (s("1j") - s("i1") + seye) - seye 




# --- SVD ------------------------------------------------------------------- #

def vjp_svd(g, out, x, sind=None, trunc=None):

    """
    https://arxiv.org/pdf/1909.02659.pdf

    Eq. 1, 2, 36 (take complex conjugate of both sides)

    """

    du, ds, dv = g[0],   g[1],   g[2].H
    u,  s,  v  = out[0], out[1], out[2].H

    f = fmatrix(s**2)("ij")

    uTdu = u.T("im") @ du("mj")
    vTdv = v.T("im") @ dv("mj")

    grad = eye(s,"ij") * ds("i1") 
    grad = grad + f * s("1j") * (uTdu("ij") - uTdu.H("ij"))  
    grad = grad + f * s("i1") * (vTdv("ij") - vTdv.H("ij"))
 

    if tn.iscomplex(u):
       grad = grad + 1j * tn.imag(eye(uTdu) * uTdu) / s("1j")


    grad = u("li").C @ grad("ij") @ v.T("jr") 


    if x.shape[0] < x.shape[1]: 

       vvH  = v("bm") @ v.H("mr")
       grad = grad \
            + ((u("la") / s("1a")) @ dv.T("ab") @ (eye(vvH) - vvH)).C

       return grad(*tn.union_inds(x))


    if x.shape[0] > x.shape[1]:

       uuH  = u("bm") @ u.H("ml")
       grad = grad \
            + ((v("ra") / s("1a")) @ du.T("ab") @ (eye(uuH) - uuH)).T

       return grad(*tn.union_inds(x))


    return grad(*tn.union_inds(x))




# --- Eigendecomposition (general) ------------------------------------------ #

def vjp_eig(g, out, x, sind=None):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.77 (take complex conjugate of both sides)

    """

    dv, ds = g
    v,  s  = out

    f    = fmatrix(s)("ij")
    vTdv = v.T("im") @ dv("mj")

    grad1 = f * vTdv  
    grad2 = f * ((v.T("im") @ v.C("mn")) @ (tn.real(vTdv) * eye(vTdv))("nj"))

    grad = ds("1j") * eye(s,"ij") + grad1 - grad2
    grad = la.inv(v.T)("li") @ grad("ij") @ v.T("jr")
    
    if not tn.iscomplex(x):
       grad = tn.real(grad)

    return grad(*tn.union_inds(x))




# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def vjp_eigh(g, out, x, sind=None):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.71 (take complex conjugate of both sides)

    Comments:

    * numpy and pytorch use UPLO="L" by default

    * tensorflow always uses UPLO="L"
      https://www.tensorflow.org/api_docs/python/tf/linalg/eigh

    """

    dv, ds = g
    v,  s  = out

    grad = eye(s,"ij") * ds("i1")

    if not tn.allclose(dv, tn.space(dv).zeros()): 
       grad = grad + fmatrix(s)("ij") * (v.T("im") @ dv("mj"))

    grad = v("li").C @ grad @ v.T("jr") 

    tl   = la.tril(tn.space(grad).ones(), k=-1)
    grad = tn.real(grad) * eye(grad) \
         + (grad("lr") + grad.H("lr")) * tl("lr") 
       
    return grad(*tn.union_inds(x))


"""
    grad = (v.C("lm") * ds("1m")) @ v.T("mr")

    if not tn.allclose(dv, tn.space(dv).zeros()): 

       f    = fmatrix(s)("ij")
       grad = grad + v.C("li") @ (f * (v.T("im") @ dv("mj"))) @ v.T("jr")

    tl   = la.tril(tn.space(grad).ones(), k=-1)
    grad = tn.real(grad) * eye(grad) \
         + (grad("lr") + grad.H("lr")) * tl("lr") 
       
    return grad(*tn.union_inds(x))
"""



# --- QR decomposition ------------------------------------------------------ #

def vjp_qr(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(r, a):

        return la.trisolve(r, a.H, which="upper").H


    def hcopyltu(m):

        return m.H + la.tril(m) * (m - m.H) 


    def kernel(q, dq, r, dr):

        m = r("im") @ dr.H("mj") - dq.H("im") @ q("mj")

        return trisolve(r("jr"), dq("lj") + q("li") @ hcopyltu(m)("ij"))


    dq, dr = g
    q,  r  = out

    if x.shape[0] >= x.shape[1]:
       return kernel(q, dq, r, dr)(*tn.union_inds(x))

    x1,  x2  =  x[:, : x.shape[0]],  x[:, x.shape[0] :]
    r1,  r2  =  r[:, : x.shape[0]],  r[:, x.shape[0] :]
    dr1, dr2 = dr[:, : x.shape[0]], dr[:, x.shape[0] :]

    dx1 = kernel(q, dq("li") + x2("lr") @ dr2.H("ri"), r1, dr1)
    dx2 = q("li") @ dr2("ir")

    return la.concat(
       (dx1("ia"), dx2("ib")), tuple(tn.union_inds(x)), which="right"
    )




# --- LQ decomposition ------------------------------------------------------ #

def vjp_lq(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(l, a):

        return la.trisolve(l.H, a, which="upper")


    def hcopyltu(m):

        return m.H + la.tril(m) * (m - m.H) 


    def kernel(l, dl, q, dq):

        m = l.H("im") @ dl("mj") - dq("im") @ q.H("mj")

        return trisolve(l("li"), dq("ir") + hcopyltu(m)("ij") @ q("jr"))


    dl, dq = g
    l,  q  = out

    if x.shape[0] <= x.shape[1]:
       return kernel(l, dl, q, dq)(*tn.union_inds(x))

    x1,  x2  =  x[: x.shape[1], :],  x[x.shape[1] :, :]
    l1,  l2  =  l[: x.shape[1], :],  l[x.shape[1] :, :]
    dl1, dl2 = dl[: x.shape[1], :], dl[x.shape[1] :, :]

    dx1 = kernel(l1, dl1, q, dq("ir") + dl2.H("il") @ x2("lr"))
    dx2 = dl2("li") @ q("ir")

    return la.concat(
       (dx1("ai"), dx2("bi")), tuple(tn.union_inds(x)), which="left"
    )




# --- Record decomp VJPs ---------------------------------------------------- # 

ad.makevjp(la.svd,  vjp_svd)
ad.makevjp(la.eig,  vjp_eig)
ad.makevjp(la.eigh, vjp_eigh)
ad.makevjp(la.qr,   vjp_qr)
ad.makevjp(la.lq,   vjp_lq)




###############################################################################
###                                                                         ###
###  VJP's of standard matrix properties and transformations                ###
###                                                                         ###
###############################################################################


# --- Norm ------------------------------------------------------------------ #

def vjp_norm(g, out, x, order=None, **opts):

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm

    """

    if order in (None, 'fro'):

       return (g / out) * x.C 


    if order == 'nuc':

       U, S, VH, error = la.svd(x)

       return g * (U.C @ VH.C)


    raise ValueError(
       f"vjp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc'."
    )




# --- Trace ----------------------------------------------------------------- #

def vjp_trace(g, out, x, **opts):

    return tn.space(x).eye() * g




# --- Determinant ----------------------------------------------------------- #

def vjp_det(g, out, x):

    return g * out * la.inv(x).T




# --- Inverse --------------------------------------------------------------- #

def vjp_inv(g, out, x):

    grad = -out.T("ij") @ g("jk") @ out.T("kl")

    return grad(*tn.union_inds(x))




# --- Diagonal -------------------------------------------------------------- #

def vjp_diag(g, out, x, inds, **opts): 

    xinds = list(tn.union_inds(x))

    i = min(xinds, key=len)
    j = i.retagged("j")
    k = xinds[1 - xinds.index(i)] 

    grad = (g(i,"1") * tn.space(x).eye(i,j)) @ tn.space(x).eye(j,k) 

    return tn.transpose_like(grad, x)




# --- Concatenate matrices -------------------------------------------------- #

def vjp_concat(g, adx, out, *xs, inds, which=None, **opts): 

    axis = {
            None:    0, 
            "left":  0, 
            "right": 1,
           }[which]
                           
    start = sum([x.shape[axis] for x in xs[:adx]])
    size  = xs[adx].shape[axis] 

    adx_slice       = [slice(None), slice(None)]
    adx_slice[axis] = slice(start, start + size)

    return g[tuple(adx_slice)](*tn.union_inds(xs[adx])) 




# --- Record standard linalg VJPs ------------------------------------------- #

ad.makevjp(la.norm,  vjp_norm)
ad.makevjp(la.trace, vjp_trace)
ad.makevjp(la.det,   vjp_det)
ad.makevjp(la.inv,   vjp_inv)
ad.makevjp(la.diag,  vjp_diag)

ad.makevjp(la.tril, lambda g, out, x, **opts: la.tril(g, **opts))
ad.makevjp(la.triu, lambda g, out, x, **opts: la.triu(g, **opts))

ad.makevjp_combo(la.concat, vjp_concat)




###############################################################################
###                                                                         ###
###  VJP's of linear algebra solvers                                        ###
###                                                                         ###
###############################################################################


# --- Solve the equation ax = b --------------------------------------------- #

def vjpA_solve(g, out, a, b):

    return -la.solve(a.T, g) @ out.T


def vjpB_solve(g, out, a, b):

    return la.solve(a.T, g)




# --- Solve the equation ax = b, assuming a is a triangular matrix ---------- #

def tri(which):

    if which is None:
       which = "upper"

    return {
            "lower": la.tril, 
            "upper": la.triu,
           }[which]


def opposite(which):

    if which is None:
       which = "upper"

    return {
            "lower": "upper", 
            "upper": "lower",
           }[which]


def vjpA_trisolve(g, out, a, b, which=None):

    return -tri(which)(la.trisolve(a.T, g, which=opposite(which)) @ out.T)
    

def vjpB_trisolve(g, out, a, b, which=None):

    return la.trisolve(a.T, g, which=opposite(which))




# --- Record linalg solver VJPs --------------------------------------------- #

ad.makevjp(la.solve,    vjpA_solve,    vjpB_solve)
ad.makevjp(la.trisolve, vjpA_trisolve, vjpB_trisolve)
    

    

