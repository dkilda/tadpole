#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util          as util
import tadpole.autodiff      as ad
import tadpole.tensor        as tn
import tadpole.linalg.matrix as la

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

    sind  = IndexLit(inds[0], s.shape[0])
    sind1 = IndexLit(inds[1], s.shape[1])

    return tn.space(x).eye(sind, sind1)




# --- Helpers: F-matrix ----------------------------------------------------- #

def fmatrix(s): 

    seye = eye(s,"sz")

    return 1. / (s("1z") - s("s1") + seye) - seye 




# --- SVD ------------------------------------------------------------------- #

def vjp_svd(g, out, x):

    """
    https://arxiv.org/pdf/1909.02659.pdf

    Eq. 1, 2, 36 (take complex conjugate of both sides)

    """
    lind, rind = tn.union_inds(x)

    du, ds, dv = g[0],   g[1],   g[2].H
    u,  s,  v  = out[0], out[1], out[2].H

    f = fmatrix(s**2)

    uTdu = u.T("sl") @ du("lz")
    vTdv = v.T("sr") @ dv("rz")

    g1 = eye(s,"sz") * ds("s1") 
    g1 = g1 + f("sz") * s("1z") * (uTdu - uTdu.H)("sz")  
    g1 = g1 + f("sz") * s("s1") * (vTdv - vTdv.H)("sz")


    if tn.iscomplex(u):
       g1 = g1 + 1j * tn.imag(eye(uTdu) * uTdu)("sz") / s("s1")


    g1 = u("ls").C @ g1("sz") @ v.T("zr") 


    if len(lind) < len(rind): 

       vvH = v("Rs") @ v.H("sr")

       g1 = g1 + (u("ls") / s("1s")) @ dv.T("sR") @ (eye(vvH) - vvH)("Rr")
       return g1(lind, rind)


    if len(lind) > len(rind):

       uuH = u("ls") @ u.H("sL")

       g1 = g1 + (eye(uuH) - uuH)("lL") @ du("Ls") @ (v.T("sr") / s("s1")) 
       return g1(lind, rind)


    return g1(lind, rind)

    


# --- Eigendecomposition (general) ------------------------------------------ #

def vjp_eig(g, out, x):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.77 (take complex conjugate of both sides)

    """
    lind, rind = tn.union_inds(x)

    dv, ds = g
    v,  s  = out

    f     = fmatrix(s)
    vdiag = tn.real(v.T("sr") @ dv("rz")) * eye(s, "sz")

    g1 = f("sz") * (v.T("sr") @  dv("rz"))
    g2 = f("sz") * (v.T("sr") @ v.C("rz")) @ vdiag("sz")

    g12 = ds("1z") * eye(s, "sz") + g1("sz") - g2("sz")
    g12 = la.inv(v.T)("ls") @ g12("sz") @ v.T("zr")
    
    if not tn.iscomplex(x):
       return tn.real(g12)(lind, rind)

    return g12(lind, rind)




# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def vjp_eigh(g, out, x):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.71 (take complex conjugate of both sides)

    Comments:

    * numpy and pytorch use UPLO="L" by default

    * tensorflow always uses UPLO="L"
      https://www.tensorflow.org/api_docs/python/tf/linalg/eigh

    """
    lind, rind = tn.union_inds(x)

    dv, ds = g
    v,  s  = out

    grad = (v.C("ls") * ds("1s")) @ v.T("sr")

    if not tn.allclose(dv, tn.space(dv).zeros()): 

       f    = fmatrix(s)
       grad = (
          grad + 
          v.C("ls") @ (f("sz") * (v.T("sn") @ dv.C("nz"))) @ v.T("zr")
       )

    tl = la.tril(tn.space(x).ones()("lr"), k=-1)

    grad = tn.real(grad) * eye(grad, "lr") + (grad + grad.H) * tl        
    return grad(lind, rind)



# --- QR decomposition ------------------------------------------------------ #

def vjp_qr(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(r, a):

        return la.trisolve(r, a.H, which="upper").H


    def hcopyltu(m):

        return m.H("ji") + tril(m("ij")) * (m("ij") - m.H("ji")) 


    def kernel(q, dq, r, dr):

        m = r("sr") @ dr.H("rz") - dq.H("sl") @ q("lz")

        return trisolve(r("sr"), dq("ls") + q("lz") @ hcopyltu(m)("zs"))


    lind, rind = tn.union_inds(x)

    dq, dr = g
    q,  r  = out

    if len(lind) >= len(rind):
       return kernel(q, dq, r, dr)(lind, rind)

    x1,  x2  =  x[:, : len(lind)],  x[:, len(lind) :]
    r1,  r2  =  r[:, : len(lind)],  r[:, len(lind) :]
    dr1, dr2 = dr[:, : len(lind)], dr[:, len(lind) :]

    dx1 = kernel(q, dq("ls") + x2("lr") @ dr2.H("rs"), r1, dr1)
    dx2 = q("ls") @ dr2("sr")

    return la.concat((dx1("lr"), dx2("lR")), (lind, rind), which="right")




# --- LQ decomposition ------------------------------------------------------ #

def vjp_lq(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(l, a):

        return la.trisolve(l.H, a, which="upper")


    def hcopyltu(m):

        return m.H("ji") + la.tril(m("ij")) * (m("ij") - m.H("ji")) 


    def kernel(l, dl, q, dq):

        m = l.H("sl") @ dl("lz") - dq("sr") @ q.H("rz")

        return trisolve(l("ls"), dq("sr") + q("sR") @ hcopyltu(m)("Rr"))


    lind, rind = tn.union_inds(x)

    dl, dq = g
    l,  q  = out

    if len(lind) <= len(rind):
       return kernel(l, dl, q, dq)(lind, rind)

    x1,  x2  =  x[: len(rind), :],  x[len(rind) :, :]
    l1,  l2  =  l[: len(rind), :],  l[len(rind) :, :]
    dl1, dl2 = dl[: len(rind), :], dl[len(rind) :, :]

    dx1 = kernel(l1, dl1, q, dq("sr") + dl2.H("sl") @ x2("lr"))
    dx2 = dl2("ls") @ q("sr")

    return la.concat((dx1("lr"), dx2("lR")), (lind, rind), which="left")




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

def vjp_norm(g, out, x, order=None):

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm

    """

    if order in (None, 'fro'):

       return (g / out) * x


    if order == 'nuc':

       U, S, VH = la.svd(x)

       return g * (U @ VH)


    if isinstance(order, int):

       if not (x.shape[0] == 1 or x.shape[1] == 1):
          raise ValueError(
             f"vjp_norm: an integer norm order {order} is only valid for " 
             f"vectors, but x has dimensions ({x.shape[0]}, {x.shape[1]})."
          )

       return (g / out**(order-1)) * x * tn.abs(x)**(order-2)


    raise ValueError(
       f"vjp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc', or an integer vector norm order."
    )




# --- Inverse --------------------------------------------------------------- #

def vjp_inv(g, out, x):

    lind, rind = tn.union_inds(x)

    grad = - out.T("ij") @ g("jk") @ out.T("kl")
    return grad(lind, rind)



# --- Determinant ----------------------------------------------------------- #

def vjp_det(g, out, x):

    return g * out * la.inv(x).T




# --- Trace ----------------------------------------------------------------- #

def vjp_trace(g, out, x):

    return tn.space(x).eye() * g




# --- Diagonal -------------------------------------------------------------- #

def vjp_diag(g, out, x, inds, **opts): 

    return la.diag(g, tuple(tn.union_inds(x)))

   


# --- Stack matrices -------------------------------------------------------- #

def vjp_concat(g, adx, out, xs, inds, which=None, **opts): 
                              
    axis  = {None: 0, "left": 0, "right": 1}[which]   
                                        
    start = sum([x("left","right").shape[axis] for x in xs[:adx]])
    size  = xs[adx].shape[axis] 

    adx_slice = slice(start, start + size)

    return g[adx_slice](*tn.union_inds(xs[adx])) 




# --- Record standard linalg VJPs ------------------------------------------- #

ad.makevjp(la.norm,  vjp_norm)
ad.makevjp(la.inv,   vjp_inv)
ad.makevjp(la.det,   vjp_det)
ad.makevjp(la.trace, vjp_trace)
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

    return {
            "lower": la.tril, 
            "upper": la.triu,
           }[which]


def vjpA_trisolve(g, out, a, b, which=None):

    return - tri(which)(la.trisolve(a.H, g, which=which) @ out.T)
    

def vjpB_trisolve(g, out, a, b, which=None):

    return la.trisolve(a.H, g, which)




# --- Record linalg solver VJPs --------------------------------------------- #

ad.makevjp(la.solve,    vjpA_solve,    vjpB_solve)
ad.makevjp(la.trisolve, vjpA_trisolve, vjpB_trisolve)
    

    

