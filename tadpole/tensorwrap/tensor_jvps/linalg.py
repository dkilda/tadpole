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

from tadpole.tensorwrap import (
   ContainerGen,
)

from tadpole.index import (
   Index,
   IndexGen, 
   IndexLit,
   Indices,
)




###############################################################################
###                                                                         ###
###  Decompositions                                                         ###
###                                                                         ###
###############################################################################


# --- SVD ------------------------------------------------------------------- #

def jvp_svd(g, out, x, sind=None, trunc=None):

    """
    https://arxiv.org/pdf/1909.02659.pdf
    https://j-towns.github.io/papers/svd-derivative.pdf

    Eq. 16-18

    """
    u, s, v = out[0], out[1], out[2].H  
    f       = fmatrix(s**2)

    g1 = u.H("sl") @ dx("lr")   @ v("rz")
    g2 = v.H("sr") @ dx.H("rl") @ u("lz") 

    ds = 0.5 * eye(s,"sz")  * (g1("sz") + g2("sz"))
    du = u("ls") @ (f("sz") * (g1("sz") * s("1z")  + s("s1")  * g2("sz")))
    dv = v("rs") @ (f("sz") * (s("s1")  * g1("sz") + g2("sz") * s("1z")))


    if x.shape[0] < x.shape[1]:

       vvH = v("rs") @ v.H("sR")
       dv  = dv("rs") \
           + (eye(vvH) - vvH)("rR") @ dx.H("RL") @ (u("Ls") / s("1s"))


    if x.shape[0] > x.shape[1]:

       uuH = u("ls") @ u.H("sL")
       du  = du("ls") \
           + (eye(uuH) - uuH)("lL") @ dx("LR") @ (v("Rs") / s("1s")) 

    du = du(*tn.union_inds(u))
    dv = dv(*tn.union_inds(v))
    ds = la.diag(ds, tuple(tn.union_inds(s)))

    return ContainerGen(du, ds, dv.H)




# --- Eigendecomposition (general) ------------------------------------------ #

def jvp_eig(g, out, x, sind=None): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """
    v, s = out
    f    = fmatrix(s)

    dv = v("rs") @ (f("sz") * (la.inv(v)("sm") @ g("mn") @ v("nz")))
    ds = eye(s,"sz") * (la.inv(v)("sm") @ g("mn") @ v("nz"))

    dv = dv(*tn.union_inds(v))
    ds = la.diag(ds, tuple(tn.union_inds(s)))

    return ContainerGen(dv, ds)

    


# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def jvp_eigh(g, out, x, sind=None): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """

    v, s = out
    f    = fmatrix(s)

    dv = v("ls") @ (f("sz") * (v.H("sm") @ g("mn") @ v("nz")))
    ds = eye(s, "sz")       * (v.H("sm") @ g("mn") @ v("nz"))

    dv = dv(*tn.union_inds(v))
    ds = la.diag(ds, tuple(tn.union_inds(s)))

    return ContainerGen(dv, ds)




# --- QR decomposition ------------------------------------------------------ #

def jvp_qr(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    p.3, p.7, p.17 (Variations)

    """

    def trisolve(r, a):
    
        return la.trisolve(r.H, a.H, which="lower").H


    def trisym(m):

        E = 2 * la.tril(m("ij")) + eye(m("ij"))

        return 0.5 * (m("ij") + m.H("ji")) * E.H("ji")


    def kernel(q, r, dx):

        c  = trisolve(r("sr"), q.H("zl") @ dx("lr")) 
        dr = trisym(c)("sz") @ r("zr")
        dq = trisolve(r("sr"), dx("lr") - q("ls") @ dr("sr"))

        return dq, dr


    q, r = out

    if x.shape[0] >= x.shape[1]:
       return kernel(q, r, g)

    dx1, dx2 = g[:, : x.shape[0]], g[:, x.shape[0] :]
    x1,  x2  = x[:, : x.shape[0]], x[:, x.shape[0] :]
    r1,  r2  = r[:, : x.shape[0]], r[:, x.shape[0] :]

    dq, dr1 = kernel(q, r1, dx1)
    dr2     = q.H("sl") @ (dx2("lr") - dq("lz") @ r2("zr")) 

    dr = la.concat(
            (dr1("sr"), dr2("sR")), tuple(tn.union_inds(r)), which="right"
         )

    dq = dq(*tn.union_inds(q)) 
    dr = dr(*tn.union_inds(r)) 

    return ContainerGen(dq, dr)




# --- LQ decomposition ------------------------------------------------------ #

def jvp_lq(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    p. 16

    """

    def trisolve(l, a):
    
        return la.trisolve(l, a, which="lower")


    def trisym(m):

        E = 2 * la.tril(m("ij")) + eye(m("ij"))

        return 0.5 * (m("ij") + m.H("ji")) * E("ij")


    def kernel(l, q, dx):

        c = trisolve(l("ls"), dx("lr") @ q.H("rz")) 

        dl = l("ls") @ trisym(c)("sz")
        dq = trisolve(l("ls"), dx("lr") - dl("lz") @ q("zr"))

        return dl, dq


    l, q = out

    if x.shape[0] <= x.shape[1]:
       return kernel(l, q, g)

    dx1, dx2 = g[: x.shape[1], :], g[: x.shape[1], :]
    x1,  x2  = x[: x.shape[1], :], x[: x.shape[1], :]
    l1,  l2  = l[: x.shape[1], :], l[: x.shape[1], :]

    dl1, dq = kernel(l1, q, dx1)
    dl2     = (dx2("Lr") - l2("Ls") @ dq("sr")) @ q.H("rz") 

    dl = la.concat(
            (dl1("ls"), dl2("Ls")), tuple(tn.union_inds(l)), which="left"
         )

    return dl, dq




# --- Record decomp JVPs to JVP map ----------------------------------------- # 

ad.makejvp(la.svd,  jvp_svd)
ad.makejvp(la.eig,  jvp_eig)
ad.makejvp(la.eigh, jvp_eigh)
ad.makejvp(la.qr,   jvp_qr)
ad.makejvp(la.lq,   jvp_lq)




###############################################################################
###                                                                         ###
###  Standard matrix properties and transformations                         ###
###                                                                         ###
###############################################################################


# --- Norm ------------------------------------------------------------------ #

def jvp_norm(g, out, x, order=None, **opts):

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm

    """

    if order in (None, 'fro'):

       return tn.sum(x * g) / out


    if order == 'nuc':

       U, S, VH = la.svd(x)

       return tn.sum(g * (U @ VH))


    raise ValueError(
       f"jvp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc'."
    )




# --- Determinant ----------------------------------------------------------- #

def jvp_det(g, out, x):

    return out * la.trace(la.inv(x) * g)



  
# --- Inverse --------------------------------------------------------------- #

def jvp_inv(g, out, x):

    grad = -out("ij") @ g("jk") @ out("kl")

    return grad(*tn.union_inds(out))




# --- Concatenate matrices -------------------------------------------------- #

def jvp_concat(g, adx, out, xs, inds, which=None, **opts):

    def grad(idx):

        if idx == adx:
           return g

        return tn.space(xs[idx]).zeros()

    gs = tuple(grad(idx) for idx in range(len(xs)))
      
    return la.concat(gs, inds, which=which, **opts)




# --- Record standard linalg JVPs ------------------------------------------- #

ad.makejvp(la.norm,  jvp_norm)
ad.makejvp(la.trace, "linear")
ad.makejvp(la.det,   jvp_det)
ad.makejvp(la.inv,   jvp_inv)
ad.makejvp(la.diag,  "linear")
ad.makejvp(la.tril,  "linear")
ad.makejvp(la.triu,  "linear")

ad.makejvp_combo(la.concat, jvp_concat)




###############################################################################
###                                                                         ###
###  Linear algebra solvers                                                 ###
###                                                                         ###
###############################################################################


# --- Solve the equation ax = b --------------------------------------------- #

def jvpA_solve(g, out, a, b):

    return -la.solve(a, g @ out)


def jvpB_solve(g, out, a, b):

    return la.solve(a, g)




# --- Solve the equation ax = b, assuming a is a triangular matrix ---------- #

def jvpA_trisolve(g, out, a, b, which=None):

    return -tri(which)(la.trisolve(a, g @ out, which=which))


def jvpB_trisolve(g, out, a, b, which=None):

    return la.trisolve(a, g, which)




# --- Record linalg solver JVPs --------------------------------------------- #

ad.makejvp(la.solve,    jvpA_solve,    jvpB_solve)
ad.makejvp(la.trisolve, jvpA_trisolve, jvpB_trisolve)




