#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.container as tc
import tadpole.tensor    as tn

import tadpole.linalg.unwrapped as la

from tadpole.tensorwrap.vjps.linalg import (
   eye,
   fmatrix,
   tri,
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

    dx = g
    f  = fmatrix(s**2)("ij")

    grad1 = u.H("il") @ dx("lr")   @ v("rj")
    grad2 = v.H("ir") @ dx.H("rl") @ u("lj") 

    ds = 0.5 * eye(s,"ij") * (grad1 + grad2)
    du = u("li") @ (f * (grad1 * s("1j") + s("i1") * grad2))
    dv = v("ri") @ (f * (grad1 * s("i1") + s("1j") * grad2))


    if x.shape[0] < x.shape[1]:

       vvH = v("rm") @ v.H("ma")
       dv  = dv("ri") \
           + (eye(vvH) - vvH) @ dx.H("ab") @ (u("bi") / s("1i"))


    if x.shape[0] > x.shape[1]:

       uuH = u("lm") @ u.H("ma")
       du  = du("li") \
           + (eye(uuH) - uuH) @ dx("ab") @ (v("bi") / s("1i")) 


    du = du(*tn.union_inds(u))
    dv = dv(*tn.union_inds(v))
    ds = la.diag(tn.astype_like(ds, s), next(tn.union_inds(s)))

    return tc.container(du, ds, dv.H, tn.NullGrad(tn.space(out[-1]))) 




# --- Eigendecomposition (general) ------------------------------------------ #

def jvp_eig(g, out, x, sind=None): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """

    v, s = out
    grad = la.inv(v)("im") @ g("mn") @ v("nj")

    dv = v("li") @ (fmatrix(s)("ij") * grad("ij"))
    ds = eye(s,"ij") * grad("ij")

    dv = dv(*tn.union_inds(v))
    ds = la.diag(ds, next(tn.union_inds(s)))

    return tc.container(dv, ds)

    


# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def jvp_eigh(g, out, x, sind=None): 

    """
    https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf    

    """

    v, s = out

    grad = (g("mn") + g.H("mn")) / 2
    grad = v.H("im") @ grad @ v("nj")

    dv = v("li") @ (fmatrix(s)("ij") * grad("ij"))
    ds = eye(s,"ij") * grad("ij")

    dv = dv(*tn.union_inds(v))
    ds = la.diag(ds, next(tn.union_inds(s)))

    return tc.container(dv, ds)




# --- QR decomposition ------------------------------------------------------ #

def jvp_qr(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    p.3, p.7, p.17 (Variations)

    """

    def trisolve(r, a):
    
        return la.trisolve(r.H, a.H, which="lower").H


    def trisym(m):

        E = 2 * la.triu(tn.space(m).ones(), k=1) + tn.space(m).eye()

        return (m("ij") + m.H("ij")) * E("ij") / 2


    def kernel(q, r, dx):

        c = q.H("il") @ trisolve(r("jr"), dx("lr")) 

        dr = trisym(c)("ij") @ r("jr")
        dq = trisolve(r("ir"), dx("lr") - q("lj") @ dr("jr"))

        return tc.container(
           dq(*tn.union_inds(q)), 
           dr(*tn.union_inds(r)) 
        )

    q, r = out

    if x.shape[0] >= x.shape[1]:
       return kernel(q, r, g)

    dx1, dx2 = g[:, : x.shape[0]], g[:, x.shape[0] :]
    x1,  x2  = x[:, : x.shape[0]], x[:, x.shape[0] :]
    r1,  r2  = r[:, : x.shape[0]], r[:, x.shape[0] :]

    dq, dr1 = kernel(q, r1, dx1)
    dr2     = q.H("il") @ (dx2("lr") - dq("lm") @ r2("mr")) 

    dr = la.concat(
            dr1("ia"), 
            dr2("ib"), 
            inds=tuple(tn.union_inds(r)), 
            which="right"
         )
    dq = dq(*tn.union_inds(q)) 

    return tc.container(dq, dr)





def OLD_jvp_qr(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    p.3, p.7, p.17 (Variations)

    """

    def trisolve(r, a):
    
        return la.trisolve(r.H, a.H, which="lower").H


    def trisym(m):

        E = 2 * la.tril(m) + eye(m)

        return 0.5 * (m + m.H) * E.H


    def kernel(q, r, dx):

        c  = trisolve(r("jr"), q.H("il") @ dx("lr")) 
        dr = trisym(c)("ij") @ r("jr")
        dq = trisolve(r("ir"), dx("lr") - q("lj") @ dr("jr"))

        return tc.container(dq, dr)


    q, r = out

    if x.shape[0] >= x.shape[1]:
       return kernel(q, r, g)

    dx1, dx2 = g[:, : x.shape[0]], g[:, x.shape[0] :]
    x1,  x2  = x[:, : x.shape[0]], x[:, x.shape[0] :]
    r1,  r2  = r[:, : x.shape[0]], r[:, x.shape[0] :]

    dq, dr1 = kernel(q, r1, dx1)
    dr2     = q.H("il") @ (dx2("lr") - dq("lm") @ r2("mr")) 

    dr = la.concat(
            dr1("ia"), 
            dr2("ib"), 
            inds=tuple(tn.union_inds(r)), 
            which="right"
         )
    dq = dq(*tn.union_inds(q)) 

    return tc.container(dq, dr)




# --- LQ decomposition ------------------------------------------------------ #

def jvp_lq(g, out, x, sind=None):

    """
    https://arxiv.org/abs/2009.10071

    p. 16

    """

    def trisolve(l, a):
    
        return la.trisolve(l, a, which="lower")


    def trisym(m):

        E = 2 * la.tril(m) + eye(m)

        return 0.5 * (m + m.H) * E


    def kernel(l, q, dx):

        c  = trisolve(l("li"), dx("lr") @ q.H("rj")) 
        dl = l("li") @ trisym(c)("ij")
        dq = trisolve(l("lj"), dx("lr") - dl("li") @ q("ir"))

        return tc.container(dl, dq)


    l, q = out

    if x.shape[0] <= x.shape[1]:
       return kernel(l, q, g)

    dx1, dx2 = g[: x.shape[1], :], g[: x.shape[1], :]
    x1,  x2  = x[: x.shape[1], :], x[: x.shape[1], :]
    l1,  l2  = l[: x.shape[1], :], l[: x.shape[1], :]

    dl1, dq = kernel(l1, q, dx1)
    dl2     = (dx2("lr") - l2("lm") @ dq("mr")) @ q.H("ri") 

    dl = la.concat(
            (dl1("ai"), dl2("bi")), tuple(tn.union_inds(l)), which="left"
         )
    dq = dq(*tn.union_inds(q)) 

    return tc.container(dl, dq)




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

       return tn.sumover(x * g) / out


    if order == 'nuc':

       U, S, VH, error = la.svd(x)

       return tn.sumover(g * (U @ VH))


    raise ValueError(
       f"jvp_norm: invalid norm order {order} provided. The order must "
       f"be one of: None, 'fro', 'nuc'."
    )




# --- Determinant ----------------------------------------------------------- #

def jvp_det(g, out, x):

    return out * la.trace(la.inv(x)("im") @ g("mj"))



  
# --- Inverse --------------------------------------------------------------- #

def jvp_inv(g, out, x):

    grad = -out("ij") @ g("jk") @ out("kl")

    return grad(*tn.union_inds(out))




# --- Concatenate matrices -------------------------------------------------- #

def jvp_concat(g, adx, out, *xs, inds, which=None, **opts):

    def grad(idx):

        if idx == adx:
           return g

        return tn.space(xs[idx]).zeros()

    gs = (grad(idx) for idx in range(len(xs)))
      
    return la.concat(*gs, inds=inds, which=which, **opts)




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

    return -la.trisolve(a, tri(which)(g) @ out, which=which) 


def jvpB_trisolve(g, out, a, b, which=None):

    return la.trisolve(a, g, which=which) 




# --- Record linalg solver JVPs --------------------------------------------- #

ad.makejvp(la.solve,    jvpA_solve,    jvpB_solve)
ad.makejvp(la.trisolve, jvpA_trisolve, jvpB_trisolve)




