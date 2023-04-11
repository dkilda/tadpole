#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




###############################################################################
###                                                                         ###
###  VJP's of tensor decomposition functions                                ###
###                                                                         ###
###############################################################################


# --- Helpers: F-matrix ----------------------------------------------------- #

def fmatrix(s): # Assume S = Matrix(ldim=1, rdim=size) by default 
                # We should overload all elemwise methods for Matrix objects! 
    eye = tn.space(s).eye()

    return 1. / (s - s.T + eye) - eye 




# --- SVD ------------------------------------------------------------------- #

def vjp_svd(g, out, x):

    """
    https://arxiv.org/pdf/1909.02659.pdf

    Eq. 1, 2, 36 (take complex conjugate of both sides)

    """

    du, ds, dv = g[0],   g[1],   g[2].H
    u,  s,  v  = out[0], out[1], out[2].H

    f = fmatrix(s**2)

    uTdu = u.T @ du
    vTdv = v.T @ dv

    g1 = tn.space(s).eye() * ds.T 
    g1 = g1 + f * s   * (uTdu - uTdu.H)  
    g1 = g1 + f * s.T * (vTdv - vTdv.H)

    if tn.iscomplex(u):
       g1 = g1 + 1j * tn.imag(tn.diag(tn.space(uTdu).eye() * uTdu)) / s


    g1 = u.C @ tn.diag(g1) @ v.T


    if x.ldim < x.rdim:

       vvH = v @ v.H

       return g1 + (u / s) @ dv.T @ (tn.space(vvH).eye() - vvH)


    if x.ldim > x.rdim:

       uuH = u @ u.H

       return g1 + (tn.space(uuH).eye() - uuH) @ du @ (v.T / s.T) 


    return g1

    


# --- Eigendecomposition (general) ------------------------------------------ #

def vjp_eig(g, out, x):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.77 (take complex conjugate of both sides)

    """

    dv, ds = g
    v,  s  = out

    f   = fmatrix(s)
    eye = tn.space(v).eye()

    g1 = f * (v.T @ dv)
    g2 = f * ((v.T @ v.C) @ (tn.real(v.T @ dv) * eye))

    g12 = tn.inv(v.T) @ (tn.diag(ds) + g1 - g2) @ v.T
    
    if not tn.iscomplex(x):
       return tn.real(g12)

    return g12




# --- Eigendecomposition (Hermitian) ---------------------------------------- #

def vjp_eigh(g, out, x):

    """
    https://arxiv.org/abs/1701.00392
 
    Eq. 4.71 (take complex conjugate of both sides)

    Comments:

    * numpy and pytorch use UPLO="L" by default

    * tensorflow always uses UPLO="L"
      https://www.tensorflow.org/api_docs/python/tf/linalg/eigh

    But vjp_eigh should not depend on the details of a backend...
    Could add grad_eigh to backend instead! Usage:

    -- REMOVE THIS:

    eye  = tn.space(grad).eye()
    tril = tn.tril(tn.space(x).ones(), -1)

    return tn.real(grad) * eye + (grad + grad.H) * tril  

    -- ADD THIS:

    return tn.grad_eigh(grad) 

    which delegates the work to backend or grad internal Array
    and computes 

    tn.real(grad) * eye + (grad + grad.H) * tril 
   
    internally.    

    """

    dv, ds = g
    v,  s  = out

    grad = v.C @ tn.diag(ds) @ v.T

    if not tn.allclose(dv, tn.space(dv).zeros()): 

       f    = fmatrix(s)
       grad = grad + v.C @ (f * (v.T @ dv.C)) @ v.T

    eye  = tn.space(grad).eye()
    tril = tn.tril(tn.space(x).ones(), -1)

    return tn.real(grad) * eye + (grad + grad.H) * tril        
    



# --- QR decomposition ------------------------------------------------------ #

def vjp_qr(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(r, a):

        return tn.trisolve(r, a.H, "upper").H


    def hcopyltu(m):

        return m.H + tn.space(m).tril() * (m - m.H) 


    def kernel(q, dq, r, dr):

        m = r @ dr.H - dq.H @ q

        return trisolve(r, dq + q @ hcopyltu(m))


    dq, dr = g
    q,  r  = out

    if x.ldim >= x.rdim:
       return kernel(q, dq, r, dr)

    x1,  x2  =  x[:, : x.ldim],  x[:, x.ldim :]
    r1,  r2  =  r[:, : x.ldim],  r[:, x.ldim :]
    dr1, dr2 = dr[:, : x.ldim], dr[:, x.ldim :]

    dx1 = kernel(q, dq + x2 @ dr2.H, r1, dr1)
    dx2 = q @ dr2

    return tn.concat((dx1, dx2), "right")




# --- LQ decomposition ------------------------------------------------------ #

def vjp_lq(g, out, x):

    """
    https://arxiv.org/abs/2009.10071

    """

    def trisolve(l, a):

        return tn.trisolve(l.H, a, "lower")


    def hcopyltu(m):

        return m.H + tn.space(m).tril() * (m - m.H) 


    def kernel(l, dl, q, dq):

        m = l.H @ dl - dq @ q.H

        return trisolve(l, dq + q @ hcopyltu(m))


    dl, dq = g
    l,  q  = out

    if x.ldim <= x.rdim:
       return kernel(l, dl, q, dq)

    x1,  x2  =  x[: x.ldim, :],  x[x.ldim :, :]
    l1,  l2  =  l[: x.ldim, :],  l[x.ldim :, :]
    dl1, dl2 = dl[: x.ldim, :], dl[x.ldim :, :]

    dx1 = kernel(l1, dl1, q, dq + dl2.H @ x2)
    dx2 = dl2 @ q

    return tn.concat((dx1, dx2), "left")




# --- Adding VJP's for every type of decomposition -------------------------- # 

ad.makevjp(tn.svd,  vjp_svd)
ad.makevjp(tn.eig,  vjp_eig)
ad.makevjp(tn.eigh, vjp_eigh)
ad.makevjp(tn.qr,   vjp_qr)
ad.makevjp(tn.lq,   vjp_lq)




###############################################################################
###                                                                         ###
###  VJP's of standard matrix properties and transformations                ###
###                                                                         ###
###############################################################################




###############################################################################
###                                                                         ###
###  VJP's of linear algebra solvers                                        ###
###                                                                         ###
###############################################################################


































