#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn

from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Unary elementwise functions                                            ###
###                                                                         ###
###############################################################################


# --- Data type methods ----------------------------------------------------- #

ad.makevjp(tn.astype, lambda g, out, x, dtype: tn.astype(g, x.dtype))




# --- Value methods --------------------------------------------------------- #

def vjp_clip(g, out, x, minval, maxval):

    return g * tn.logical_and(
                  tn.notequal(out, minval), 
                  tn.notequal(out, maxval)
               )


ad.makevjp(tn.clip, vjp_clip)
ad.makevjp(tn.flip, lambda g, out, x, inds=None: tn.flip(g, inds=inds))


def vjp_cumsum(g, out, x, ind=None, **opts):

    if not ind:
       return tn.flip(tn.cumsum(tn.flip(g), **opts))

    g1 = tn.flip(g, inds=(ind,))
    g1 = tn.cumsum(g1, ind=ind, **opts) 
    g1 = tn.flip(g1, inds=(ind,))

    return g1


ad.makevjp(tn.cumsum, vjp_cumsum)




# --- Element access -------------------------------------------------------- #

ad.makevjp(tn.getitem,    
              lambda g, out, x, pos: tn.ungetitem(g, pos, tn.space(x)) 
)

 
ad.makevjp(tn.ungetitem, 
              lambda g, out, x, pos, space: g[pos] 
) 




# --- Standard math --------------------------------------------------------- #

ad.makevjp(tn.neg,      lambda g, out, x: -g)
ad.makevjp(tn.conj,     lambda g, out, x: tn.conj(g))
ad.makevjp(tn.real,     lambda g, out, x: tn.astype_like(g,       x))
ad.makevjp(tn.imag,     lambda g, out, x: tn.astype_like(-1j * g, x))
ad.makevjp(tn.absolute, lambda g, out, x: g * tn.conj(x) / out)
ad.makevjp(tn.sqrt,     lambda g, out, x: g * 0.5 * x**(-0.5))
ad.makevjp(tn.log,      lambda g, out, x: g / x)
ad.makevjp(tn.exp,      lambda g, out, x: g * out)

ad.makevjp(tn.sin,    lambda g, out, x:  g * tn.cos(x))
ad.makevjp(tn.cos,    lambda g, out, x: -g * tn.sin(x))
ad.makevjp(tn.tan,    lambda g, out, x:  g / (tn.cos(x)**2))

ad.makevjp(tn.arcsin, lambda g, out, x:  g / tn.sqrt(1 - x**2))
ad.makevjp(tn.arccos, lambda g, out, x: -g / tn.sqrt(1 - x**2))
ad.makevjp(tn.arctan, lambda g, out, x:  g / (1 + x**2))

ad.makevjp(tn.sinh,    lambda g, out, x:  g * tn.cosh(x))
ad.makevjp(tn.cosh,    lambda g, out, x:  g * tn.sinh(x))
ad.makevjp(tn.tanh,    lambda g, out, x:  g / (tn.cosh(x)**2))

ad.makevjp(tn.arcsinh, lambda g, out, x:  g / tn.sqrt(x**2 + 1))
ad.makevjp(tn.arccosh, lambda g, out, x:  g / tn.sqrt(x**2 - 1))
ad.makevjp(tn.arctanh, lambda g, out, x:  g / (1 - x**2))




