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

@ad.differentiable
def sparsegrad(x, pos, source):

    print("SPARSEGRAD: ", x, pos, source)

    # sparsex = tn.space(source).sparsegrad(pos, x)
    # print("SPARSEGRAD: ", pos, vals, sparsex._pos, sparsex._vals, sparsex.todense()._data._data)

    return tn.space(source).sparsegrad([pos], [x.item()])


ad.makevjp(tn.getitem,    
              lambda g, out, x, pos: sparsegrad(g, pos, x)
)

 
ad.makevjp(sparsegrad, 
              lambda g, out, x, pos, source: g[pos] # tn.match(g[pos[0]], x)
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




