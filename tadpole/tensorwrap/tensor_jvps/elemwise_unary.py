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


from tadpole.tensorwrap.tensor_vjps.elemwise_unary import (
   sparsegrad,
)




###############################################################################
###                                                                         ###
###  Unary elementwise functions                                            ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def jvp_clip(g, out, x, minval, maxval):

    return g * tn.logical_and(
                  tn.notequal(out, minval), 
                  tn.notequal(out, maxval)
               )


ad.makejvp(tn.clip,   jvp_clip)
ad.makejvp(tn.flip,   lambda g, out, x, inds=None: tn.flip(g, inds=inds))
ad.makejvp(tn.cumsum, "linear")




# --- Element access -------------------------------------------------------- #

ad.makejvp(tn.getitem, "linear")    
ad.makejvp(sparsegrad, "linear")




# --- Standard math --------------------------------------------------------- #

ad.makejvp(tn.neg,      "linear")
ad.makejvp(tn.conj,     lambda g, out, x: tn.conj(g))
ad.makejvp(tn.real,     lambda g, out, x: tn.real(g))
ad.makejvp(tn.imag,     lambda g, out, x: tn.astype_like(-1j * g, out))
ad.makejvp(tn.absolute, lambda g, out, x: tn.real(g * tn.conj(x)) / out)
ad.makejvp(tn.sqrt,     lambda g, out, x: g * 0.5 * x**(-0.5))
ad.makejvp(tn.log,      lambda g, out, x: g / x)
ad.makejvp(tn.exp,      lambda g, out, x: g * out)

ad.makejvp(tn.sin,    lambda g, out, x:  g * tn.cos(x))
ad.makejvp(tn.cos,    lambda g, out, x: -g * tn.sin(x))
ad.makejvp(tn.tan,    lambda g, out, x:  g / (tn.cos(x)**2))

ad.makejvp(tn.arcsin, lambda g, out, x:  g / tn.sqrt(1 - x**2))
ad.makejvp(tn.arccos, lambda g, out, x: -g / tn.sqrt(1 - x**2))
ad.makejvp(tn.arctan, lambda g, out, x:  g / (1 + x**2))

ad.makejvp(tn.sinh,    lambda g, out, x:  g * tn.cosh(x))
ad.makejvp(tn.cosh,    lambda g, out, x:  g * tn.sinh(x))
ad.makejvp(tn.tanh,    lambda g, out, x:  g / (tn.cosh(x)**2))

ad.makejvp(tn.arcsinh, lambda g, out, x:  g / tn.sqrt(x**2 + 1))
ad.makejvp(tn.arccosh, lambda g, out, x:  g / tn.sqrt(x**2 - 1))
ad.makejvp(tn.arctanh, lambda g, out, x:  g / (1 - x**2))




