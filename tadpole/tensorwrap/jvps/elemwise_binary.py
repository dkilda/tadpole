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
###  Tensor binary elementwise functions                                    ###
###                                                                         ###
###############################################################################


# --- Standard math --------------------------------------------------------- #

ad.makejvp(tn.add, lambda g, out, x, y: tn.match(g, out), 
                   lambda g, out, x, y: tn.match(g, out)
)


ad.makejvp(tn.sub, lambda g, out, x, y: tn.match( g, out), 
                   lambda g, out, x, y: tn.match(-g, out)
)


ad.makejvp(tn.mul, lambda g, out, x, y: tn.match(y * g, out), 
                   lambda g, out, x, y: tn.match(x * g, out)
)


ad.makejvp(tn.div, lambda g, out, x, y: tn.match( g / y,        out),   
                   lambda g, out, x, y: tn.match(-g * x / y**2, out)
)


ad.makejvp(tn.mod, lambda g, out, x, y: tn.match( g,                   out),   
                   lambda g, out, x, y: tn.match(-g * tn.floor(x / y), out)
)


def jvpA_power(g, out, x, y):

    g1 = g * y * (x ** tn.where(y, y-1, 1.))
    return tn.match(g1, x)


def jvpB_power(g, out, x, y):

    g1 = g * out * tn.log(tn.where(x, x, 1.))
    return tn.match(g1, y)


ad.makejvp(tn.power, jvpA_power, jvpB_power)




# --- Gradient accumulation ------------------------------------------------- #

ad.makejvp(tn.addgrads, "identity", "identity") 




