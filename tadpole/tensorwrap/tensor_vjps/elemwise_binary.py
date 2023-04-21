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
###  Binary elementwise functions                                           ###
###                                                                         ###
###############################################################################


# --- Standard math --------------------------------------------------------- #

ad.makevjp(tn.add, lambda g, out, x, y: tn.match(g, x), 
                   lambda g, out, x, y: tn.match(g, y)
)


ad.makevjp(tn.sub, lambda g, out, x, y: tn.match( g, x), 
                   lambda g, out, x, y: tn.match(-g, y),
)


ad.makevjp(tn.mul, lambda g, out, x, y: tn.match(y * g, x), 
                   lambda g, out, x, y: tn.match(x * g, y)
)


ad.makevjp(tn.div, lambda g, out, x, y: tn.match( g / y,        x),   
                   lambda g, out, x, y: tn.match(-g * x / y**2, y)
)


ad.makevjp(tn.mod, lambda g, out, x, y: tn.match( g,                   x),   
                   lambda g, out, x, y: tn.match(-g * tn.floor(x / y), y)
)


def vjpA_power(g, out, x, y):

    g1 = g * y * (x ** tn.where(y, y-1, 1.))
    return tn.match(g1, x)


def vjpB_power(g, out, x, y):

    g1 = g * out * tn.log(tn.where(x, x, 1.))
    return tn.match(g1, y)


ad.makevjp(tn.power, vjpA_power, vjpB_power)




# --- Gradient accumulation ------------------------------------------------- #

ad.makevjp(tn.addgrads, "identity", "identity")




