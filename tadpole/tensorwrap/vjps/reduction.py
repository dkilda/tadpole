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
###  Tensor reduction functions                                             ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def vjp_reduce(g, out, x, inds=None): 

    g    = tn.expand_like(g,   x, inds)  
    out  = tn.expand_like(out, x, inds)  
    mask = tn.isequal(x, out)

    return g * mask / tn.expand_like(tn.sumover(mask, inds), x, inds) 


ad.makevjp(tn.amax, vjp_reduce)
ad.makevjp(tn.amin, vjp_reduce)




# --- Shape methods --------------------------------------------------------- #

def vjp_sumover(g, out, x, inds=None, **opts):

    return tn.expand_like(g, x, inds=inds)


ad.makevjp(tn.sumover, vjp_sumover)




