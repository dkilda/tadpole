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

def jvp_reduce(g, out, x, inds=None):

    out  = tn.expand_like(out, x, inds)
    mask = tn.isequal(x, out)
     
    return tn.sum(g * mask, inds) / tn.sum(mask, inds) 


ad.makejvp(tn.amax, jvp_reduce)
ad.makejvp(tn.amin, jvp_reduce)




# --- Shape methods --------------------------------------------------------- #

ad.makejvp(tn.sumover, "linear")




