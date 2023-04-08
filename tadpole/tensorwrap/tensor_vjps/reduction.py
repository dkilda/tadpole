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
###  VJP's of tensor reduction functions                                    ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def vjp_unreduce(g, out, x, inds=None):

    return tn.unreduce_like(out, x, inds=inds)(g)


# TODO: create tensorwrap.util module for vjp/jvp supporting functions
# -- move unreduce_like there, rename it to vjp_reduce 
# -- also create jvp_reduce
# (these are analogous to grad_chooser/fw_grad_chooser)

ad.makevjp(tn.amax, vjp_unreduce)
ad.makevjp(tn.amin, vjp_unreduce)




# --- Shape methods --------------------------------------------------------- #

def vjp_sumover(g, out, x, inds=None, **opts):

    return tn.expand_like(g, x, inds=inds)


ad.makevjp(tn.sumover, vjp_sumover)





# --- Linear algebra methods ------------------------------------------------ #

def vjp_norm(g, out, x, inds=None, order=None, **opts):

    pass

# TODO: create tensor.linalg module, move norm there.
# Also move there all the other linalg ops like trace, inv, det, etc.
# We may end up merging this with decomp, cuz they share a lot in common!  


ad.makevjp(tn.norm, vjp_norm)














































