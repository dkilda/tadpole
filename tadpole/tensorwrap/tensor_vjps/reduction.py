#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




###############################################################################
###                                                                         ###
###  VJP's of tensor reduction functions                                    ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def vjp_unreduce(g, out, x, inds=None):

    return tn.unreduce_like(out, x, inds=inds)(g)


ad.makevjp(tn.amax, vjp_unreduce)
ad.makevjp(tn.amin, vjp_unreduce)




# --- Shape methods --------------------------------------------------------- #

def vjp_sumover(g, out, x, inds=None, **opts):

    return tn.expand_like(g, x, inds=inds)


ad.makevjp(tn.sumover, vjp_sumover)





# --- Linear algebra methods ------------------------------------------------ #

def vjp_norm(g, out, x, inds=None, order=None, **opts):

    pass


ad.makevjp(tn.norm, vjp_norm)














































