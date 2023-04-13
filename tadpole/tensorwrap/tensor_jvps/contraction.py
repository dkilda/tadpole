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
###  JVP's of tensor contraction                                            ###
###                                                                         ###
###############################################################################


# --- Contraction ----------------------------------------------------------- #

def jvp_contract(g, adx, out, *xs, **opts):

    xs      = list(xs)
    xs[adx] = g

    return tn.contract(*xs, **opts)

    
ad.makevjp_combo(tn.contract, jvp_contract)
















































