#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn

import tadpole.tensor.contraction as tnc

from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  VJP's of tensor contraction                                            ###
###                                                                         ###
###############################################################################


# --- Contraction ----------------------------------------------------------- #

def vjp_contract(g, adx, out, *xs, **opts):

    this   = xs[adx]
    others = xs[:adx] + xs[adx+1:]   

    inds    = Indices(*tn.complement_inds(this, *others, g))
    product = Indices(*tn.union_inds(this)) ^ inds

    print("\nVJP_CONTRACT: ", g.shape, [other_.shape for other_ in others]) 
    
    result = tn.contract(g, *others, product=product) 

    return tn.match(tn.expand(result, inds), this)

 


# --- Record VJPs ----------------------------------------------------------- #

ad.makevjp_combo(tn.contract, vjp_contract)




