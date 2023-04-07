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
###  VJP's of tensor contraction functions                                  ###
###                                                                         ###
###############################################################################


# --- Contraction ----------------------------------------------------------- #

def vjp_contract(g, adx, out, *xs, **opts):

    this   = xs[adx]
    others = xs[:adx] + xs[adx+1:]    

    inds  = tn.Indices(*tn.complement_inds(this, *others, g))
    space = tn.space(this).reshape(inds)
    ones  = space.ones()
    
    result = tn.contract(g, ones, *others, product=tn.union_inds(this)) 

    return tn.match(result, this)


ad.makevjp_combo(tn.contract, vjp_contract)




# --- Dot ------------------------------------------------------------------- #

def vjpA_dot(g, out, x, y):

    return vjp_contract(g, 0, out, x, y)


def vjpB_dot(g, out, x, y):

    return vjp_contract(g, 1, out, x, y)


ad.makevjp(tn.dot, vjpA_dot, vjpB_dot)












































