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

    #space = tn.space(this).reshape(inds)
    #ones  = space.ones()
    
    result = tn.contract(g, *others, product=product) 

    return tn.match(tn.expand(result, inds), this)

 
"""
    inds    = Indices(*tn.complement_inds(this, *others, g, unique=False)) # TODO find dupes in inds = Indices!
    product = Indices(*tn.union_inds(this)) ^ inds

    #space = tn.space(this).reshape(inds)
    #ones  = space.ones() # FIXME consider using expand(result, inds) instead of contracting with ones!

    indmap = {}
    for ind, freq in util.frequencies(inds).items(): 
        if freq > 1:
           indmap[ind] = tuple(ind.retagged((f"{str(ind)}", f"{i}")) for i in range(freq))

    this = tn.reindex(this, indmap)


    #print("VJPCON-1: ", inds, util.frequencies(inds))

    # indmap = {ind: tuple(ind.retagged(f"{i}") for i in range(freq)) for ind, freq in util.frequencies(inds).items() if freq > 1}
    # result = tn.contract(g, ones, *others, product=tuple(tn.union_inds(this))) 

    result = tn.contract(g, *others, product=product) 

    #print("VJPCON-2: ", result._inds, tuple(tn.union_inds(this)), this._inds, tn.match(result, this)._inds)
    #print("VJPCON-2: ", inds, tn.expand(result, inds)._inds, this._inds)

    #return tn.expand(tn.match(result, this), inds) # tn.expand_like(result, this)


    print("VJPCON-2: ", indmap, util.unpacked_dict(util.inverted_dict(indmap)))
    print("VJPCON-3: ", tn.match(result, this)._inds, tn.reindex(tn.match(result, this), util.unpacked_dict(util.inverted_dict(indmap)))._inds)

    return tn.reindex(tn.match(result, this), util.unpacked_dict(util.inverted_dict(indmap)))
"""



"""
# --- Trace ----------------------------------------------------------------- #

def vjp_trace(g, adx, out, x, inds):

    inds  = tn.Indices(*tn.complement_inds(x, g))
    space = tn.space(this).reshape(inds)
    ones  = space.ones()


    tnc.trace_eyes(x, inds)
    
    result = tn.contract(g, *eyes, product=tn.union_inds(this)) 


    eyes    = [tn.space(x).eye(lind, rind) for rind in rinds]
    product = Indices(*tni.complement_inds(x, *eyes))

     = tn.contract(x, *tnc.trace_eyes(x, inds), product=tnc.TraceIndexProduct(inds))

    return tn.match(result, this)
"""



# --- Record VJPs ----------------------------------------------------------- #

ad.makevjp_combo(tn.contract, vjp_contract)
#ad.makevjp_combo(tn.trace,    vjp_trace)



