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
###  VJP's of tensor reindexing functions                                   ###
###                                                                         ###
###############################################################################


# --- Reindexing and reshaping methods -------------------------------------- #

def vjp_reindex(g, out, x, indmap):

    return tn.reindex(g, util.inverted_dict(indmap)) 



def vjp_transpose(g, out, x, *output_inds):

    return tn.transpose(g, *tn.union_inds(x))
 


def vjp_fuse(g, out, x, fusemap):

    inputs   = Indices(*tn.union_inds(x))
    outputs  = Indices(*tn.union_inds(out))
    splitmap = {}

    for inp, out in fusemap.items():

        inp = inputs.map(*inp)
        out = outputs.map(out)

        splitmap[out] = inp 

    return tn.transpose_like(tn.split(g, splitmap), x)



def vjp_split(g, out, x, splitmap):

    inputs  = Indices(*tn.union_inds(x))
    outputs = Indices(*tn.union_inds(out))
    fusemap = {}

    for inp, out in splitmap.items():

        inp, = inputs.map(inp)
        out  = outputs.map(*out)

        fusemap[out] = inp 

    return tn.transpose_like(tn.fuse(g, fusemap), x)



def vjp_squeeze(g, out, x, inds=None):

    singletons = tn.complement_inds(x, out)

    return tn.transpose_like(tn.unsqueeze(g, singletons), x)



def vjp_unsqueeze(g, out, x, inds):

    return tn.transpose_like(tn.squeeze(g, inds), x)



def vjp_expand(g, out, x, inds):

    return tn.reshape_like(g, x)



def vjp_diag(g, out, x, ind=None):

    if not ind:
       return tn.diag(g)

    xinds = tuple(tn.union_inds(x))

    return tn.reindex(tn.diag(g), {ind: xinds})



    
ad.makevjp(tn.reindex,    vjp_reindex)
ad.makevjp(tn.transpose,  vjp_transpose)    
ad.makevjp(tn.fuse,       vjp_fuse)
ad.makevjp(tn.split,      vjp_split)
ad.makevjp(tn.squeeze,    vjp_squeeze)
ad.makevjp(tn.unsqueeze,  vjp_unsqueeze)
ad.makevjp(tn.expand,     vjp_expand)
ad.makevjp(tn.diag,       vjp_diag)




