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
###  VJP's of tensor decomposition functions                                ###
###                                                                         ###
###############################################################################


# --- Explicit-rank decompositions ------------------------------------------ #

def vjpU_svd(g, out, x, inds, **opts):

    U, S, VH = out



    alignment = opts.get("alignment", "left")

    if not isinstance(alignment, Alignment):
       alignment = {
                    "left":  LeftAlignment,
                    "right": RightAlignment,
                   }[alignment](inds)

    V = tn.htranspose(
                      VH, 
                      alignment.linds(Indices(*tn.union_inds(x))), 
                      alignment.rinds(Indices(*tn.union_inds(x)))
                     )



def vjpS_svd(g, out, x, inds, **opts):
    pass
    
    



def vjpV_svd(g, out, x, inds, **opts):
    pass



ad.makevjp(tn.svd, vjpU_svd, vjpS_svd, vjpV_svd)



# --- Hidden-rank decompositions -------------------------------------------- #
















































