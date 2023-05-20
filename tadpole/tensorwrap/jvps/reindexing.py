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
###  Tensor reindexing functions                                            ###
###                                                                         ###
###############################################################################


# --- Reindexing and reshaping methods -------------------------------------- #

ad.makejvp(tn.reindex,    "linear")
ad.makejvp(tn.transpose,  "linear")    
ad.makejvp(tn.fuse,       "linear")
ad.makejvp(tn.split,      "linear")
ad.makejvp(tn.squeeze,    "linear")
ad.makejvp(tn.unsqueeze,  "linear")
ad.makejvp(tn.expand,     "linear")




def jvp_flatten(g, out, x, ind):

    ind, = tn.union_inds(out)

    return tn.flatten(g, ind)


ad.makejvp(tn.flatten, "linear")



















