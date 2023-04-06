#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn



# --- Reindexing and reshaping methods -------------------------------------- #

def vjp_transpose(g, out, x, *output_inds):

    return tn.transpose(g, *tn.union_inds(x))
 

ad.makevjp(tn.transpose, vjp_transpose)















































