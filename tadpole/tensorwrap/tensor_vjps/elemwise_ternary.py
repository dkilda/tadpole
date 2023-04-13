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
###  Ternary elementwise functions                                          ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def vjpA_where(g, out, condition, x, y):

    return tn.where(condition, g, tn.space(g).zeros())


def vjpB_where(g, out, condition, x, y):

    return tn.where(condition, tn.space(g).zeros(), g)


ad.makevjp(tn.where, "null", vjpA_where, vjpB_where)




