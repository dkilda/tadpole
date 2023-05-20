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

def jvpA_where(g, out, condition, x, y):

    return tn.where(condition, g, tn.space(g).zeros())


def jvpB_where(g, out, condition, x, y):

    return tn.where(condition, tn.space(g).zeros(), g)


ad.makejvp(tn.where, "null", jvpA_where, jvpB_where)




