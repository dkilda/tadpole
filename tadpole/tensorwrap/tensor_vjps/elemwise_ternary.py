#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




###############################################################################
###                                                                         ###
###  VJP's of tensor ternary elementwise functions                          ###
###                                                                         ###
###############################################################################


# --- Value methods --------------------------------------------------------- #

def vjp1_where(g, out, condition, x, y):

    return tn.where(condition, g, tn.space(g).zeros())


def vjp2_where(g, out, condition, x, y):

    return tn.where(condition, tn.space(g).zeros(), g)


ad.makevjp(tn.where, None, vjp1_where, vjp2_where)



















































