#!/usr/bin/env python3
# -*- coding: utf-8 -*-




###############################################################################
###                                                                         ###
###  Creation methods                                                       ###
###                                                                         ###
###############################################################################


# --- Create unit array ----------------------------------------------------- #

def unit(self, shape, idx, **opts):

    array      = self.zeros(shape, **opts)
    array[idx] = 1

    return array

