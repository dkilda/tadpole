#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools




###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Decorator for cacheable methods --------------------------------------- #

def cacheable(fun):

    name = f"_{fun.__name__}_cached"

    @functools.wraps(fun)
    def wrap(self):

        try:
           return getattr(self, name) 

        except AttributeError:
           out = fun(self)
           setattr(self, name, out)
           return out

    return wrap

















































