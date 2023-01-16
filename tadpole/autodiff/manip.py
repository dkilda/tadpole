#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.wrapper as tdwrap




# --- Wrapper for gradient addition ----------------------------------------- #

def add_grads(net_g, g): # TODO impl and use add() function, with @differentiable decorator 
                         #      (or overload __add__ operator to make it @differentiable)
    if net_g is None:  
       return g

    return tdwrap.functions.add(net_g, g)






















































