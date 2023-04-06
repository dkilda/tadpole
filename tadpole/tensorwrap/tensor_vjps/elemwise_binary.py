#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




# --- Standard math --------------------------------------------------------- #

ad.makevjp(tn.add, lambda g, out, x, y: tn.match(g, x), 
                   lambda g, out, x, y: tn.match(g, y)
)


ad.makevjp(tn.sub, lambda g, out, x, y: tn.match( g, x), 
                   lambda g, out, x, y: tn.match(-g, y),
)


ad.makevjp(tn.mul, lambda g, out, x, y: tn.match(y * g, x), 
                   lambda g, out, x, y: tn.match(x * g, y)
)


ad.makevjp(tn.div, lambda g, out, x, y: tn.match( g / y,        x),   
                   lambda g, out, x, y: tn.match(-g * x / y**2, y)
)




# --- Gradient accumulation ------------------------------------------------- #

ad.makevjp(tn.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g
)













































