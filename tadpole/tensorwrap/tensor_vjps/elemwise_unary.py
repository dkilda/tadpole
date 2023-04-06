#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn






# --- Value methods --------------------------------------------------------- #





# --- Element access -------------------------------------------------------- #




# --- Standard math --------------------------------------------------------- #

ad.makevjp(tn.neg, lambda g, out, x: -g)
ad.makevjp(tn.sin, lambda g, out, x:  g * tn.cos(x))
ad.makevjp(tn.cos, lambda g, out, x: -g * tn.sin(x))






































