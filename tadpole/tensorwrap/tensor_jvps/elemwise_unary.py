#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn






# --- Value methods --------------------------------------------------------- #





# --- Element access -------------------------------------------------------- #




# --- Standard math --------------------------------------------------------- #

ad.makejvp(tn.neg, lambda g, out, x: -g)
ad.makejvp(tn.sin, lambda g, out, x:  g * tn.cos(x))
ad.makejvp(tn.cos, lambda g, out, x: -g * tn.sin(x))






































