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
###  JVP's of tensor contraction                                            ###
###                                                                         ###
###############################################################################


# --- Contraction ----------------------------------------------------------- #

ad.makejvp_combo(tn.contract, "linear")
ad.makejvp_combo(tn.trace,    "linear")















































