#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .backends import set_default as set_backend


from .core import (
   sparse,
   zeros,
   ones,
   unit,
   rand, 
   randn,
   randuniform,
   units,
   basis,
   fromfun,  
   asarray,
)


from .core import (
   Array,
   ArraySpace,
)


from .grad import (
   ZeroGrad,
   SparseGrad,
)


from .logical    import *
from .operations import *




