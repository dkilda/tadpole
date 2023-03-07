#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from . import my_vjps
from . import my_jvps


from .backends import set_default as set_backend


from .types import (
   Pluggable,
   ArrayLike,
)


from .grad import (
   ZeroGrad,
   SparseGrad,
)


from .core import (
   Array,
   allequal,
   allclose,
   allallequal,
   allallclose,
)


from .core import (
   fromfun,  
   asarray,
   sparse,
   zeros,
   ones,
   unit,
   rand, 
   randn,
   randuniform,
   units,
   basis,
)


from .operations import (
   getitem,  
   put,
   reshape,
)


from .operations import (
   sin,
   cos,   
)


from .operations import (
   addto,
   add,
   sub,
   mul,   
)


from .operations import (
   einsum,   
)











