#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from . import vjps
from . import jvps


from tadpole.array.core import (
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


from tadpole.array.grad import (
   ZeroGrad,
   SparseGrad,
)



from .node import (
   Node,
)


from .operations import (
   dtype,
   size,
   ndim,
   shape,
   allequal,
   allclose,
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
   addgrads,
   add,
   sub,
   mul,   
)


from .operations import (
   einsum,   
)




