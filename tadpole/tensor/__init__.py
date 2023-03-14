#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .backends import set_default as set_backend


from .types import (
   TensorLike,
   Pluggable,
)


from .grad import (
   ZeroGrad,
   SparseGrad,
)


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
   astensor,
)


from .core import (
   Tensor,
   TensorSpace,
)


from .core import (
   allclose,
   allequal,
   isclose,
   isequal,
   notequal,
   logical_and,
   logical_or,
)


from .core import (
   allclose,
   allequal,
   isclose,
   isequal,
   notequal,
   logical_and,
   logical_or,
)


from .core import (
   copy,
   todense,
   withdata,
   space,
   item,
   dtype,
   size,
   ndim,
   shape,
)


from .core import (
   addgrads,
   getitem,
   neg,
   add,
   sub,
   mul,
   div,
   power,
)


from .operations import *




