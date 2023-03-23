#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .backends import set_default as set_backend


from .types import (
   TensorLike,
   Pluggable,
)


from .core import (
   Tensor,
   NullGrad,
   SparseGrad,
)


from .core import (
   astensor,
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
   allclose,
   allequal,
   isclose,
   isequal,
   notequal,
   logical_and,
   logical_or,
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


from .space import (
   TensorSpace,
)


from .space import (
   sparsegrad,
   nullgrad,
   zeros,
   ones,
   unit,
   rand, 
   randn,
   randuniform,
   units,
   basis,  
)


from .space import (
   sparsegrad_from_space,
   nullgrad_from_space,
   zeros_from_space,
   ones_from_space,
   unit_from_space,
   rand_from_space, 
   randn_from_space,
   randuniform_from_space,
   units_from_space,
   basis_from_space,  
)


from .operations import (
   allof,
   anyof,
   count_nonzero,
   sign,
   put,
   iscomplex,
)

from .operations import (
   reindex,
   fuse,
   split,
   transpose,
   squeeze,
   unsqueeze,
   sumover,
   cumsum,
)


from .operations import (
   amax,
   amin,
   absolute,
   flip,
   clip,
   where,
)


from .operations import (
   conj,
   real,
   imag,
   sqrt,
   log,
   exp,
   sin,
   cos,
   tan,
   arcsin,
   arccos,
   arctan,
   sinh,
   cosh,
   tanh,
   arcsinh,
   arccosh,
   arctanh,
)


from .operations import (
   norm,
   expm,
   htranspose,
)


from .contract import (
   einsum,
   dot,
   kron,
)


from .contract import (
   svd,
   eig,
   eigh,
   qr,
   lq,
)


from .trunc import (
   Trunc,
   NullTrunc,
   TruncRank,
   TruncAbs,
   TruncRel,
   TruncSum1,
   TruncSum2,
   TruncRelSum1,
   TruncRelSum2,
)



















