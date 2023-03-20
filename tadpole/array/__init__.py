#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.array.backends import set_default as set_backend
from tadpole.array.core     import ArrayLike, ArraySpaceLike

from tadpole.array.void   import Array as VoidArray
from tadpole.array.unary  import Array as UnaryArray
from tadpole.array.binary import Array as BinaryArray
from tadpole.array.nary   import Array as NaryArray




from .space import (
   arrayspace,
   ArraySpace,
)


from .void import (
   iscomplex_type,
)


from .void import (
   zeros,
   ones,
   unit,
   eye,
   rand,
   randn,
   randuniform,
)


from .unary import (
   asarray,
   copy,
   astype,
   dtype,
   iscomplex,
)


from .unary import (
   size,
   ndim,
   shape,
   reshape,
   transpose,
   moveaxis,
   squeeze,
   unsqueeze,
   sumover,
   cumsum,
)


from .unary import (
   item,
   allof,
   anyof,
   amax,
   amin,
   sign,
   absolute,
   flip,
   clip,
   count_nonzero,
   put,
   argsort,
   diag,
)


from .unary import (
   conj,
   real,
   imag,
   sqrt,
   log,
   exp,
   neg,
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


from .unary import (
   svd,
   qr,
   eig,
   eigh,
   expm,
   norm,
)


from .binary import (
   allclose,
   allequal,
   isclose,
   isequal,
   notequal,
   logical_and,
   logical_or,
   add,
   sub,
   mul,
   div,
   power,
   dot,
   kron,
)


from .nary import (
   where,
   einsum,
)


