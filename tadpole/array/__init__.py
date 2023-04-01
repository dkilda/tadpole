#!/usr/bin/env python3
# -*- coding: utf-8 -*-




# --- Backends -------------------------------------------------------------- #

from tadpole.array.backends import set_default as set_backend




# --- Types ----------------------------------------------------------------- #

from .types import (
   Array, 
   Space,
)




# --- Arrays ---------------------------------------------------------------- #

from tadpole.array.void   import Array as ArrayVoid
from tadpole.array.unary  import Array as ArrayUnary
from tadpole.array.binary import Array as ArrayBinary
from tadpole.array.nary   import Array as ArrayNary




# --- Space ----------------------------------------------------------------- #

from .space import (
   arrayspace,
   ArraySpace,
)




# --- Void ------------------------------------------------------------------ #

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




# --- Unary ----------------------------------------------------------------- #

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
   broadcast_to,
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
   lq,
   eig,
   eigh,
   expm,
   norm,
)




# --- Binary ---------------------------------------------------------------- #

from .binary import (
   allclose,
   allequal,
   isclose,
   isequal,
   notequal,
   ismore,
   isless,
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




# --- Nary ------------------------------------------------------------------ #

from .nary import (
   where,
   einsum,
)





