#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# --- Types ----------------------------------------------------------------- #

from .types import (
   Pluggable,
   Grad,
   Tensor,
   Space,
   Element,
)


from .types import (
   Engine,
   IndexProduct,
)




# --- Element --------------------------------------------------------------- #

from .element import (
   elem,
)




# --- Space ----------------------------------------------------------------- #

from .space import (
   tensorspace,
   TensorSpace,
)


from .space import (
   sparsegrad,
   nullgrad,
   zeros,
   ones,
   unit,
   eye,
   rand, 
   randn,
   randuniform,
   units,
   basis,  
)




# --- Core ------------------------------------------------------------------ #

from .core import (
   TensorGen,
   NullGrad,
   SparseGrad,
)


from .core import (
   astensor,
   copy,
   todense,
   tonull,
   withdata,
   space,
   item,
   dtype,
   size,
   ndim,
   shape,
)




# --- Interaction ----------------------------------------------------------- #

from .interaction import (
   union_inds,
   overlap_inds,
   complement_inds,
   match,
   astype_like,
   reshape_like,
   transpose_like,
   expand_like,
)




# --- Reduction ------------------------------------------------------------- #

from .reduction import (
   allof,
   anyof,
   amax,
   amin,
   count_nonzero,
)


from .reduction import (
   sumover,
)




# --- Reindexing ------------------------------------------------------------ #

from .reindexing import (
   reindexto,
   reindex, 
   transpose,
   htranspose,
   fuse,
   split,
   squeeze,
   unsqueeze,
   expand,
   flatten,
)




# --- Elemwise Unary -------------------------------------------------------- #

from .elemwise_unary import (
   put,
   clip,
   flip,
   cumsum,
)


from .elemwise_unary import (
   asdata,
   iscomplex,
   astype,
   getitem,
   ungetitem,
)


from .elemwise_unary import (
   floor, 
   neg,
   sign,
   conj,
   real,
   imag,
   absolute,
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




# --- Elemwise Binary ------------------------------------------------------- #

from .elemwise_binary import (
   addgrads,
   add,
   sub,
   mul,
   div,
   mod,
   floordiv,
   power,
)


from .elemwise_binary import (
   allclose,
   isclose,
   allequal,
   isequal,
   notequal,
   greater,
   less,
   greater_equal,
   less_equal,
   logical_and,
   logical_or,
)




# --- Elemwise Ternary ------------------------------------------------------ #

from .elemwise_ternary import (
   where,
)




# --- Contraction ----------------------------------------------------------- #

from .contraction import (
   contract,
   dot,
   kron,
   trace,
)




