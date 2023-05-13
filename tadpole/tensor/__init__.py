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
   Alignment,
   CutoffMode,
   ErrorMode,
   Trunc,
)




# --- Element --------------------------------------------------------------- #

from .element import (
   elem,
)




# --- Space ----------------------------------------------------------------- #

from .space import (
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


from .space import (
   sparsegrad_from_space,
   nullgrad_from_space,
   zeros_from_space,
   ones_from_space,
   unit_from_space,
   eye_from_space,
   rand_from_space, 
   randn_from_space,
   randuniform_from_space,
   units_from_space,
   basis_from_space,  
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
   withinds,
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
   norm,
)




# --- Reindexing ------------------------------------------------------------ #

from .reindexing import (
   reindex, 
   transpose,
   htranspose,
   fuse,
   split,
   squeeze,
   unsqueeze,
   expand,
   flatten,
   diag, # TODO move diag to linalg!
)




# --- Elemwise Unary -------------------------------------------------------- #

from .elemwise_unary import (
   put,
   clip,
   flip,
   cumsum,
)


from .elemwise_unary import (
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


from .elemwise_binary import (
   combos,
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




# --- Decomposition --------------------------------------------------------- #

from .decomp import (
   svd,
   eig,
   eigh,
   qr,
   lq,
)


from .truncation import (
   TruncNull,
   TruncRank,
   TruncAbs,
   TruncRel,
   TruncSum1,
   TruncSum2,
   TruncRelSum1,
   TruncRelSum2,
)


from .truncation import (
   RankCutoff,
   DirectCutoff,
   SumCutoff,
   Error,
)




