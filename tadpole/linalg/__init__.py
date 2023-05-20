#!/usr/bin/env python3
# -*- coding: utf-8 -*-




# --- Types ----------------------------------------------------------------- #

from .types import (
   CutoffMode,
   ErrorMode,
   Trunc,
)




# --- Truncation ------------------------------------------------------------ #

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




# --- Linalg routines ------------------------------------------------------- #

from .core import (
   svd,
   eig,
   eigh,
   qr,
   lq,
)


from .core import (
   norm,
   trace,
   det,
   inv,
   tril,
   triu,
   diag,
)


from .core import (
   solve,
   trisolve,
)










