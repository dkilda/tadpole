#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

from tadpole.linalg.types import (
   CutoffMode,
   ErrorMode,
   Trunc,
)

from tadpole.linalg.truncation import (
   TruncNull,
)

from tadpole.tensor.engine import (
   TrainTensorData,
   TooManyArgsError,
)

from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Linalg solver engine and operator                                      ###
###                                                                         ###
###############################################################################


# --- Linalg solver factory ------------------------------------------------- #

def linalg_solver(*xs):

    engine = EngineLinalgSolver()

    for x in xs:
        engine = x.pluginto(engine)

    return engine.operator()




# --- Linalg solver engine -------------------------------------------------- #

class EngineLinalgSolver(tn.Engine):

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._train, other._train)

       return bool(log)


   @property
   def _size(self):

       return 2


   def attach(self, data, inds):

       if self._train.size() == self._size:
          raise TooManyArgsError(self, self._size)

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       return LinalgSolver(self._train.data(), self._train.inds())




# --- Linalg solver operator ------------------------------------------------ #

class LinalgSolver:

   # --- Construction --- #

   def __init__(self, data, inds): 

       if all(x.ndim == 2 for x in inds):
          raise ValueError(
             f"LinalgSolver: input must have ndim = 2, but "
             f"data  ndims = {tuple(x.ndim for x in data)}, "
             f"index ndims = {tuple(x.ndim for x in inds)}."
          )

       self._data = data
       self._inds = inds


   # --- Private helpers --- #

   def _apply(self, fun, *args, **kwargs):

       if self._indsA[0] != self._indsB[0]:
          raise ValueError(
             f"LinalgSolver: "
             f"A and B tensors must have matching left indices, but "
             f"lind of A {self._indsA[0]} != lind of B {self._indsB[0]}."
          )

       data = fun(self._dataA, self._dataB, *args, **kwargs)
       inds = Indices(self._indsA[1], self._indsB[1]) 

       return tn.TensorGen(data, inds) 


   @property
   def _dataA(self):
       return self._data[0]

   @property
   def _dataB(self):
       return self._data[1]

   @property
   def _indsA(self):
       return self._inds[0]

   @property
   def _indsB(self):
       return self._inds[1]


   # --- Linear algebra solvers --- #

   def solve(self):
   
       return self._apply(ar.solve)


   def trisolve(self, which=None):

       if which is None:
          which = "upper"

       return self._apply(ar.trisolve, which=which)
      



###############################################################################
###                                                                         ###
###  Standalone functions corresponding to LinalgSolver methods             ###
###                                                                         ###
###############################################################################


# --- Linear algebra solvers ------------------------------------------------ #

@ad.differentiable
def solve(a, b):

    op = linalg_solver(a, b)
    return op.solve()  


@ad.differentiable
def trisolve(a, b, which=None):

    op = linalg_solver(a, b)
    return op.trisolve(which=which) 




