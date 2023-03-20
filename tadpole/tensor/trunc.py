#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util  as util
import tadpole.array as ar




###############################################################################
###                                                                         ###
###  Cutoff mode for truncation                                             ###
###                                                                         ###
###############################################################################


# --- Cutoff mode interface ------------------------------------------------- #

class CutoffMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S):
       pass
       



# --- Cutoff by rank -------------------------------------------------------- #

class RankCutoff(CutoffMode):

   def apply(self, S):

       return S.shape[0]




# --- Cutoff by direct tolerance threshold ---------------------------------- #

class DirectCutoff(CutoffMode):

   def __init__(self, cutoff, relative):

       self._cutoff   = cutoff
       self._relative = relative


   def _target_cutoff(self, S):

       if self._relative:
          return self._cutoff * S[0]

       return self._cutoff


   def apply(self, S):

       return ar.count_nonzero(S, lambda x: x > self._target_cutoff(S))




# --- Cutoff by sum threshold ----------------------------------------------- #

class SumCutoff(CutoffMode):

   def __init__(self, cutoff, power, relative):

       self._cutoff   = cutoff
       self._power    = power
       self._relative = relative


   def _target_cutoff(self, cumsum):

       if self._relative:
          return self._cutoff * cumsum[-1]

       return self._cutoff


   def apply(self, S):

       cumsum = ar.cumsum(S**power, 0)

       return 1 + ar.count_nonzero(
                     (cumsum[-1] - cumsum) > self._target_cutoff(cumsum)
                  ) 




###############################################################################
###                                                                         ###
###  Error mode of truncation                                               ###
###                                                                         ###
###############################################################################


# --- Error mode interface -------------------------------------------------- #

class ErrorMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S, rank):
       pass




# --- Error mode ------------------------------------------------------------ #

class Error(ErrorMode):

   def __init__(self, power, relative):

       self._power    = power
       self._relative = relative


   def apply(self, S, rank):

       if S.shape[0] == rank: 
          return 0

       cumsum = ar.cumsum(S**power, 0)
       error  = (cumsum[-1] - cumsum)[rank-1]  

       if self._relative:
          error = error / cumsum[-1]

       return error




###############################################################################
###                                                                         ###
###  Truncation of singular/eigenvalue decompositions                       ###
###                                                                         ###
###############################################################################


# --- Truncation interface -------------------------------------------------- #

class Trunc(abc.ABC):

   @abc.abstractmethod
   def rank(self, S):
       pass

   @abc.abstractmethod
   def error(self, S):
       pass

   @abc.abstractmethod
   def apply(self, U, S, VH):
       pass




# --- Null truncation ------------------------------------------------------- #

class NullTrunc(Trunc):

   def rank(self, S):

       return S.shape[0]


   def error(self, S):

       return 0


   def apply(self, U, S, VH):

       return U, S, VH




# --- General truncation ---------------------------------------------------- #

class GenTrunc(Trunc):

   def __init__(self, max_rank=None, cutoff=None, error=None, renorm=None):

       if max_rank is None:
          max_rank = -1

       if cutoff is None:
          cutoff = DirectCutoff(1e-10, relative=True)

       if error is None:
          error = Error(power=2, relative=True)

       if renorm is None:
          renorm = 0
        
       self._max_rank = max_rank
       self._cutoff   = cutoff
       self._error    = error
       self._renorm   = renorm


   def rank(self, S):

       rank = self._cutoff.apply(S)
       rank = max(rank, 1)

       if self._max_rank > 0:
          return min(rank, self._max_rank)

       return rank


   def error(self, S):

       return self._error.apply(S, self.rank(S))


   def apply(self, U, S, VH):

       rank = self.rank(S)

       S1  = S[: rank]
       U1  = U[..., : rank]
       VH1 = VH[: rank, ...]   

       if self._renorm > 0:
      
          cumsum = ar.cumsum(S**self._renorm, 0)
          renorm = (cumsum[-1] / cumsum[rank - 1]) ** (1 / self._renorm) 

          S1 = S1 * renorm

       return U1, S1, VH1  




# --- Truncation by rank ---------------------------------------------------- #

class TruncRank(GenTrunc):

   def __init__(self, max_rank=None, error=None, renorm=None):  

       super().__init__(RankCutoff(), max_rank, error, renorm)       




# --- Truncation by absolute cutoff ----------------------------------------- #

class TruncAbs(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):   

       cutoff = DirectCutoff(cutoff, relative=False)
       
       super().__init__(cutoff, max_rank, error, renorm) 
      



# --- Truncation by relative cutoff ----------------------------------------- # 

class TruncRel(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):   

       cutoff = DirectCutoff(cutoff, relative=True)
       
       super().__init__(cutoff, max_rank, error, renorm)  




# --- Template for truncation-by-sum classes -------------------------------- #

def trunc_sum_cls(power, relative):

    class TruncSum(GenTrunc):

       def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

           cutoff = SumCutoff(cutoff, power, relative)

           if error is None: 
              error = Error(power, relative) 
       
           if renorm is None:
              renorm = power

           super().__init__(cutoff, max_rank, error, renorm) 


    return TruncSum




# --- Truncation by absolute sum cutoff ------------------------------------- #

TruncSum1 = trunc_sum_cls(power=1, relative=False)
TruncSum2 = trunc_sum_cls(power=2, relative=False)




# --- Truncation by relative sum cutoff ------------------------------------- #
       
TruncRelSum1 = trunc_sum_cls(power=1, relative=True)
TruncRelSum2 = trunc_sum_cls(power=2, relative=True)




