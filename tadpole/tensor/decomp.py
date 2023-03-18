#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad




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




class NullTrunc(Trunc):

   def rank(self, S):

       return S.shape[0]


   def error(self, S):

       return 0


   def apply(self, U, S, VH):

       return U, S, VH




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
      
          cumsum = op.cumsum(S**self._renorm, 0)
          renorm = (cumsum[-1] / cumsum[rank - 1]) ** (1 / self._renorm) 

          S1 = S1 * renorm

       return U1, S1, VH1  





class CutoffMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S):
       pass
       



class RankCutoff(CutoffMode):

   def apply(self, S):

       return S.shape[0]




class DirectCutoff(CutoffMode):

   def __init__(self, cutoff, relative):

       self._cutoff   = cutoff
       self._relative = relative


   def _target_cutoff(self, S):

       if self._relative:
          return self._cutoff * S[0]

       return self._cutoff


   def apply(self, S):

       return op.count_nonzero(S, lambda x: x > self._target_cutoff(S))




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

       cumsum = op.cumsum(S**power, 0)

       return 1 + op.count_nonzero(
                     (cumsum[-1] - cumsum) > self._target_cutoff(cumsum)
                  ) 




class ErrorMode(abc.ABC):

   @abc.abstractmethod
   def apply(self, S, rank):
       pass



class Error(ErrorMode):

   def __init__(self, power, relative):

       self._power    = power
       self._relative = relative


   def apply(self, S, rank):

       if S.shape[0] == rank: 
          return 0

       cumsum = op.cumsum(S**power, 0)
       error  = (cumsum[-1] - cumsum)[rank-1]  

       if self._relative:
          error = error / cumsum[-1]

       return error




class TruncRank(GenTrunc):

   def __init__(self, max_rank=None, error=None, renorm=None):  

       super().__init__(RankCutoff(), max_rank, renorm, error)       




class TruncAbs(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):   

       cutoff = DirectCutoff(cutoff, relative=False)
       
       super().__init__(cutoff, max_rank, renorm, error) 
       



class TruncRel(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):   

       cutoff = DirectCutoff(cutoff, relative=True)
       
       super().__init__(cutoff, max_rank, renorm, error)    
       


"""
class TruncRelSum1(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

       cutoff = SumCutoff(cutoff, power=1, relative=True)

       if error is None: 
          error = Error(power=1, relative=True) 
       
       if renorm is None:
          renorm = 1

       super().__init__(cutoff, max_rank, renorm, error)        



class TruncRelSum2(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

       cutoff = SumCutoff(cutoff, power=2, relative=True)

       if error is None: 
          error = Error(power=2, relative=True) 
       
       if renorm is None:
          renorm = 2

       super().__init__(cutoff, max_rank, renorm, error)  
       



class TruncSum1(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

       cutoff = SumCutoff(cutoff, power=1, relative=False)

       if error is None: 
          error = Error(power=1, relative=False) 
       
       if renorm is None:
          renorm = 1

       super().__init__(cutoff, max_rank, renorm, error)        



class TruncSum2(GenTrunc):

   def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

       cutoff = SumCutoff(cutoff, power=2, relative=False)

       if error is None: 
          error = Error(power=2, relative=False) 
       
       if renorm is None:
          renorm = 2

       super().__init__(cutoff, max_rank, renorm, error) 
"""



def trunc_sum_cls(power, relative):

    class TruncSum(GenTrunc):

       def __init__(self, cutoff, max_rank=None, error=None, renorm=None):  

           cutoff = SumCutoff(cutoff, power, relative)

           if error is None: 
              error = Error(power, relative) 
       
           if renorm is None:
              renorm = power

           super().__init__(cutoff, max_rank, renorm, error) 


    return TruncSum




TruncRelSum1 = trunc_sum_cls(power=1, relative=True)
TruncRelSum2 = trunc_sum_cls(power=2, relative=True)
TruncSum1    = trunc_sum_cls(power=1, relative=False)
TruncSum2    = trunc_sum_cls(power=2, relative=False)





#################################################################################
#################################################################################
#################################################################################

class RankTrunc(Trunc):

   def __init__(self, max_rank):

       self._max_rank = max_rank


   def rank(self, S):


   def error(self, S):


   def trunc(self, U, S, VH):




class RelTrunc(Trunc):

   def __init__(self, cutoff=1e-10, max_rank=None):

       self._cutoff   = cutoff
       self._max_rank = max_rank


   def rank(self, S):

       rank = misc.count_nonzero(S > self._cutoff * S[0])
       rank = max(rank, 1)

       if self._max_rank > 0:
          return min(rank, self._max_rank)

       return rank



   def error(self, S):

       return 


   def trunc(self, U, S, VH):

       if self._renorm_power > 0:
          fac = self.renorm_factor(S)

       S  = S[: self.rank(S)]
       U  = U[..., : self.rank(S)]
       VH = VH[: self.rank(S), ...]   

       if self._renorm_power > 0:
          S *= fac

       return U, S, VH




class Trunc:

   def __init__(self, engine, cutoff=1e-10, max_rank=None):

       self._engine   = engine
       self._cutoff   = cutoff
       self._max_rank = max_rank


   def rank(self, S):

       rank = self._engine.rank(S, self._cutoff)
       rank = max(rank, 1)

       if self._max_rank > 0:
          return min(rank, self._max_rank)

       return rank


   def renorm(self, S):

       def _renorm(x):

           if self._renorm == 0:
              return x

           cumsum = op.cumsum(S**self._renorm, 0)
           renorm = (cumsum[-1] / cumsum[x.shape[0] - 1]) ** (1 / self._renorm)

           return x * renorm

       return _renorm
 

   def apply(self, U, S, VH):

       rank = self.rank(S)

       S1  = S[: rank]
       U1  = U[..., : rank]
       VH1 = VH[: rank, ...]   

       if self._renorm > 0:
      
          cumsum = op.cumsum(S**self._renorm, 0)
          renorm = (cumsum[-1] / cumsum[rank - 1]) ** (1 / self._renorm) 

          S1 = S1 * renorm

       return U1, S1, VH1   


   def error(self, S):    

       if chi == S.shape[0]: 
          return 0

       rank   = self.rank(S)
       cumsum = op.cumsum(S**power, 0)
       error  = (cumsum[-1] - cumsum)[rank-1]  

       if relative:
          return error / cumsum[-1]

       return error         














































