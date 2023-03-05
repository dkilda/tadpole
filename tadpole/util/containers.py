#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util as util




###############################################################################
###                                                                         ###
### Interface for tuple-like types.                                         ###
###                                                                         ###
###############################################################################


# --- TupleLike interface --------------------------------------------------- #

class TupleLike(abc.ABC):

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass

   @abc.abstractmethod
   def __len__(self):
       pass

   @abc.abstractmethod
   def __contains__(self, x):
       pass

   @abc.abstractmethod
   def __iter__(self):
       pass

   @abc.abstractmethod
   def __getitem__(self, idx):
       pass




###############################################################################
###                                                                         ###
###  Function outputs                                                       ###
###                                                                         ###
###############################################################################


# --- Function outputs ------------------------------------------------------ #

class Outputs(TupleLike):  

   def __init__(self, *outputs):

       if len(outputs) == 0:
          raise ValueError(
             f"Outputs: cannot construct Outputs with zero values"
          ) 
          
       self._outputs = outputs


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("outputs", self._outputs)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.ref(self._outputs, other._outputs)

       return bool(log)


   def __hash__(self):

       return hash(self._outputs)


   def __len__(self):

       return len(self._outputs)


   def __contains__(self, x):

       return x in self._outputs


   def __iter__(self):

       return iter(self._outputs)  


   def __getitem__(self, idx):

       return self._outputs[idx]


   def unpack(self):

       if len(self._outputs) == 1:
          return self._outputs[0]

       return self._outputs




