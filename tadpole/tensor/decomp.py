#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.tensor.funcall    as funcall
import tadpole.tensor.truncation as truncation

from tadpole.tensor.funcall import (
   FunCall,
)

from tadpole.tensor.truncation import (
   NullTrunc,
)




###############################################################################
###                                                                         ###
###  Index partitioning logic:                                              ### 
###  -- aligns input data by defining left/right indices                    ###
###  -- creates left/right/middle output tensors from data computed         ###
###     by an array decomposition function                                  ###
###                                                                         ###
###############################################################################


# --- Alignment interface --------------------------------------------------- #

class Alignment(abc.ABC):

   @abc.abstractmethod
   def left(self, inds):
       pass

   @abc.abstractmethod
   def right(self, inds):
       pass




# --- Left alignment -------------------------------------------------------- #

class LeftAlignment(Alignment):

   def __init__(self, partinds):

       self._partinds = partinds


   def linds(self, inds):

       return self._partinds
       

   def rinds(self, inds):

       return inds.remove(*self._partinds) 




# --- Right alignment -------------------------------------------------------- #

class RightAlignment(Alignment):

   def __init__(self, partinds):

       self._partinds = partinds


   def linds(self, inds):

       return inds.remove(*self._partinds)
       

   def rinds(self, inds):

       return self._partinds




# --- LinkLike interface ---------------------------------------------------- #

class LinkLike(abc.ABC):

   @abc.abstractmethod
   def ind(self, size):
       pass




# --- Link between partitions ----------------------------------------------- #

class Link(LinkLike):

   def __init__(self, name):

       self._name = name
       self._ind  = None


   def ind(self, size):

       if self._ind is None:
          self._ind = Index(self._name, size)

       if size != len(self._ind):
          raise ValueError(
             f"{type(self).__name__}: an attempt to resize link index to "
             f"an incompatible size {size} != original size {len(self._ind)}"
          )

       return self._ind
 



# --- Partition ------------------------------------------------------------- #

class Partition:

   def __init__(self, inds, linds, rinds, link):

       self._inds  = inds
       self._linds = linds
       self._rinds = rinds
       self._link  = link

   
   @property
   def _laxes(self):

       return self._inds.axes(*self._linds)


   @property
   def _raxes(self):

       return self._inds.axes(*self._rinds)


   def aligndata(self, data):

       data = ar.transpose(data, (*self._laxes, *self._raxes))
       data = ar.reshape(data,   (self._linds.size, self._rinds.size))

       return data


   def ltensor(self, data):

       data = ar.reshape(data, (*self._linds.shape, -1))

       sind = self._link.ind(data.shape[-1])
       inds = self._linds.push(sind)

       return core.Tensor(data, inds)


   def rtensor(self, data):

       data = ar.reshape(rdata, (-1, *self._rinds.shape))

       sind = self._link.ind(data.shape[0])
       inds = self._rinds.add(sind)

       return core.Tensor(data, inds)


   def stensor(self, data):

       sind = self._link.ind(data.shape[0])

       return core.Tensor(data, Indices(sind))




# --- Partition factory ----------------------------------------------------- #

class Partitions:

   def __init__(self, alignment, link):

       self._alignment = alignment
       self._link      = link


   def create(self, inds):

       return Partition(
          inds, self._alignment.linds(), self._alignment.rinds(), self._link
       )




# --- Creating partitions factory ------------------------------------------- #

def make_partitions(linkname, inds, which):

    link      = Link(linkname)
    alignment = {
                 "left":  LeftAlignment,
                 "right": RightAlignment,
                }[which](inds)

    return Partitions(alignment, link)




###############################################################################
###                                                                         ###
###  Tensor decomposition calls                                             ###
###                                                                         ###
###############################################################################


# --- Explicit-rank decomposition call -------------------------------------- #

class ExplicitDecomp(FunCall):

   def __init__(self, engine, partitions, trunc):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine     = engine
       self._partitions = partitions
       self._trunc      = trunc


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()

       partition = self._partitions.create(inds)
       outdata   = self._engine.execute(partition.aligndata(data))

       error   = self._trunc.error(outdata[1])
       outdata = self._trunc.apply(*outdata)

       return (
               partition.ltensor(outdata[0]), 
               partition.stensor(outdata[1]), 
               partition.rtensor(outdata[2]), 
               error,
              )




# --- Hidden-rank decomposition call ---------------------------------------- #

class HiddenDecomp(FunCall):

   def __init__(self, engine, partitions):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine     = engine
       self._partitions = partitions


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       data, = self._engine.datas()
       inds, = self._engine.inds()

       partition = self._partitions.create(inds)
       outdata   = self._engine.execute(partition.aligndata(data))

       return (
               partition.ltensor(outdata[0]), 
               partition.rtensor(outdata[1]),
              )




# --- Specialized decomposition methods ------------------------------------- #

@ad.differentiable
def svd(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.svd(data)

    partitions = make_partitions(name, inds, which)
    decomp     = ExplicitDecomp(fun, partitions, trunc)
    
    return Args(x).pluginto(decomp)




@ad.differentiable
def eig(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eig(data)

    partitions = make_partitions(name, inds, which)
    decomp     = ExplicitDecomp(fun, partitions, trunc)
    
    return Args(x).pluginto(decomp)




@ad.differentiable
def eigh(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eigh(data)

    partitions = make_partitions(name, inds, which)
    decomp     = ExplicitDecomp(fun, partitions, trunc)
    
    return Args(x).pluginto(decomp)




@ad.differentiable
def qr(x, name, inds, which="left"):

    def fun(data):
        return ar.qr(data)

    partitions = make_partitions(name, inds, which)
    decomp     = HiddenDecomp(fun, partitions)
    
    return Args(x).pluginto(decomp) 
       



@ad.differentiable
def lq(x, name, inds, which="left"):

    def fun(data):
        return ar.lq(data)

    partitions = make_partitions(name, inds, which)
    decomp     = HiddenDecomp(fun, partitions)
    
    return Args(x).pluginto(decomp) 




