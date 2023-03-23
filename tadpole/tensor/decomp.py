#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad




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




# --- PartitionLike's factory interface ------------------------------------- #

class PartitionLikes(abc.ABC):

   @abc.abstractmethod
   def create(self, inds):
       pass




# --- Partition factory ----------------------------------------------------- #

class Partitions(PartitionLikes):

   def __init__(self, alignment, link):

       self._alignment = alignment
       self._link      = link


   def create(self, inds):

       return Partition(
          inds, self._alignment.linds(), self._alignment.rinds(), self._link
       )
         


 
# --- PartitionLike interface ----------------------------------------------- #

class PartitionLike(abc.ABC):

   @abc.abstractmethod
   def aligndata(self, data):
       pass

   @abc.abstractmethod
   def ltensor(self, data):
       pass

   @abc.abstractmethod
   def rtensor(self, data):
       pass

   @abc.abstractmethod
   def stensor(self, data):
       pass




# --- Partition ------------------------------------------------------------- #

class Partition(PartitionLike):

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











"""

class IndexPartition(abc.ABC):

   @abc.abstractmethod
   def left(self, inds):
       pass

   @abc.abstractmethod
   def right(self, inds):
       pass

   @abc.abstractmethod   
   def mid(self, size):
       pass



class LeftIndexPartition(IndexPartition):

   def __init__(self, partinds, midname):

       self._partinds = partinds
       self._midname  = midname


   def left(self, inds): 
 
       return Indices(*self._partinds)
    

   def right(self, inds):

       return inds.remove(*self._partinds) 

  
   def mid(self, size):
           return {
            "left":  LeftIndexPartition,
            "right": RightIndexPartition,
           }[which](sname, inds)
       return Index(self._midname, size)




class RightIndexPartition(IndexPartition):

   def __init__(self, partinds, midname):

       self._partinds = partinds
       self._midname  = midname


   def left(self, inds): 

       return inds.remove(*self._partinds) 


   def right(self, inds): 

       return Indices(*self._partinds)

  
   def mid(self, size):
       
       return Index(self._midname, size)





def indpartition(sname, inds, which):

    return {
            "left":  LeftIndexPartition,
            "right": RightIndexPartition,
           }[which](sname, inds)

"""








###############################################################################
###                                                                         ###
###  Tensor decomposition calls                                             ###
###                                                                         ###
###############################################################################


# --- Explicit-rank decomposition call -------------------------------------- #

class ExplicitDecomp(fn.FunCall):

   def __init__(self, engine, partitions, trunc):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

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





       



"""


   @util.cacheable
   def linds(self):

       inds, = self._engine.inds()
       return self._indpart.left(inds)


   @util.cacheable
   def rinds(self):

       inds, = self._engine.inds()
       return self._indpart.right(inds)


       linds, rinds = self.linds(), self.rinds()

       data = ar.transpose(data, (*inds.axes(*linds), *inds.axes(*rinds)))
       data = ar.reshape(data,   (linds.size, rinds.size))
       
       ldata, sdata, rdata = self._engine.execute(data)



       ldata, sdata, rdata = self._trunc.apply(ldata, sdata, rdata)

       ldata = ar.reshape(ldata, (*linds.shape, -1))
       rdata = ar.reshape(rdata, (-1, *rinds.shape))

       ltensor = core.Tensor(ldata, linds.push(sind))
       stensor = core.Tensor(sdata, sind)
       rtensor = core.Tensor(rdata, rinds.add(sind))

       return ltensor, stensor, rtensor, error
"""



# --- Hidden-rank decomposition call ---------------------------------------- #

class HiddenDecomp(fn.FunCall):

   def __init__(self, engine, partitions):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

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






"""
   @util.cacheable
   def linds(self):

       inds, = self._engine.inds()
       return self._indpart.left(inds)


   @util.cacheable
   def rinds(self):

       inds, = self._engine.inds()
       return self._indpart.right(inds)



       data, = self._engine.datas()
       inds, = self._engine.inds()

       linds, rinds = self.linds(), self.rinds()

       data = ar.transpose(data, (*inds.axes(*linds), *inds.axes(*rinds)))
       data = ar.reshape(data,   (linds.size, rinds.size))
       
       ldata, rdata = self._engine.execute(data)

       ldata = ar.reshape(ldata, (*linds.shape, -1))
       rdata = ar.reshape(rdata, (-1, *rinds.shape))

       sind = self._indpart.mid(ldata.shape[-1])

       ltensor = core.Tensor(ldata, linds.push(sind))
       rtensor = core.Tensor(rdata, rinds.add(sind))

       return ltensor, rtensor 

"""



       

# --- A shorthand execution of any function with TensorLike arguments ------- # 

def execute(fun, *xs): # TODO move to funcall.py

    if not isinstance(fun, EngineLike):
       fun = Engine(engine)

    return Args(*xs).pluginto(fun)



"""
def make_alignment(inds, which):

    return {
            "left":  LeftAlignment,
            "right": RightAlignment,
           }[which](inds)
"""




# --- General methods for explicit-rank and hidden-rank decompositions ------ #

def make_partitions(linkname, inds, which):

    link      = Link(linkname)
    alignment = {
                 "left":  LeftAlignment,
                 "right": RightAlignment,
                }[which](inds)

    return Partitions(alignment, link)




def explicit_decomp(fun, name, inds, which, trunc):

    partitions = make_partitions(name, inds, which)
    decomp     = ExplicitDecomp(fun, partitions, trunc)

    return funcall.execute(decomp)




def hidden_decomp(fun, name, inds, which):

    partitions = make_partitions(name, inds, which)
    decomp     = HiddenDecomp(fun, partitions)

    return funcall.execute(decomp)




# --- Specialized decomposition methods ------------------------------------- #

@ad.differentiable
def svd(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.svd(data)

    return explicit_decomp(fun, name, inds, which, trunc)

 


@ad.differentiable
def eig(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eig(data)

    return explicit_decomp(fun, name, inds, which, trunc)




@ad.differentiable
def eigh(x, name, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eigh(data)

    return explicit_decomp(fun, name, inds, which, trunc) 




@ad.differentiable
def qr(x, name, inds, which="left"):

    def fun(data):
        return ar.qr(data)

    return hidden_decomp(fun, name, inds, which) 
       



@ad.differentiable
def lq(x, name, inds, which="left"):

    def fun(data):
        return ar.lq(data)

    return hidden_decomp(fun, name, inds, which) 




