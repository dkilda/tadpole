#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad





class Alignment(abc.ABC):

   @abc.abstractmethod
   def left(self, inds):
       pass

   @abc.abstractmethod
   def right(self, inds):
       pass




class LeftAlignment(Alignment):

   def __init__(self, partinds):

       self._partinds = partinds


   def linds(self, inds):

       return self._partinds
       

   def rinds(self, inds):

       return inds.remove(*self._partinds) 




class RightAlignment(Alignment):

   def __init__(self, partinds):

       self._partinds = partinds


   def linds(self, inds):

       return inds.remove(*self._partinds)
       

   def rinds(self, inds):

       return self._partinds




class Link:

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




class Partitions:

   def __init__(self, alignment, link):

       self._alignment = alignment
       self._link      = link


   def create(self, inds):

       return Partition(
          inds, self._alignment.linds(), self._alignment.rinds(), self._link
       )
         



 




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



       



def execute(fun, *xs): # TODO move to funcall.py

    if not isinstance(fun, EngineLike):
       fun = Engine(engine)

    return Args(*xs).pluginto(fun)




def make_alignment(inds, which):

    return {
            "left":  LeftAlignment,
            "right": RightAlignment,
           }[which](inds)



def make_partitions(inds, which, name):

    link      = Link(name)
    alignment = {
                 "left":  LeftAlignment,
                 "right": RightAlignment,
                }[which](inds)

    return Partitions(alignment, link)



    



def explicit_decomp(fun, inds, which, name, trunc):

    partitions = make_partitions(inds, which, name)
    decomp     = ExplicitDecomp(fun, partitions, trunc)

    return funcall.execute(decomp)




def hidden_decomp(fun, inds, which, name):

    partitions = make_partitions(inds, which, name)
    decomp     = HiddenDecomp(fun, partitions)

    return funcall.execute(decomp)






# --- Linear algebra: decomposition methods --------------------------------- #

@ad.differentiable
def svd(x, sname, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.svd(data)

    indpart = indpartition(sname, inds, which)
    decomp  = ExplicitDecomp(fun, indpart, trunc)

    return fn.Args(x).pluginto(decomp)



@ad.differentiable
def eig(x, sname, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eig(data)

    indpart = indpartition(sname, inds, which)
    decomp  = ExplicitDecomp(fun, indpart, trunc)

    return fn.Args(x).pluginto(decomp)



@ad.differentiable
def eigh(x, sname, inds, which="left", trunc=NullTrunc()):

    def fun(data):
        return ar.eigh(data)

    indpart = indpartition(sname, inds, which)
    decomp  = ExplicitDecomp(fun, indpart, trunc)

    return fn.Args(x).pluginto(decomp)
       


@ad.differentiable
def qr(x, sname, inds, which="left"):

    def fun(data):
        return ar.qr(data)

    indpart = indpartition(sname, inds, which)
    decomp  = HiddenDecomp(fun, indpart)

    return fn.Args(x).pluginto(decomp)
       


@ad.differentiable
def lq(x, sname, inds, which="left"):

    def fun(data):
        return ar.lq(data)

    indpart = indpartition(sname, inds, which)
    decomp  = HiddenDecomp(fun, indpart)

    return fn.Args(x).pluginto(decomp)





















































