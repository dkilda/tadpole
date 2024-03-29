#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn

import tadpole.autodiff.node as an
import tadpole.container     as tc




###############################################################################
###                                                                         ###
###  Node for Tensor objects                                                ###
###  (implements Node, Tensor, Grad interfaces)                             ###
###                                                                         ###
###############################################################################


# --- Node for Tensor objects ----------------------------------------------- #

class NodeTensor(an.NodeGen, tn.Tensor, tn.Grad): 

   # --- Gradient operations --- #

   def addto(self, other):

       if not other and not isinstance(other, tn.NullGrad):
          other = tn.NullGrad(self.space())

       return tn.addgrads(self, other)

 
   def todense(self):

       return tn.todense(self)

 
   def tonull(self):

       return tn.tonull(self)


   # --- Tensor methods: basic functionality --- #

   def copy(self, **opts):

       return tn.copy(self, **opts)


   def withdata(self, data):

       return tn.withdata(self, data)


   def space(self):
 
       return tn.space(self)


   def item(self, *pos):

       return tn.item(self, *pos)


   # --- Tensor methods: tensor properties --- #

   @property
   def dtype(self):
       return tn.dtype(self)  

   @property 
   def size(self):
       return tn.size(self) 

   @property 
   def ndim(self):
       return tn.ndim(self) 

   @property
   def shape(self):
       return tn.shape(self) 


   # --- Tensor methods: tensor manipulation --- # 

   def __call__(self, *inds):

       return tn.reindexto(self, *inds) 


   @property
   def C(self):
       return tn.conj(self)

   @property
   def T(self):
       return tn.transpose(self)

   @property
   def H(self):
       return tn.htranspose(self)


   # --- Tensor methods: element access --- # 

   def __getitem__(self, idx):

       return tn.getitem(self, idx) 


   # --- Tensor methods: arithmetics --- # 

   def __neg__(self):

       return tn.neg(self)


   def __add__(self, other):

       return tn.add(self, other)


   def __sub__(self, other):

       return tn.sub(self, other)


   def __mul__(self, other):

       return tn.mul(self, other)


   def __truediv__(self, other):

       return tn.div(self, other)


   def __floordiv__(self, other):

       return tn.floordiv(self, other)


   def __mod__(self, other):

       return tn.mod(self, other)


   def __pow__(self, other):

       return tn.power(self, other)


   def __matmul__(self, other):

       return tn.contract(self, other)


   # --- Tensor methods: reflected arithmetics --- # 

   def __radd__(self, other):

       return tn.add(other, self)

 
   def __rsub__(self, other):

       return tn.sub(other, self)


   def __rmul__(self, other):

       return tn.mul(other, self)


   def __rtruediv__(self, other):

       return tn.div(other, self)


   def __rfloordiv__(self, other):

       return tn.floordiv(other, self)


   def __rmod__(self, other):

       return tn.mod(other, self)


   def __rpow__(self, other):

       return tn.power(other, self)


   def __rmatmul__(self, other):

       return tn.contract(other, self)




# --- Register NodeTensor with the types it can wrap ------------------------ #

an.register(tn.TensorGen,  NodeTensor)
an.register(tn.SparseGrad, NodeTensor)
an.register(tn.NullGrad,   NodeTensor)




###############################################################################
###                                                                         ###
###  Node for Container objects                                             ###
###  (implements Node, Container, Grad interfaces)                          ###
###                                                                         ###
###############################################################################


# --- Node for Container objects -------------------------------------------- #

class NodeContainer(an.NodeGen, tc.Container, tn.Grad):

   # --- Gradient operations --- #

   def addto(self, other):

       if not other and not isinstance(other, tc.NullGrad):
          other = tc.NullGrad(len(self))

       return tc.addgrads(self, other)

 
   def todense(self):

       return tc.todense(self)

 
   def tonull(self):

       return tc.tonull(self)


   # --- Container methods --- #

   def copy(self, **opts):

       return tc.copy(self, **opts) 


   def withdata(self, data):

       return tc.withdata(self, data) 


   def space(self):

       return tc.space(self) 


   def item(self, pos):

       return tc.item(self, pos)


   def __len__(self):

       return tc.size(self) 


   def __contains__(self, x):
       
       return tc.contains(self, x)


   def __iter__(self):

       return tc.iterate(self)


   def __getitem__(self, idx):
       
       return tc.getitem(self, idx)


 

# --- Register NodeContainer with the types it can wrap --------------------- #

an.register(tc.ContainerGen, NodeContainer) 
an.register(tc.SparseGrad,   NodeContainer)
an.register(tc.NullGrad,     NodeContainer)








