#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
from tadpole.util import Container




###############################################################################
###                                                                         ###
###  Nary                                                                   ###
###                                                                         ###
###############################################################################


# --- Argument proxy type --------------------------------------------------- #

class ArgProxy(abc.ABC):

   @abc.abstractmethod
   def insert(self, args, x):
       pass

   @abc.abstractmethod
   def extract(self, args):
       pass




###############################################################################
###                                                                         ###
###  Node                                                                   ###
###                                                                         ###
###############################################################################


# --- Adjoint operator ------------------------------------------------------ #

class AdjointOp(abc.ABC):

   @abc.abstractmethod
   def vjp(self, seed):
       pass

   @abc.abstractmethod
   def jvp(self, seed):
       pass




# --- Flow ------------------------------------------------------------------ #

class Flow(abc.ABC):

   @abc.abstractmethod
   def __eq__(self, other):
       pass

   @abc.abstractmethod
   def __hash__(self):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __radd__(self, other):
       pass

   @abc.abstractmethod
   def gate(self, parents, op):
       pass




# --- Gate ------------------------------------------------------------------ #

class Gate(abc.ABC):

   @abc.abstractmethod
   def flow(self):
       pass

   @abc.abstractmethod
   def trace(self, node, traceable):
       pass

   @abc.abstractmethod
   def grads(self, node, grads):
       pass




# --- Node ------------------------------------------------------------------ #

class Node(abc.ABC):

   @abc.abstractmethod
   def connected(self):
       pass

   @abc.abstractmethod
   def flow(self):
       pass

   @abc.abstractmethod
   def concat(self, concatenable):
       pass

   @abc.abstractmethod
   def trace(self, traceable):
       pass

   @abc.abstractmethod
   def grads(self, grads):
       pass




# --- Parents --------------------------------------------------------------- #

class Parents(Container, abc.ABC):

   @abc.abstractmethod
   def next(self, source, layer, op):
       pass




###############################################################################
###                                                                         ###
###  Graph                                                                  ###
###                                                                         ###
###############################################################################


# --- Differentiable function type ------------------------------------------ #

class DifferentiableFun(abc.ABC):

   @abc.abstractmethod
   def __call__(self, *args, **kwargs):
       pass

   @abc.abstractmethod
   def vjp(self, *args, **kwargs):
       pass

   @abc.abstractmethod
   def jvp(self, *args, **kwargs):
       pass




# --- Function arguments ---------------------------------------------------- #

class Args(Container, abc.ABC):

   """
   @abc.abstractmethod
   def nodify(self):
       pass
   """

   @abc.abstractmethod
   def concat(self):
       pass

   @abc.abstractmethod
   def pack(self, **kwargs):
       pass

   @abc.abstractmethod
   def deshelled(self):
       pass




# --- Sequential type ------------------------------------------------------- #

class Sequential(abc.ABC):

   @abc.abstractmethod
   def attach(self, node, source, layer):
       pass




# --- Cohesive type --------------------------------------------------------- #

class Cohesive(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

   @abc.abstractmethod
   def layer(self):
       pass

   @abc.abstractmethod
   def adxs(self):
       pass

   @abc.abstractmethod
   def parents(self):
       pass

   @abc.abstractmethod
   def deshell(self):
       pass




# --- Pack type ------------------------------------------------------------- #

class Pack(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

   @abc.abstractmethod
   def deshell(self):
       pass

   @abc.abstractmethod
   def deshelled(self):
       pass

   @abc.abstractmethod
   def fold(self, funwrap, out):
       pass


       

# --- Envelope type --------------------------------------------------------- #

class Envelope(abc.ABC): 

   @abc.abstractmethod
   def packs(self):
       pass

   @abc.abstractmethod
   def apply(self, fun):
       pass

   @abc.abstractmethod
   def applywrap(self, funwrap, out):
       pass




###############################################################################
###                                                                         ###
###  Grad                                                                   ###
###                                                                         ###
###############################################################################


# --- Gradient propagation -------------------------------------------------- #

class Propagation(abc.ABC):

   @abc.abstractmethod
   def graphop(self, fun, x):
       pass

   @abc.abstractmethod
   def accum(self, end, seed):
       pass




# --- Traceable type -------------------------------------------------------- #

class Traceable(abc.ABC):

   @abc.abstractmethod
   def record(self, node, parents):
       pass




# --- Countable type -------------------------------------------------------- #

class Countable(abc.ABC):

   @abc.abstractmethod
   def collect(self, node):
       pass

   @abc.abstractmethod
   def increase(self, node):
       pass

   @abc.abstractmethod
   def decrease(self, node):
       pass




# --- Cumulative type ------------------------------------------------------- #

class Cumulative(abc.ABC):

   @abc.abstractmethod
   def add(self, nodes, parents):
       pass

   @abc.abstractmethod
   def pick(self, nodes):
       pass

   @abc.abstractmethod
   def result(self):
       pass




