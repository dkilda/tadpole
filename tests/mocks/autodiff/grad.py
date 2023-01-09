#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential operator ------------------------------------------------- #

class MockDiffOp:

   def __init__(self, evaluate=NULL, grad=NULL):

       self._evaluate = evaluate
       self._grad     = grad


   @mockify
   def evaluate(self):

       return self._evaluate


   @mockify
   def grad(self, seed):

       return self._grad


   @mockify
   def evaluate_and_grad(self, seed):

       return self._evaluate, self._grad




###############################################################################
###                                                                         ###
###  Backpropagation through the computation graph.                         ###
###                                                                         ###
###############################################################################


# --- Child-node counter ---------------------------------------------------- #

class MockChildCount:

   def __init__(self, compute=NULL, toposort=NULL):

       self._compute  = compute
       self._toposort = toposort


   @mockify
   def add(self, node, parents):

       return self  


   @mockify
   def compute(self):

       return self._compute


   @mockify
   def toposort(self):

       return self._toposort




# --- Topological sort ------------------------------------------------------ #

class MockTopoSort:

   def __init__(self, iterate=NULL):

       self._iterate = iterate


   @mockify
   def add(self, node):

       return self


   @mockify
   def iterate(self):

       return self._iterate 




# --- Gradient accumulation ------------------------------------------------- #

class MockGradAccum:

   def __init__(self, pop=NULL, result=NULL):

       self._pop    = pop
       self._result = result


   @mockify
   def result(self):

       return self._result


   @mockify
   def push(self, node, grad):

       return self


   @mockify
   def pop(self, node):

       return self._pop[node]


   @mockify
   def accumulate(self, node, grad):

       return self




# --- Backpropagation ------------------------------------------------------- # 

class MockBackprop:

   def __init__(self, call): 

       self._call = call


   @mockify
   def __call__(self, seed):

       return self._call















































