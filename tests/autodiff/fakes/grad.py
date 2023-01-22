#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph

from tests.common.fakes import NULL, fakeit





class ChildCount:

   def __init__(self, iterate=NULL, toposort=NULL):

       self._visited  = []       # Sentinels which show that a void method 
       self._added    = []       # has been called with a correct input 
       self._computed = False    # (e.g. .visit() sets ._visited)

       self._iterate  = iterate  # Fake return values for non-void methods
       self._toposort = toposort # (e.g. .iterate() returns ._iterated)


   def visited(self, idx=-1):
 
       return self._visited[idx]


   def added(self, idx=-1):

       return self._added[idx]


   def computed(self):

       return self._computed()


   @fakeit
   def visit(self, node):

       self._visited.append(node) 
       return self


   @fakeit
   def add(self, nodes):
 
       self._added.append(nodes)
       return self  


   @fakeit
   def compute(self):

       self._computed = True
       return self


   @fakeit
   def iterate(self):

       return self._iterate


   @fakeit
   def toposort(self):

       return self._toposort




class TopoSort:

   def __init__(self, iterate=NULL):

       self._added   = []
       self._iterate = iterate


   def added(self, idx=-1):

       return self._added[idx]


   @fakeit
   def add(self, node):

       self._added.append(node)
       return self


   @fakeit
   def iterate(self):

       return self._iterate 





class GradAccum:

   def __init__(self, pop=NULL, result=NULL):

       self._pushed      = {}
       self._accumulated = {}

       self._pop    = pop
       self._result = result


   def pushed(self, node):

       return self._pushed[node]


   def accumulated(self, node):

       return self._accumulated[node]


   @fakeit
   def result(self):

       return self._result


   @fakeit
   def push(self, node, grad):

       self._pushed[node] = grad
       return self 


   @fakeit
   def pop(self, node):

       return self._pop[node]


   @fakeit
   def accumulate(self, node, grad):

       self._accumulated[node] = grad
       return self

























































