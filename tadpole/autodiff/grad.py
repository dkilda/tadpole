#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.adjoint_factory as adj



###############################################################################
###                                                                         ###
###  Backpropagation through the computation graph.                         ###
###                                                                         ###
###############################################################################


# --- Child-node counter ---------------------------------------------------- #

class ChildCount:

   def __init__(self, last_node):

       self._last_node = last_node
       self._count     = {}
       self._pool      = None


   def add(self, node, parents):

       try:
           self._count[node] += 1
       except KeyError:
           self._count[node] = 1

       if self._count.get(node) == 1:
          self._pool.extend(parents)

       return self  


   def compute(self):

       self._pool = [self._last_node]

       while self._pool:

          node = self._pool.pop()
          node.add_to_childcount(self)

       return self


   def toposort(self):

       return TopoSort(self._count, self._last_node)

       


# --- Topological sort ------------------------------------------------------ #

class TopoSort:

   def __init__(self, count, last_node):

       self._last_node = last_node
       self._count     = dict(count)
       self._pool      = None


   def add(self, node):

       self._count[parent] -= 1 

       if self._count[parent] == 0:
          self._pool.append(parent) 

       return self


   def iterate(self):

       self._pool = [self._last_node]

       while self._pool:

          node = self._pool.pop()
          yield node

          node.add_to_toposort(self)




# --- Create a topologically sorted iterator over the computation graph ----- #

def toposort(last_node):
    return ChildCount(last_node).compute()
                                .toposort()
                                .iterate()




# --- Add gradients --------------------------------------------------------- #

def add_grads(net_g, g): # TODO impl and use add() function, with @diffable decorator 
                         #      (or overload __add__ operator to make it @diffable)
    if net_g is None:  
       return g

    return adj.add(net_g, g)




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum:

   def __init__(self):

       self._map  = {}
       self._last = None


   def result(self):

       return self._last


   def push(self, node, grad):

       self._map[node] = grad
       return self


   def pop(self, node):

       self._last = self._map.pop(node)
       return self._last


   def accumulate(self, node, grad):

       self._map[node] = add_grads(self._map.get(node), grad)
       return self




# --- Backpropagation ------------------------------------------------------- # 

class Backprop:

   def __init__(self, last_node): 

       self._last_node = last_node


   def __call__(self, seed):

       grads = GradAccum()  
       grads.push(self._last_node, seed)

       for node in toposort(self._last_node): 
           node.accumulate_parent_grads(grads)

       return grads.result()











































































