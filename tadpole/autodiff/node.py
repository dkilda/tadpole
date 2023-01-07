#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc



###############################################################################
###                                                                         ###
###  Common code for nodes and gates                                        ###
###                                                                         ###
###############################################################################


# --- Forward propagation interface ----------------------------------------- #

class Forward(abc.ABC):

   @abc.abstractmethod
   def grad(self):
       pass




# --- Reverse propagation interface ----------------------------------------- #

class Reverse(abc.ABC):

   @abc.abstractmethod
   def increment_parent_grads(self, grads):
       pass

   @abc.abstractmethod
   def add_to_childcount(self, childcount):
       pass

   @abc.abstractmethod
   def add_to_toposort(self, toposort):
       pass




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #










###############################################################################
###                                                                         ###
###  Gates of the autodiff circuit                                          ###
###                                                                         ###
###############################################################################


# --- Gate ------------------------------------------------------------------ #

class Gate(abc.ABC):

   @abc.abstractmethod
   def node(self, x, layer):
       pass

   @abc.abstractmethod
   def next_input(self, others, adxs, args, source):
       pass




# --- Forward gate ---------------------------------------------------------- #

class ForwardGate(Gate, Forward): 

   def __init__(self, parents, grad):

       self._parents = parents
       self._grad    = grad


   def node(self, x, layer):

       return ForwardNode(x, self, layer)


   def next_input(self, others, adxs, args, source):

       return ForwardGateInputs((self, *others), adxs, args, source)


   def grad(self):

       return self._grad




# --- Reverse gate ---------------------------------------------------------- #

class ReverseGate(Gate, Reverse): 

   def __init__(self, parents, vjp): 

       self._parents = parents
       self._vjp     = vjp


   def node(self, x, layer):

       return ReverseNode(x, self, layer)


   def next_input(self, others, adxs, args, source):

       return ReverseGateInputs((self, *others), adxs, args, source)


   def increment_parent_grads(self, grads):

       parent_grads = self._vjp(grads.pop(self))

       for p, parent in enumerate(self._parents): 
           grads.increment(parent, parent_grads[p])

       return self


   def add_to_childcount(self, childcount):

       childcount.add(self, self._parents)
       return self


   def add_to_toposort(self, toposort):

       for parent in self._parents:
           toposort.add(parent)

       return self




# --- Create the initial gates ---------------------------------------------- #

def make_forward_gate(seed):
    return ForwardGate(tuple(), seed)


def make_reverse_gate():
    return ReverseGate(tuple(), lambda g: ())




# --- Gate inputs ----------------------------------------------------------- #

class GateInputs(abc.ABC):

   @abc.abstractmethod
   def transform(self, fun):
       pass




# --- Forward gate inputs --------------------------------------------------- #

class ForwardGateInputs(GateInputs):

   def __init__(self, gates, adxs, args, out): 

       self._gates = gates
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def transform(self, fun):

       parents = tuple(self._gates[adx] for adx in self._adxs)

       jvp = JvpFactory(fun).jvp(
                                 (p.grad() for p in parents)
                                 self._adxs, 
                                 self._out, 
                                 *self._args
                                )

       return ForwardGate(parents, jvp)




# --- Reverse gate inputs --------------------------------------------------- #

class ReverseGateInputs(GateInputs):

   def __init__(self, gates, adxs, args, out): 

       self._gates = gates
       self._adxs  = adxs
       self._args  = args
       self._out   = out


   def transform(self, fun):

       parents = tuple(self._gates[adx] for adx in self._adxs)

       vjp = VjpFactory(fun).vjp(self._adxs, self._out, *self._args)

       return ReverseGate(parents, vjp)




# --- Create gate inputs ---------------------------------------------------- #

def make_gate_inputs(gates, adxs, args, out):

    return gates[0].next_input(gates[1:], adxs, args, out)























































































































