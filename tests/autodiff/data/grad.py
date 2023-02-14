#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad




"""
       w = {
            "REVERSE": data.reverse_gate, 
            "FORWARD": data.forward_gate,
           }[which](valency)

       x = data.node(gate=w.gate)

       count  = tdgrad.ChildCount()  
       count1 = tdgrad.ChildCount({x.node: w.parents})
"""



"""
       w = data.reverse_gate(valency)
       x = data.node(gate=w.gate)

       grads  = tdgrad.GradAccum({x.node: w.seed})
       grads1 = tdgrad.GradAccum({
                                  None: w.seed, 
                                  **dict(zip(w.parents, w.grads)),
                                 })
"""



"""
       w = data.forward_gate(valency)
       x = data.node(gate=w.gate)

       init_seed = fake.Value()

       grads  = tdgrad.GradSum(init_seed, dict(zip(w.parents, w.seed))) 
       grads1 = tdgrad.GradSum(init_seed, {x.node: sum(w.grads)})
"""




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


DifferentialOpData = collections.namedtuple("DifferentialOpData", [
                        "op", "graph", "root", "fun", "x"
                     ])



# --- Forward differential operator ----------------------------------------- #

def forward_differential():




# --- Reverse differential operator ----------------------------------------- #





# --- Graph operator -------------------------------------------------------- #

GraphOpData = collections.namedtuple("GraphOpData", [
                 "op", "root", "fun", 
                 "x", "out", 
                 "start", "end",
              ])




def graphop(which="REVERSE", x=None, end=None):

    if x is None:
       x = fake.Value()

    if out is None:
       out = fake.Value()

    if start is None:
       start = fake.NodeLike(tovalue=fake.Fun(x))

    if end is None:
       end = fake.NodeLike(tovalue=fake.Fun(out))

    fun  = fake.Fun(end, start)
    root = {
            "REVERSE": tdnode.ReverseGate,
            "FORWARD": tdnode.ForwardGate,
           }[which]()

    graphop = GraphOp(root, fun, x)

    return GraphOpData(graphop, root, fun, x, out, start, end)   










###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Child count ----------------------------------------------------------- #



# --- Traversal ------------------------------------------------------------- #



# --- Topological sort ------------------------------------------------------ #




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Gradient summation ---------------------------------------------------- #





# --- Gradient accumulation ------------------------------------------------- #


























