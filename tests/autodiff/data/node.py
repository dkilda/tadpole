#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad
import tadpole.autodiff.misc  as misc




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint function data ------------------------------------------------- #

AdjFunData = collections.namedtuple("AdjFunData", [
                 "adjfun", "grads", "seed",
             ])




def forward_adjfun_dat(valency=1):

    grads  = arepeat(fake.Value, valency)
    seed   = arepeat(fake.Value, valency)
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




def reverse_adjfun_dat(valency=1):

    grads  = arepeat(fake.Value, valency)
    seed   = fake.Value()
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




# --- Adjoint data ---------------------------------------------------------- #

AdjointOpData = collections.namedtuple("AdjointOpData", [
                   "op", 
                   "fun", "adxs", "out", "args", 
                   "vjpfun", "vjpgrads", "vjpseed",
                   "jvpfun", "jvpgrads", "jvpseed",
                ])




def adjointop_dat(nargs=1, valency=1, adxs=(0,)):  

    vjpdat = reverse_adjfun_dat(valency)
    jvpdat = forward_adjfun_dat(valency)

    out  = fake.Node()
    args = arepeat(fake.Node, nargs)

    fun = fake.DifferentiableFun(
             call=fake.Fun(out, args), 
             vjp=fake.Fun(vjpdat.adjfun, adxs, out, *args),
             jvp=fake.Fun(jvpdat.adjfun, adxs, out, *args),
          )

    op = an.AdjointOpGen(fun, adxs, out, args)

    return AdjointOpData(
              op, 
              fun, adxs, out, args, 
              vjpdat.adjfun, vjpdat.grads, vjpdat.seed,
              jvpdat.adjfun, jvpdat.grads, jvpdat.seed,
           )




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- Flow data ------------------------------------------------------------- #

FlowData = collections.namedtuple("FlowData", [
              "flow", "name", "fun",
           ])




def null_flow_dat():

    def fun(parents, op):
        return an.GateNull()

    name = "NULL"
    flow = an.FlowGen(name, fun)  
    
    return FlowData(flow, name, fun)




def forward_flow_dat():

    def fun(parents, op):
        return an.GateForward(parents, op)

    name = "FORWARD"
    flow = an.FlowGen(name, fun)  
    
    return FlowData(flow, name, fun)




def reverse_flow_dat():

    def fun(parents, op):
        return an.GateReverse(parents, op)

    name = "REVERSE"
    flow = an.FlowGen(name, fun)  
    
    return FlowData(flow, name, fun)




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Gate data ------------------------------------------------------------- #

GateData = collections.namedtuple("GateData", [
              "gate", "parents", "op", "grads", "seed",
           ])




def forward_gate_dat(valency=1):

    adjfun = forward_adjfun_dat(valency)
    op     = fake.AdjointOp(jvp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = an.ParentsGen(*arepeat(fake.Node, valency))
    gate    = an.GateForward(parents, op) 

    return GateData(gate, parents, op, adjfun.grads, adjfun.seed)




def reverse_gate_dat(valency=1):

    adjfun = reverse_adjfun_dat(valency)
    op     = fake.AdjointOp(vjp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = an.ParentsGen(*arepeat(fake.Node, valency))
    gate    = an.GateReverse(parents, op) 

    return GateData(gate, parents, op, adjfun.grads, adjfun.seed)




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node data ------------------------------------------------------------- #

NodeData = collections.namedtuple("NodeData", [
              "node", "source", "layer", "gate", "value"
           ])




def node_dat(layer=0, gate=fake.Gate()):

    value  = fake.Value()
    source = fake.Node()
    node   = an.node(source, layer, gate)

    return NodeData(node, source, layer, gate, value)




# --- Directed node data ---------------------------------------------------- #

DirectedNodeData = collections.namedtuple("DirectedNodeData", [
                       "node", "source",  "layer", "value", 
                       "gate", "parents", "grads", "seed",
                    ])




def forward_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = arepeat(fake.Node, parent_nodes)

    if grads is None or seed is None:
       adjfun = forward_adjfun_dat(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.AdjointOp(jvp=fake.Fun(grads, seed))
    parents = an.ParentsGen(*parent_nodes)
    gate    = an.GateForward(parents, op)
    nodedat = node_dat(layer, gate)

    return DirectedNodeData(
                            nodedat.node,  nodedat.source, 
                            nodedat.layer, nodedat.value,
                            gate, parents, grads, seed
                           )




def reverse_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = arepeat(fake.Node, parent_nodes)

    if grads is None or seed is None:
       adjfun = reverse_adjfun_dat(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.AdjointOp(vjp=fake.Fun(grads, seed))
    parents = an.ParentsGen(*parent_nodes)
    gate    = an.GateReverse(parents, op)
    nodedat = node_dat(layer, gate)
    
    return DirectedNodeData(
                            nodedat.node,  nodedat.source, 
                            nodedat.layer, nodedat.value,
                            gate, parents, grads, seed
                           )




# --- Point data ------------------------------------------------------------ #

PointData = collections.namedtuple("PointData", [
              "point", "source", "layer", "gate"
            ])




def point_dat():

    source = fake.Value() 
    layer  = misc.minlayer()
    gate   = an.GateNull()

    point = an.NodeGen(source, layer, gate) 

    return PointData(point, source, layer, gate)




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- Parent data ----------------------------------------------------------- #

ParentData = collections.namedtuple("ParentData", [
                "parents", "pnodes",
             ])




def forward_parents_dat(valency=1):

    def pnode():

        flow = an.FlowGen("FORWARD", 
           lambda parents, op: an.GateForward(parents, op))

        return fake.Node(flow=fake.Fun(flow))

    pnodes  = arepeat(pnode, valency)
    parents = an.ParentsGen(*pnodes)

    return ParentData(parents, pnodes)




def reverse_parents_dat(valency=1):

    def pnode():

        flow = an.FlowGen("REVERSE", 
           lambda parents, op: an.GateReverse(parents, op))

        return fake.Node(flow=fake.Fun(flow))

    pnodes  = arepeat(pnode, valency)
    parents = an.ParentsGen(*pnodes)

    return ParentData(parents, pnodes)






