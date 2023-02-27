#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake

import tadpole.autodiff.node  as anode
import tadpole.autodiff.graph as agraph




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

AdjointData = collections.namedtuple("AdjointData", [
                 "op", "fun", "adxs", "out", "args", 
              ])




def adjoint_dat(nargs=1, adxs=(0,)):  

    out  = fake.NodeLike()
    args = arepeat(fake.NodeLike, nargs)

    fun = fake.Fun(out, args)
    op  = anode.AdjointOp(fun, adxs, out, args)

    return AdjointData(
              op, 
              fun, adxs, out, args, 
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
        return anode.NullGate()

    name = "NULL"
    flow = anode.Flow(name, fun)  
    
    return FlowData(flow, name, fun)




def forward_flow_dat():

    def fun(parents, op):
        return anode.ForwardGate(parents, op)

    name = "FORWARD"
    flow = anode.Flow(name, fun)  
    
    return FlowData(flow, name, fun)




def reverse_flow_dat():

    def fun(parents, op):
        return anode.ReverseGate(parents, op)

    name = "REVERSE"
    flow = anode.Flow(name, fun)  
    
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
    op     = fake.Adjoint(jvp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = anode.Parents(arepeat(fake.NodeLike, valency))
    gate    = anode.ForwardGate(parents, op) 

    return GateData(gate, parents, op, adjfun.grads, adjfun.seed)




def reverse_gate_dat(valency=1):

    adjfun = reverse_adjfun_dat(valency)
    op     = fake.Adjoint(vjp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = anode.Parents(arepeat(fake.NodeLike, valency))
    gate    = anode.ReverseGate(parents, op) 

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




def node_dat(layer=0, gate=fake.GateLike()):

    value  = fake.Value()
    source = fake.NodeLike()
    node   = anode.Node(source, layer, gate)

    return NodeData(node, source, layer, gate, value)




# --- Directed node data ---------------------------------------------------- #

DirectedNodeData = collections.namedtuple("DirectedNodeData", [
                       "node", "source",  "layer", "value", 
                       "gate", "parents", "grads", "seed",
                    ])




def forward_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = arepeat(fake.NodeLike, parent_nodes)

    if grads is None or seed is None:
       adjfun = forward_adjfun_dat(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.Adjoint(jvp=fake.Fun(grads, seed))
    parents = anode.Parents(parent_nodes)
    gate    = anode.ForwardGate(parents, op)
    nodedat = node_dat(layer, gate)

    return DirectedNodeData(
                            nodedat.node,  nodedat.source, 
                            nodedat.layer, nodedat.value,
                            gate, parents, grads, seed
                           )




def reverse_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = arepeat(fake.NodeLike, parent_nodes)

    if grads is None or seed is None:
       adjfun = reverse_adjfun_dat(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.Adjoint(vjp=fake.Fun(grads, seed))
    parents = anode.Parents(parent_nodes)
    gate    = anode.ReverseGate(parents, op)
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
    layer  = agraph.minlayer()
    gate   = anode.NullGate()

    point = anode.Point(source)

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

        flow = anode.Flow("FORWARD", 
           lambda parents, op: anode.ForwardGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = arepeat(pnode, valency)
    parents = anode.Parents(pnodes)

    return ParentData(parents, pnodes)




def reverse_parents_dat(valency=1):

    def pnode():

        flow = anode.Flow("REVERSE", 
           lambda parents, op: anode.ReverseGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = arepeat(pnode, valency)
    parents = anode.Parents(pnodes)

    return ParentData(parents, pnodes)






