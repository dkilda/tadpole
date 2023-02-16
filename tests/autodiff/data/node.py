#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph




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




def forward_adjfun(valency=1):

    grads  = common.arepeat(fake.Value, valency)
    seed   = common.arepeat(fake.Value, valency)
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




def reverse_adjfun(valency=1):

    grads  = common.arepeat(fake.Value, valency)
    seed   = fake.Value()
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




# --- Adjoint data ---------------------------------------------------------- #

AdjointData = collections.namedtuple("AdjointData", [
                 "op", "fun", "adxs", "out", "args", 
              ])




def adjoint(nargs=1, adxs=(0,)):  

    out  = fake.NodeLike()
    args = common.arepeat(fake.NodeLike, nargs)

    fun = fake.Fun(out, args)
    op  = tdnode.AdjointOp(fun, adxs, out, args)

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




def null_flow():

    def fun(parents, op):
        return tdnode.NullGate()

    name = "NULL"
    flow = tdnode.Flow(name, fun)  
    
    return FlowData(flow, name, fun)




def forward_flow():

    def fun(parents, op):
        return tdnode.ForwardGate(parents, op)

    name = "FORWARD"
    flow = tdnode.Flow(name, fun)  
    
    return FlowData(flow, name, fun)




def reverse_flow():

    def fun(parents, op):
        return tdnode.ReverseGate(parents, op)

    name = "REVERSE"
    flow = tdnode.Flow(name, fun)  
    
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




def forward_gate(valency=1):

    adjfun = forward_adjfun(valency)
    op     = fake.Adjoint(jvp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = tdnode.Parents(common.arepeat(fake.NodeLike, valency))
    gate    = tdnode.ForwardGate(parents, op) 

    return GateData(gate, parents, op, adjfun.grads, adjfun.seed)




def reverse_gate(valency=1):

    adjfun = reverse_adjfun(valency)
    op     = fake.Adjoint(vjp=fake.Fun(adjfun.grads, adjfun.seed))

    parents = tdnode.Parents(common.arepeat(fake.NodeLike, valency))
    gate    = tdnode.ReverseGate(parents, op) 

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




def node(layer=0, gate=fake.GateLike()):

    value  = fake.Value()
    source = fake.NodeLike(tovalue=fake.Fun(value))

    node = tdnode.Node(source, layer, gate)

    return NodeData(node, source, layer, gate, value)




# --- Directed node data ---------------------------------------------------- #

DirectedNodeData = collections.namedtuple("DirectedNodeData", [
                       "node", "source",  "layer", "value", 
                       "gate", "parents", "grads", "seed",
                    ])




def forward_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = common.arepeat(fake.NodeLike, parent_nodes)

    if grads is None or seed is None:
       adjfun = forward_adjfun(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.Adjoint(jvp=fake.Fun(grads, seed))
    parents = tdnode.Parents(parent_nodes)
    gate    = tdnode.ForwardGate(parents, op)
    node_dat = node(layer, gate)

    return DirectedNodeData(
                            node_dat.node, node_dat.source, 
                            node_dat.layer, node_dat.value,
                            gate, parents, grads, seed
                           )




def reverse_node_dat(parent_nodes=1, grads=None, seed=None, layer=None):

    if isinstance(parent_nodes, int):
       parent_nodes = common.arepeat(fake.NodeLike, parent_nodes)

    if grads is None or seed is None:
       adjfun = reverse_adjfun(len(parent_nodes))

    if grads is None:
       grads = adjfun.grads 

    if seed is None:
       seed = adjfun.seed

    if layer is None:
       layer = 0

    op      = fake.Adjoint(vjp=fake.Fun(grads, seed))
    parents = tdnode.Parents(parent_nodes)
    gate    = tdnode.ReverseGate(parents, op)
    node_dat = node(layer, gate)
    
    return DirectedNodeData(
                            node_dat.node, node_dat.source, 
                            node_dat.layer, node_dat.value,
                            gate, parents, grads, seed
                           )




# --- Point data ------------------------------------------------------------ #

PointData = collections.namedtuple("PointData", [
              "point", "source", "layer", "gate"
            ])




def point():

    source = fake.Value() 
    layer  = tdgraph.minlayer()
    gate   = tdnode.NullGate()

    point = tdnode.Point(source)

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




def forward_parents(valency=1):

    def pnode():

        flow = tdnode.Flow("FORWARD", 
           lambda parents, op: tdnode.ForwardGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = common.arepeat(pnode, valency)
    parents = tdnode.Parents(pnodes)

    return ParentData(parents, pnodes)




def reverse_parents(valency=1):

    def pnode():

        flow = tdnode.Flow("REVERSE", 
           lambda parents, op: tdnode.ReverseGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = common.arepeat(pnode, valency)
    parents = tdnode.Parents(pnodes)

    return ParentData(parents, pnodes)






