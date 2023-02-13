#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.util  as tdutil
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad




# --- Parent data ----------------------------------------------------------- #

ParentData = collections.namedtuple("ParentData", [
                "parents", "pnodes",
             ])




def forward_parents(valency):

    def pnode():

        flow = tdnode.Flow("FORWARD", 
           lambda parents, op: tdnode.ForwardGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = common.arepeat(pnode, valency)
    parents = tdnode.Parents(pnodes)

    return ParentData(parents, pnodes)




def reverse_parents(valency):

    def pnode():

        flow = tdnode.Flow("REVERSE", 
           lambda parents, op: tdnode.ReverseGate(parents, op))

        return fake.NodeLike(flow=fake.Fun(flow))

    pnodes  = common.arepeat(pnode, valency)
    parents = tdnode.Parents(pnodes)

    return ParentData(parents, pnodes)




# --- Adjoint function data ---------------------------------------------------------- #

AdjFunData = collections.namedtuple("AdjFunData", [
                 "adjfun", "grads", "seed",
              ])




def forward_adjfun(valency):

    grads  = common.arepeat(fake.Value, valency)
    seed   = common.arepeat(fake.Value, valency)
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




def reverse_adjfun(valency):

    grads  = common.arepeat(fake.Value, valency)
    seed   = fake.Value()
    adjfun = fake.Fun(grads, seed) 

    return AdjFunData(adjfun, grads, seed)




# --- Adjoint data ---------------------------------------------------------- #

AdjointData = collections.namedtuple("AdjointData", [
                 "op", "fun", "adxs", "out", "args", 
              ])


# Keep this in main test
#
# adjfun_data  = forward_adjfun_data(valency)
# tda.jvpmap.add_combo(fun, adjfun_data.adjfun)
#
# vjpfun = fake.Fun(grads, seed) 
# tda.vjpmap.add_combo(fun, vjpfun)
#



def forward_adjoint(nargs, adxs):  

    out  = fake.NodeLike()
    args = common.arepeat(fake.NodeLike, nargs)

    fun = fake.Fun(out, args)
    op  = tdnode.AdjointOp(fun, adxs, out, args)

    return AdjointData(
              op, 
              fun, adxs, out, args, 
           )



def reverse_adjoint(nargs, adxs):

    out  = fake.NodeLike()
    args = common.arepeat(fake.NodeLike, nargs)

    fun = fake.Fun(out, args)
    op  = tdnode.AdjointOp(fun, adxs, out, args)

    return AdjointData(
              op, 
              fun, adxs, out, args, 
           )




# --- Gate data ------------------------------------------------------------- #

GateData = collections.namedtuple("GateData", [
              "gate", "parents", "grads", "seed",
           ])




def forward_gate(valency):

    grads = common.arepeat(fake.Value, valency)
    seed  = common.arepeat(fake.Value, valency)
    op    = fake.Adjoint(jvp=fake.Fun(grads, seed))

    parents = common.arepeat(fake.NodeLike, valency)
    gate    = tdnode.ForwardGate(parents, op) 

    return GateData(gate, parents, grads, seed)




def reverse_gate(valency):

    grads = common.arepeat(fake.Value, valency)
    seed  = fake.Value()
    op    = fake.Adjoint(vjp=fake.Fun(grads, seed))

    parents = common.arepeat(fake.NodeLike, valency)
    gate    = tdnode.ReverseGate(parents, op) 

    return GateData(gate, parents, grads, seed)




# --- Node data ------------------------------------------------------------- #

NodeData = collections.namedtuple("NodeData", [
              "node", "source", "value",
           ])




def node(layer, gate):

    value  = fake.Value()
    source = fake.NodeLike(tovalue=fake.Fun(value))

    node = tdnode.Node(source, layer, gate)

    return NodeData(node, source, value)







"""
def node_data(GateType):

   def wrap(valency, layer, op, grads, seed):

       value   = fake.Value()
       source  = fake.NodeLike(tovalue=fake.Fun(value))
       parents = common.arepeat(fake.NodeLike, valency)

       gate = GateType(parents, op) 
       node = tdnode.Node(source, layer, gate)

       return NodeData(
                 node, source, layer, gate, 
                 value, parents, grads, seed
              )

   return wrap




@pytest.fixture
def forward_node_data():
 
   def wrap(valency=1, layer=0):

       grads = common.arepeat(fake.Value, valency)
       seed  = common.arepeat(fake.Value, valency)
       op    = fake.Adjoint(jvp=fake.Fun(grads, (seed,)))

       return node_data(ForwardGate)(valency, layer, op, grads, seed)

   return wrap




@pytest.fixture
def reverse_node_data():
 
   def wrap(valency=1, layer=0):

       grads = common.arepeat(fake.Value, valency)
       seed  = fake.Value()
       op    = fake.Adjoint(vjp=fake.Fun(grads, (seed,)))

       return node_data(ReverseGate)(valency, layer, op, grads, seed)

   return wrap
"""



















































