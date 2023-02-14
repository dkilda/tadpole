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




# --- Node data ------------------------------------------------------------- #

NodeData = collections.namedtuple("NodeData", [
              "node", "source", "layer", "gate", "value"
           ])




def node(layer=0, gate=fake.GateLike()):

    value  = fake.Value()
    source = fake.NodeLike(tovalue=fake.Fun(value))

    node = tdnode.Node(source, layer, gate)

    return NodeData(node, source, layer, gate, value)




# --- Point data ------------------------------------------------------------ #

PointData = collections.namedtuple("PointData", [
              "point", "source", "layer", "gate"
           ])




def point():

    source = fake.Value() 
    layer  = tdgraph.minlayer()
    gate   = tdnode.NullGate()

    point = tdnode.Point(source)

    return PointData(node, source, layer, gate)




# --- Helpers --------------------------------------------------------------- #

def make_combos(typ):

    def wrap(*xs):

        yield common.amap(typ, xs)

        for xcombo in itertools.product(*xs):
            yield common.amap(typ, xcombo) 

    return wrap




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- Flow ------------------------------------------------------------------ # 

class TestFlow:

   @pytest.mark.parametrize("name", ["REVERSE", "FORWARD", "NULL"])
   def test_eq(self, name):
       
       x = tdnode.Flow(name, fake.Fun(fake.GateLike()))
       y = tdnode.Flow(name, fake.Fun(fake.GateLike()))  

       assert x == y


   @pytest.mark.parametrize("nameA, nameB", [
      ["REVERSE", "FORWARD"], 
      ["FORWARD", "NULL"], 
      ["REVERSE", "NULL"],
   ])
   def test_ne(self, nameA, nameB):

       gate = fake.GateLike()
       
       x = tdnode.Flow(nameA, fake.Fun(gate))
       y = tdnode.Flow(nameB, fake.Fun(gate))  

       assert x != y


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD", "NULL"])
   def test_add(self, which):

       def select(which):
           return {
                   "REVERSE": data.reverse_flow, 
                   "FORWARD": data.forward_flow, 
                   "NULL":    data.null_flow,
                  }[which]()

       x = select(which)
       y = select(which)

       assert x.flow + y.flow == x.flow
       assert x.flow + y.flow == y.flow

       assert x.flow + 0    == x.flow
       assert x.flow + None == x.flow


   @pytest.mark.parametrize("flowA, flowB", [
      ["REVERSE", "FORWARD"], 
      ["FORWARD", "NULL"], 
      ["REVERSE", "NULL"],
   ])
   def test_failed_add(self, flowA, flowB):

       def select(which):
           return {
                   "REVERSE": data.reverse_flow, 
                   "FORWARD": data.forward_flow, 
                   "NULL":    data.null_flow,
                  }[which]()

       x = select(flowA)
       y = select(flowB)

       try:
           out = x.flow + y.flow
       except ValueError:
           assert True
       else:
           assert False


   def test_reverse_gate(self):

       w = data.reverse_gate()
       x = data.reverse_flow()

       assert x.gate(w.parents, w.op) == w.gate


   def test_forward_gate(self):

       w = data.forward_gate()
       x = data.forward_flow()

       assert x.gate(w.parents, w.op) == w.gate

       


###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint operator ------------------------------------------------------ #

class TestAdjointOp:

   @pytest.mark.parametrize("nargs, adxs", 
      [1, (0,)],
      [2, (0,)],
      [2, (1,)],
      [2, (0,1)],
      [3, (0,)],
      [3, (1,)],
      [3, (2,)],
      [3, (0,1)],
      [3, (0,2)],
      [3, (1,2)],
      [3, (0,1,2)],
   ])
   def test_eq(self):

       x = data.adjoint(nargs, adxs)

       opA = AdjointOp(x.fun, x.adxs, x.out, x.args)
       opB = AdjointOp(x.fun, x.adxs, x.out, x.args)

       return opA == opB
 

   @pytest.mark.parametrize("nargs, adxs", 
      [1, (0,)],
      [2, (0,)],
      [2, (1,)],
      [2, (0,1)],
      [3, (0,)],
      [3, (1,)],
      [3, (2,)],
      [3, (0,1)],
      [3, (0,2)],
      [3, (1,2)],
      [3, (0,1,2)],
   ])
   def test_ne(self):

       x = data.adjoint(nargs, adxs)
       y = data.adjoint(nargs, adxs)

       combos = make_combos(tdnode.AdjointOp)

       for opA, opB in combos(
                              (x.fun, x.adxs, x.out, x.args), 
                              (y.fun, y.adxs, y.out, y.args)
                             ):
           assert opA != opB


   @pytest.mark.parametrize("nargs, valency, adxs", 
      [1, 1, (0,)],
      [2, 1, (0,)],
      [2, 1, (1,)],
      [2, 2, (0,1)],
      [3, 1, (0,)],
      [3, 1, (1,)],
      [3, 1, (2,)],
      [3, 2, (0,1)],
      [3, 2, (0,2)],
      [3, 2, (1,2)],
      [3, 3, (0,1,2)],
   ])
   def test_vjp(self, nargs, valency, adxs):

       w = data.reverse_adjfun(valency)
       x = data.adjoint(nargs, adxs)

       tda.vjpmap.add_combo(x.fun, w.adjfun)

       assert x.op.vjp(w.seed) == w.grads


   @pytest.mark.parametrize("nargs, valency, adxs", 
      [1, 1, (0,)],
      [2, 1, (0,)],
      [2, 1, (1,)],
      [2, 2, (0,1)],
      [3, 1, (0,)],
      [3, 1, (1,)],
      [3, 1, (2,)],
      [3, 2, (0,1)],
      [3, 2, (0,2)],
      [3, 2, (1,2)],
      [3, 3, (0,1,2)],
   ])
   def test_jvp(self, nargs, valency, adxs):

       w = data.forward_adjfun(valency)
       x = data.adjoint(nargs, adxs)

       tda.jvpmap.add_combo(x.fun, w.adjfun)

       assert x.op.jvp(w.seed) == w.grads




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Null logic gate ------------------------------------------------------- #

class TestNullGate:

   def test_flow(self):

       f    = data.null_flow()
       gate = tdnode.NullGate()
       assert gate.flow() == f.flow


   def test_trace(self):

       node    = fake.NodeLike()
       parents = (fake.NodeLike(), fake.NodeLike())

       count  = tdgrad.ChildCount({node: parents})
       count1 = tdgrad.ChildCount({node: parents})
      
       gate = tdnode.NullGate()
       assert gate.trace(count) == count1 


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       x    = data.forward_gate(valency)
       node = fake.NodeLike()

       grads  = tdgrad.GradSum(x.seed, {node: sum(x.grads)}) 
       grads1 = tdgrad.GradSum(x.seed, {node: sum(x.grads)})

       gate = tdnode.NullGate()
       assert gate.grads(node, grads) == grads1


   @pytest.mark.parametrize("valency", [2])
   def test_reverse_grads(self, valency):

       x    = data.reverse_gate(valency)
       node = fake.NodeLike()

       grads  = tdgrad.GradAccum(dict(zip(x.parents, x.grads)))
       grads1 = tdgrad.GradAccum(dict(zip(x.parents, x.grads)))

       gate = tdnode.NullGate()
       assert x.gate.grads(node, grads) == grads1


      

# --- Forward logic gate ---------------------------------------------------- #

class TestForwardGate:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.forward_gate(valency)

       gateA = tdnode.ForwardGate(x.parents, x.op)
       gateB = tdnode.ForwardGate(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.forward_gate(valency)
       y = data.forward_gate(valency)

       combos = make_combos(tdnode.ForwardGate)

       for gateA, gateB in combos(
                                  (x.parents, x.op), 
                                  (y.parents, y.op)
                                 ):
           assert gateA != gateB


   def test_flow(self):

       f = data.forward_flow()
       x = data.forward_gate()
       assert x.gate.flow() == f.flow


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, valency):

       x    = data.forward_gate(valency)
       node = fake.NodeLike()

       count  = tdgrad.ChildCount()  
       count1 = tdgrad.ChildCount({node: x.parents})

       assert x.gate.trace(node, count) == count1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.forward_gate(valency)
       node = fake.NodeLike()

       grads  = tdgrad.GradSum(x.seed) 
       grads1 = tdgrad.GradSum(x.seed, {node: sum(x.grads)})

       assert x.node.grads(grads) == grads1




# --- Reverse logic gate ---------------------------------------------------- #

class TestReverseGate:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_gate(valency)

       gateA = tdnode.ReverseGate(x.parents, x.op)
       gateB = tdnode.ReverseGate(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_gate(valency)
       y = data.reverse_gate(valency)

       combos = make_combos(tdnode.ReverseGate)

       for gateA, gateB in combos(
                                  (x.parents, x.op), 
                                  (y.parents, y.op)
                                 ):
           assert gateA != gateB


   def test_flow(self):

       f = data.reverse_flow()
       x = data.reverse_gate()
       assert x.gate.flow() == f.flow


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, valency):

       x    = data.reverse_gate(valency)
       node = fake.NodeLike()

       count  = tdgrad.ChildCount()  
       count1 = tdgrad.ChildCount({node: x.parents})

       assert x.gate.trace(node, count) == count1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.reverse_gate(valency)
       node = fake.NodeLike()

       grads  = tdgrad.GradAccum({node: x.seed})
       grads1 = tdgrad.GradAccum(dict(zip(x.parents, x.grads)))

       assert x.gate.grads(node, grads) == grads1




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #

class TestNode:

   def test_eq(self):

       x = data.node()

       nodeA = tdnode.Node(x.source, x.layer, x.gate)
       nodeB = tdnode.Node(x.source, x.layer, x.gate)

       assert nodeA == nodeB


   def test_ne(self):

       x = data.node()
       y = data.node()

       combos = make_combos(tdnode.Node)

       for nodeA, nodeB in combos(
                                  (x.source, x.layer, x.gate), 
                                  (y.source, y.layer, y.gate)
                                 ):
           assert nodeA != nodeB


   def test_tovalue(self):

       x = data.node()
       assert x.node.tovalue() == x.value


   def test_concat(self):

       x = data.node()

       concat  = tdgraph.Concatenation()
       concat1 = tdgraph.Concatenation(
                                       tdutil.Sequence([x.node]), 
                                       tdutil.Sequence([x.source]), 
                                       tdutil.Sequence([x.layer])
                                      ) 

       assert x.node.concat(concat) == concat1 


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_flow(self, which):

       f = {
            "REVERSE": data.reverse_flow, 
            "FORWARD": data.forward_flow,
           }[which]()

       gate = fake.GateLike(flow=fake.Fun(f.flow))
       x    = data.node(gate=gate)

       assert x.node.flow() == f.flow  


   @pytest.mark.parametrize("which",   ["REVERSE", "FORWARD"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, which, valency):

       w = {
            "REVERSE": data.reverse_gate, 
            "FORWARD": data.forward_gate,
           }[which](valency)

       x = data.node(gate=w.gate)

       count  = tdgrad.ChildCount()  
       count1 = tdgrad.ChildCount({x.node: w.parents})

       assert x.node.trace(count) == count1 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_reverse_grads(self, valency):

       w = data.reverse_gate(valency)
       x = data.node(gate=w.gate)

       grads  = tdgrad.GradAccum({x.node: w.seed})
       grads1 = tdgrad.GradAccum(dict(zip(w.parents, w.grads)))

       assert x.node.grads(grads) == grads1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_forward_grads(self, valency):

       w = data.forward_gate(valency)
       x = data.node(gate=w.gate)

       grads  = tdgrad.GradSum(w.seed)
       grads1 = tdgrad.GradSum(w.seed, {x.node: sum(w.grads)})

       assert x.node.grads(grads) == grads1




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class TestPoint:

   def test_eq(self):

       x = data.point()
       
       pointA = tdnode.Point(x.source)
       pointB = tdnode.Point(x.source)

       assert pointA == pointB


   def test_ne(self):

       x = data.point()
       y = data.point()

       pointA = tdnode.Point(x.source)
       pointB = tdnode.Point(y.source)

       assert pointA != pointB


   def test_tovalue(self):

       x = data.point()
       assert x.point.tovalue() == x.source


   def test_concat(self):

       x = data.point()

       concat  = tdgraph.Concatenation() 
       concat1 = tdgraph.Concatenation(
                                       tdutil.Sequence([x.point]), 
                                       tdutil.Sequence([x.point]), 
                                       tdutil.Sequence([x.layer])
                                      ) 

       assert x.point.concat(concat) == concat1 


   def test_flow(self):    

       f = data.null_flow()
       x = data.point() 
       assert x.point.flow() == f.flow  


   def test_trace(self):

       node    = fake.NodeLike()
       parents = (fake.NodeLike(), fake.NodeLike())

       count  = tdgrad.ChildCount({node: parents})  
       count1 = tdgrad.ChildCount({node: parents})

       x = data.point()
       assert x.point.trace(count) == count1 


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       w    = data.forward_gate(valency)
       node = fake.NodeLike()

       grads  = tdgrad.GradSum(w.seed, {node: sum(w.grads)}) 
       grads1 = tdgrad.GradSum(w.seed, {node: sum(w.grads)})

       x = data.point()
       assert x.point.grads(grads) == grads1


   @pytest.mark.parametrize("valency", [2])
   def test_reverse_grads(self, valency):

       w = data.reverse_gate(valency)

       grads  = tdgrad.GradAccum(dict(zip(w.parents, w.grads)))
       grads1 = tdgrad.GradAccum(dict(zip(w.parents, w.grads)))

       x = data.point()
       assert x.point.grads(grads) == grads1




###############################################################################
###                                                                         ###
###  Parents of an autodiff Node.                                           ###
###                                                                         ###
###############################################################################


# --- Parents --------------------------------------------------------------- #

class TestParents:

   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0])
   def test_forward_next(self, valency, layer):

       x = data.forward_parents(valency)
        
       source = fake.NodeLike()
       op     = fake.Adjoint()
       gate   = tdnode.ForwardGate(x.parents, op)
       node   = tdnode.Node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node


   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0])
   def test_reverse_next(self, valency, layer):

       x = data.reverse_parents(valency)
        
       source = fake.NodeLike()
       op     = fake.Adjoint()
       gate   = tdnode.ReverseGate(x.parents, op)
       node   = tdnode.Node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_parents(valency)
       
       parentsA = tdnode.Parents(x.pnodes)
       parentsB = tdnode.Parents(x.pnodes)

       assert parentsA == parentsB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_parents(valency)
       y = data.reverse_parents(valency)
       
       parentsA = tdnode.Parents(x.pnodes)
       parentsB = tdnode.Parents(y.pnodes)

       assert parentsA != parentsB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_len(self, valency):

       x = data.reverse_parents(valency)

       assert len(x.parents) == valency


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_contains(self, valency):

       x = data.reverse_parents(valency)

       for node in x.pnodes:
           assert node in x.parents


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_iter(self, valency):

       x = data.reverse_parents(valency)

       for nodeA, nodeB in zip(x.parents, x.pnodes):
           assert nodeA == nodeB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_getitem(self, valency):

       x = data.reverse_parents(valency)

       for i, node in enumerate(x.pnodes):
           assert x.parents[i] == node
   





