#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tests.common as common
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util           as util
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad

import tadpole.autodiff.map_jvp as jvpmap
import tadpole.autodiff.map_vjp as vjpmap




###############################################################################
###                                                                         ###
###  Adjoint operators associated with a specific function,                 ###
###  with knowledge of how to compute VJP's and JVP's.                      ###
###                                                                         ###
###############################################################################


# --- Adjoint operator ------------------------------------------------------ #

class TestAdjointOp:

   @pytest.mark.parametrize("nargs, adxs", [
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
   def test_eq(self, nargs, adxs):

       x = data.adjoint_dat(nargs, adxs)

       opA = an.AdjointOp(x.fun, x.adxs, x.out, x.args)
       opB = an.AdjointOp(x.fun, x.adxs, x.out, x.args)

       assert opA == opB
 

   @pytest.mark.parametrize("nargs, adxs", [
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
   def test_ne(self, nargs, adxs):

       x = data.adjoint_dat(nargs, adxs)
       y = data.adjoint_dat(nargs, adxs)

       ops = common.combos(an.AdjointOp)(
                (x.fun, x.adxs, x.out, x.args), 
                (y.fun, y.adxs, y.out, y.args),
             )
  
       opA = next(ops)
       for opB in ops:
           assert opA != opB


   @pytest.mark.parametrize("nargs, valency, adxs", [
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

       w = data.reverse_adjfun_dat(valency)
       x = data.adjoint_dat(nargs, adxs)
    
       vjpmap.add_raw(x.fun, fake.Fun(w.adjfun, x.adxs, x.out, *x.args)) 
                          
       assert tuple(x.op.vjp(w.seed)) == w.grads


   @pytest.mark.parametrize("nargs, valency, adxs", [
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

       w = data.forward_adjfun_dat(valency)
       x = data.adjoint_dat(nargs, adxs)

       jvpmap.add_raw(x.fun, fake.Fun(w.adjfun, x.adxs, x.out, *x.args))

       assert tuple(x.op.jvp(w.seed)) == w.grads




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- Flow ------------------------------------------------------------------ # 

class TestFlow:

   @pytest.mark.parametrize("name", ["REVERSE", "FORWARD", "NULL"])
   def test_eq(self, name):
       
       x = an.Flow(name, fake.Fun(fake.GateLike()))
       y = an.Flow(name, fake.Fun(fake.GateLike()))  

       assert x == y


   @pytest.mark.parametrize("nameA, nameB", [
      ["REVERSE", "FORWARD"], 
      ["FORWARD", "NULL"], 
      ["REVERSE", "NULL"],
   ])
   def test_ne(self, nameA, nameB):

       gate = fake.GateLike()
       
       x = an.Flow(nameA, fake.Fun(gate))
       y = an.Flow(nameB, fake.Fun(gate))  

       assert x != y


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD", "NULL"])
   def test_add(self, which):

       def select(which):
           return {
                   "REVERSE": data.reverse_flow_dat, 
                   "FORWARD": data.forward_flow_dat, 
                   "NULL":    data.null_flow_dat,
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
                   "REVERSE": data.reverse_flow_dat, 
                   "FORWARD": data.forward_flow_dat, 
                   "NULL":    data.null_flow_dat,
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

       w = data.reverse_gate_dat()
       x = data.reverse_flow_dat()

       assert x.flow.gate(w.parents, w.op) == w.gate


   def test_forward_gate(self):

       w = data.forward_gate_dat()
       x = data.forward_flow_dat()

       assert x.flow.gate(w.parents, w.op) == w.gate

       


###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Null logic gate ------------------------------------------------------- #

class TestNullGate:

   def test_flow(self):

       f    = data.null_flow_dat()
       gate = an.NullGate()
       assert gate.flow() == f.flow


   def test_trace(self):

       node    = fake.NodeLike()
       parents = (fake.NodeLike(), fake.NodeLike())

       count  = ad.ChildCount({node: parents})
       count1 = ad.ChildCount({node: parents})
      
       gate = an.NullGate()
       assert gate.trace(node, count) == count1 


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       x    = data.forward_gate_dat(valency)
       node = fake.NodeLike()

       init_seed = fake.Value()

       grads  = ad.GradSum(init_seed, dict(zip(x.parents, x.seed))) 
       grads1 = ad.GradSum(init_seed, dict(zip(x.parents, x.seed)))

       gate = an.NullGate()
       assert gate.grads(node, grads) == grads1


   @pytest.mark.parametrize("valency", [2])
   def test_reverse_grads(self, valency):

       x    = data.reverse_gate_dat(valency)
       node = fake.NodeLike()

       grads  = ad.GradAccum(dict(zip(x.parents, x.grads)))
       grads1 = ad.GradAccum(dict(zip(x.parents, x.grads)))

       gate = an.NullGate()
       assert gate.grads(node, grads) == grads1


      

# --- Forward logic gate ---------------------------------------------------- #

class TestForwardGate:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.forward_gate_dat(valency)

       gateA = an.ForwardGate(x.parents, x.op)
       gateB = an.ForwardGate(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.forward_gate_dat(valency)
       y = data.forward_gate_dat(valency)

       gates = common.combos(an.ForwardGate)(
                  (x.parents, x.op), 
                  (y.parents, y.op)
               )

       gateA = next(gates)
       for gateB in gates:
           assert gateA != gateB


   def test_flow(self):

       f = data.forward_flow_dat()
       x = data.forward_gate_dat()
       assert x.gate.flow() == f.flow


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, valency):

       x    = data.forward_gate_dat(valency)
       node = fake.NodeLike()

       count  = ad.ChildCount()  
       count1 = ad.ChildCount({node: x.parents})

       assert x.gate.trace(node, count) == count1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.forward_node_dat(valency)
       node = fake.NodeLike()
       
       init_seed = fake.Value()

       gradmap  = dict(zip(x.parents, x.seed))
       gradmap1 = {**gradmap, node: sum(x.grads)}

       grads  = ad.GradSum(init_seed, gradmap) 
       grads1 = ad.GradSum(init_seed, gradmap1)

       assert x.gate.grads(node, grads) == grads1




# --- Reverse logic gate ---------------------------------------------------- #

class TestReverseGate:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_gate_dat(valency)

       gateA = an.ReverseGate(x.parents, x.op)
       gateB = an.ReverseGate(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_gate_dat(valency)
       y = data.reverse_gate_dat(valency)

       gates = common.combos(an.ForwardGate)(
                  (x.parents, x.op), 
                  (y.parents, y.op)
               )

       gateA = next(gates)
       for gateB in gates:
           assert gateA != gateB


   def test_flow(self):

       f = data.reverse_flow_dat()
       x = data.reverse_gate_dat()
       assert x.gate.flow() == f.flow


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, valency):

       x    = data.reverse_gate_dat(valency)
       node = fake.NodeLike()

       count  = ad.ChildCount()  
       count1 = ad.ChildCount({node: x.parents})

       assert x.gate.trace(node, count) == count1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.reverse_node_dat(valency)
       node = fake.NodeLike()

       grads  = ad.GradAccum({node: x.seed})
       grads1 = ad.GradAccum({
                              None: x.seed, 
                              **dict(zip(x.parents, x.grads)),
                             })

       assert x.gate.grads(node, grads) == grads1




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Node ------------------------------------------------------------------ #

class TestNode:

   def test_eq(self):

       x = data.node_dat()

       nodeA = an.node(x.source, x.layer, x.gate)
       nodeB = an.node(x.source, x.layer, x.gate)

       assert nodeA == nodeB


   def test_ne(self):

       x = data.node_dat()
       y = data.node_dat()

       nodes = common.combos(an.Node)(
                  (x.source, x.layer, x.gate), 
                  (y.source, y.layer, y.gate)
               )

       nodeA = next(nodes)
       for nodeB in nodes:
           assert nodeA != nodeB


   def test_concat(self):

       x = data.node_dat()

       concat  = ag.Concatenation()
       concat1 = ag.Concatenation().attach(x.node, x.source, x.layer)

       assert x.node.concat(concat) == concat1 


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_flow(self, which):

       f = {
            "REVERSE": data.reverse_flow_dat, 
            "FORWARD": data.forward_flow_dat,
           }[which]()

       gate = fake.GateLike(flow=fake.Fun(f.flow))
       x    = data.node_dat(gate=gate)

       assert x.node.flow() == f.flow  


   @pytest.mark.parametrize("which",   ["REVERSE", "FORWARD"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_trace(self, which, valency):

       w = {
            "REVERSE": data.reverse_gate_dat, 
            "FORWARD": data.forward_gate_dat,
           }[which](valency)

       x = data.node_dat(gate=w.gate)

       count  = ad.ChildCount()  
       count1 = ad.ChildCount({x.node: w.parents})

       assert x.node.trace(count) == count1 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_reverse_grads(self, valency):

       x = data.reverse_node_dat(valency)

       grads  = ad.GradAccum({x.node: x.seed})
       grads1 = ad.GradAccum({
                              None: x.seed, 
                              **dict(zip(x.parents, x.grads)),
                             })

       assert x.node.grads(grads) == grads1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_forward_grads(self, valency):

       x = data.forward_node_dat(valency)

       init_seed = fake.Value()

       gradmap  = dict(zip(x.parents, x.seed))
       gradmap1 = {**gradmap, x.node: sum(x.grads)}

       grads  = ad.GradSum(init_seed, gradmap) 
       grads1 = ad.GradSum(init_seed, gradmap1)

       assert x.node.grads(grads) == grads1




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class TestPoint:

   def test_eq(self):

       x = data.point_dat()
       
       pointA = an.point(x.source)
       pointB = an.point(x.source)

       assert pointA == pointB


   def test_ne(self):

       x = data.point_dat()
       y = data.point_dat()

       pointA = an.point(x.source)
       pointB = an.point(y.source)

       assert pointA != pointB


   def test_concat(self):

       x = data.point_dat()

       concat  = ag.Concatenation() 
       concat1 = ag.Concatenation().attach(x.point, x.source, x.layer)

       assert x.point.concat(concat) == concat1 


   def test_flow(self):    

       f = data.null_flow_dat()
       x = data.point_dat() 
       assert x.point.flow() == f.flow  


   def test_trace(self):

       node    = fake.NodeLike()
       parents = (fake.NodeLike(), fake.NodeLike())

       count  = ad.ChildCount({node: parents})  
       count1 = ad.ChildCount({node: parents})

       x = data.point_dat()
       assert x.point.trace(count) == count1 


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       w    = data.forward_gate_dat(valency)
       node = fake.NodeLike()

       grads  = ad.GradSum(w.seed, {node: sum(w.grads)}) 
       grads1 = ad.GradSum(w.seed, {node: sum(w.grads)})

       x = data.point_dat()
       assert x.point.grads(grads) == grads1


   @pytest.mark.parametrize("valency", [2])
   def test_reverse_grads(self, valency):

       w = data.reverse_gate_dat(valency)

       grads  = ad.GradAccum(dict(zip(w.parents, w.grads)))
       grads1 = ad.GradAccum(dict(zip(w.parents, w.grads)))

       x = data.point_dat()
       assert x.point.grads(grads) == grads1




# --- NodeScape: draws new nodes -------------------------------------------- #

class TestNodeScape:

   def test_register(self):

       from tadpole.array.core   import randn
       from tadpole.array.core   import Array
       from tadpole.wrapper.node import Node 

       w = data.reverse_node_dat()

       nodescape = an.NodeScape()
       nodescape.register(Array, Node)

       source = randn((2,3,4))
       layer  = 0
       gate   = fake.GateLike()      

       out = nodescape.node(source, layer, gate)
       ans = Node(Node(source, -1, an.NullGate()), layer, gate) 

       assert isinstance(out, Node)
       assert out.allclose(ans)
       assert out == ans


   @pytest.mark.parametrize("which",   ["REVERSE", "FORWARD"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_node(self, which, valency):

       w = {
            "REVERSE": data.reverse_node_dat, 
            "FORWARD": data.forward_node_dat,
           }[which](valency)

       nodescape = an.NodeScape()
       assert nodescape.node(w.source, w.layer, w.gate) == w.node

       
   def test_point(self):

       w = data.point_dat()

       nodescape = an.NodeScape()
       assert nodescape.point(w.source) == w.point




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

       x = data.forward_parents_dat(valency)
        
       source = fake.NodeLike()
       op     = fake.Adjoint()
       gate   = an.ForwardGate(x.parents, op)
       node   = an.node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node


   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0])
   def test_reverse_next(self, valency, layer):

       x = data.reverse_parents_dat(valency)
        
       source = fake.NodeLike()
       op     = fake.Adjoint()
       gate   = an.ReverseGate(x.parents, op)
       node   = an.node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_parents_dat(valency)
       
       parentsA = an.Parents(*x.pnodes)
       parentsB = an.Parents(*x.pnodes)

       assert parentsA == parentsB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_parents_dat(valency)
       y = data.reverse_parents_dat(valency)
       
       parentsA = an.Parents(*x.pnodes)
       parentsB = an.Parents(*y.pnodes)

       assert parentsA != parentsB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_len(self, valency):

       x = data.reverse_parents_dat(valency)

       assert len(x.parents) == valency


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_contains(self, valency):

       x = data.reverse_parents_dat(valency)

       for node in x.pnodes:
           assert node in x.parents


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_iter(self, valency):

       x = data.reverse_parents_dat(valency)

       for nodeA, nodeB in zip(x.parents, x.pnodes):
           assert nodeA == nodeB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_getitem(self, valency):

       x = data.reverse_parents_dat(valency)

       for i, node in enumerate(x.pnodes):
           assert x.parents[i] == node
   




