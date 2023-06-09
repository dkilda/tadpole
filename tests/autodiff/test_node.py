#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tests.common as common
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad




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

       x = data.adjointop_dat(nargs, len(adxs), adxs)

       opA = an.AdjointOpGen(x.fun, x.adxs, x.out, x.args)
       opB = an.AdjointOpGen(x.fun, x.adxs, x.out, x.args)

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

       x = data.adjointop_dat(nargs, len(adxs), adxs)
       y = data.adjointop_dat(nargs, len(adxs), adxs)

       ops = common.combos(an.AdjointOpGen)(
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

       w = data.adjointop_dat(nargs, valency, adxs)
                          
       assert tuple(w.op.vjp(w.vjpseed)) == w.vjpgrads


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

       w = data.adjointop_dat(nargs, valency, adxs)

       assert tuple(w.op.jvp(w.jvpseed)) == w.jvpgrads




###############################################################################
###                                                                         ###
###  Flow: defines the direction of propagation through AD graph.           ###
###                                                                         ###
###############################################################################


# --- Flow ------------------------------------------------------------------ # 

class TestFlow:

   @pytest.mark.parametrize("name", ["REVERSE", "FORWARD", "NULL"])
   def test_eq(self, name):
       
       x = an.FlowGen(name, fake.Fun(fake.Gate()))
       y = an.FlowGen(name, fake.Fun(fake.Gate()))  

       assert x == y


   @pytest.mark.parametrize("nameA, nameB", [
      ["REVERSE", "FORWARD"], 
      ["FORWARD", "NULL"], 
      ["REVERSE", "NULL"],
   ])
   def test_ne(self, nameA, nameB):

       gate = fake.Gate()
       
       x = an.FlowGen(nameA, fake.Fun(gate))
       y = an.FlowGen(nameB, fake.Fun(gate))  

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

class TestGateNull:

   def test_flow(self):

       w = data.null_flow_dat()

       gate = an.GateNull()
       assert gate.flow() == w.flow


   def test_log(self):

       node    = fake.Node()
       parents = (fake.Node(), fake.Node())

       log  = ad.NodeLogVanilla(*parents)
       log1 = ad.NodeLogVanilla(*parents)
      
       gate = an.GateNull()
       gate.log(log) 

       assert log == log1


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       x    = data.forward_gate_dat(valency)
       node = fake.Node()

       init_seed = fake.Value()

       grads  = ad.GradSum(init_seed, dict(zip(x.parents, x.seed))) 
       grads1 = ad.GradSum(init_seed, dict(zip(x.parents, x.seed)))

       gate = an.GateNull()
       assert gate.grads(node, grads) == grads1


   @pytest.mark.parametrize("valency", [2])
   def test_reverse_grads(self, valency):

       x    = data.reverse_gate_dat(valency)
       node = fake.Node()

       grads  = ad.GradAccum(dict(zip(x.parents, x.grads)))
       grads1 = ad.GradAccum(dict(zip(x.parents, x.grads)))

       gate = an.GateNull()
       assert gate.grads(node, grads) == grads1


      

# --- Forward logic gate ---------------------------------------------------- #

class TestGateForward:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.forward_gate_dat(valency)

       gateA = an.GateForward(x.parents, x.op)
       gateB = an.GateForward(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.forward_gate_dat(valency)
       y = data.forward_gate_dat(valency)

       gates = common.combos(an.GateForward)(
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
   def test_log(self, valency):

       x    = data.forward_gate_dat(valency)
       node = fake.Node()

       log  = ad.NodeLogVanilla()  
       log1 = ad.NodeLogVanilla(*x.parents)

       x.gate.log(log) 
       assert log == log1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.forward_node_dat(valency)
       node = fake.Node()
       
       init_seed = fake.Value()

       gradmap  = dict(zip(x.parents, x.seed))
       gradmap1 = {**gradmap, node: sum(x.grads)}

       grads  = ad.GradSum(init_seed, gradmap) 
       grads1 = ad.GradSum(init_seed, gradmap1)

       assert x.gate.grads(node, grads) == grads1




# --- Reverse logic gate ---------------------------------------------------- #

class TestGateReverse:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_gate_dat(valency)

       gateA = an.GateReverse(x.parents, x.op)
       gateB = an.GateReverse(x.parents, x.op)

       assert gateA == gateB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_gate_dat(valency)
       y = data.reverse_gate_dat(valency)

       gates = common.combos(an.GateForward)(
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
   def test_log(self, valency):

       x    = data.reverse_gate_dat(valency)
       node = fake.Node()

       log  = ad.NodeLogVanilla()  
       log1 = ad.NodeLogVanilla(*x.parents)

       x.gate.log(log) 
       assert log == log1


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_grads(self, valency): 

       x    = data.reverse_node_dat(valency)
       node = fake.Node()

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


# --- General node ---------------------------------------------------------- #

class TestNodeGen:

   def test_eq(self):

       x = data.node_dat()

       nodeA = an.node(x.source, x.layer, x.gate)
       nodeB = an.node(x.source, x.layer, x.gate)

       assert nodeA == nodeB


   def test_ne(self):

       x = data.node_dat()
       y = data.node_dat()

       nodes = common.combos(an.NodeGen)(
                  (x.source, x.layer, x.gate), 
                  (y.source, y.layer, y.gate)
               )

       nodeA = next(nodes)
       for nodeB in nodes:
           assert nodeA != nodeB


   @pytest.mark.parametrize("layer1, layer2, result", [
      [ 0,  0,  True], 
      [ 1,  1,  True],
      [ 0,  1,  False],
      [ 1,  0,  False],
      [-1, -1,  False],
      [-1,  0,  False]
   ])
   def test_connected(self, layer1, layer2, result):

       if   layer1 > -1:
            x = data.node_dat(layer1).node
       else:
            x = data.point_dat().point

       if   layer2 > -1:
            y = data.node_dat(layer2).node
       else:
            y = data.point_dat().point

       assert x.connected(y) == result

       
   def test_concat(self):

       x = data.node_dat()

       concat  = ag.ConcatArgs()
       concat1 = ag.ConcatArgs().attach(x.node, x.source, x.layer)

       assert x.node.concat(concat) == concat1 


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_flow(self, which):

       f = {
            "REVERSE": data.reverse_flow_dat, 
            "FORWARD": data.forward_flow_dat,
           }[which]()

       gate = fake.Gate(flow=fake.Fun(f.flow))
       x    = data.node_dat(gate=gate)

       assert x.node.flow() == f.flow  


   @pytest.mark.parametrize("which",   ["REVERSE", "FORWARD"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_log(self, which, valency):

       w = {
            "REVERSE": data.reverse_gate_dat, 
            "FORWARD": data.forward_gate_dat,
           }[which](valency)

       x = data.node_dat(gate=w.gate)

       log  = ad.NodeLogVanilla()  
       log1 = ad.NodeLogVanilla(*w.parents)

       x.node.log(log)
       assert log == log1 


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

       concat  = ag.ConcatArgs() 
       concat1 = ag.ConcatArgs().attach(x.point, x.source, x.layer)

       assert x.point.concat(concat) == concat1 


   def test_flow(self):    

       f = data.null_flow_dat()
       x = data.point_dat() 
       assert x.point.flow() == f.flow  


   def test_log(self):

       node    = fake.Node()
       parents = (fake.Node(), fake.Node())

       log  = ad.NodeLogVanilla(*parents)  
       log1 = ad.NodeLogVanilla(*parents)

       x = data.point_dat()
       x.point.log(log) 

       assert log == log1 


   @pytest.mark.parametrize("valency", [2])
   def test_forward_grads(self, valency):

       w    = data.forward_gate_dat(valency)
       node = fake.Node()

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

       from tadpole.tensor     import allclose
       from tadpole.tensor     import randn
       from tadpole.tensor     import TensorGen
       from tadpole.index      import IndexGen
       from tadpole.tensorwrap import NodeTensor 

       w = data.reverse_node_dat()

       nodescape = an.NodeScape()
       nodescape.register(TensorGen, NodeTensor)

       source = randn((IndexGen("i",2), IndexGen("j",3), IndexGen("k",4)))
       layer  = 0
       gate   = fake.Gate()

       out = nodescape.node(source, layer, gate)
       ans = NodeTensor(NodeTensor(source, -1, an.GateNull()), layer, gate) 

       assert isinstance(out, NodeTensor)
       assert allclose(out, ans)
       assert out == ans


   @pytest.mark.parametrize("which",   ["REVERSE", "FORWARD"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_node(self, which, valency):

       w = {
            "REVERSE": data.reverse_node_dat, 
            "FORWARD": data.forward_node_dat,
           }[which](valency)

       nodescape = an.NodeScape()
       nodescape.register(fake.Node,  an.NodeGen)
       nodescape.register(fake.Value, an.NodeGen)

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

class TestParentsGen:

   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0])
   def test_forward_next(self, valency, layer):

       x = data.forward_parents_dat(valency)
        
       source = fake.Node()
       op     = fake.AdjointOp()
       gate   = an.GateForward(x.parents, op)
       node   = an.node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node


   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0])
   def test_reverse_next(self, valency, layer):

       x = data.reverse_parents_dat(valency)
        
       source = fake.Node()
       op     = fake.AdjointOp()
       gate   = an.GateReverse(x.parents, op)
       node   = an.node(source, layer, gate)

       assert x.parents.next(source, layer, op) == node 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       x = data.reverse_parents_dat(valency)
       
       parentsA = an.ParentsGen(*x.pnodes)
       parentsB = an.ParentsGen(*x.pnodes)

       assert parentsA == parentsB


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_ne(self, valency):

       x = data.reverse_parents_dat(valency)
       y = data.reverse_parents_dat(valency)
       
       parentsA = an.ParentsGen(*x.pnodes)
       parentsB = an.ParentsGen(*y.pnodes)

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
   




