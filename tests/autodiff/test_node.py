#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.node     as tdnode
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Boilerplate code for testing forward and reverse objects.              ###
###                                                                         ###
###############################################################################


# --- Boilerplate for testing forward objects ------------------------------- #

class TestForward:

   @pytest.fixture(autouse=True)
   def request_args(self, jvpfun_args):

       self.args = jvpfun_args


   @pytest.fixture(autouse=True)
   def request_logic(self, forward_logic):

       self.logic = forward_logic


   @pytest.fixture(autouse=True)
   def request_gate(self, forward_gate):

       self.gate = forward_gate


   @pytest.fixture(autouse=True)
   def request_node(self, forward_node):

       self.node = forward_node


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.fixture(autouse=True)
   def request_point(self, point):

       self.point = point




# --- Boilerplate for testing reverse objects ------------------------------- #

class TestReverse:

   @pytest.fixture(autouse=True)
   def request_args(self, vjpfun_args):

       self.args = vjpfun_args


   @pytest.fixture(autouse=True)
   def request_logic(self, reverse_logic):

       self.logic = reverse_logic


   @pytest.fixture(autouse=True)
   def request_gate(self, reverse_gate):

       self.gate = reverse_gate


   @pytest.fixture(autouse=True)
   def request_node(self, reverse_node):

       self.node = reverse_node


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.fixture(autouse=True)
   def request_point(self, point):

       self.point = point




###############################################################################
###                                                                         ###
###  Logic of forward and reverse propagation, creates logic gates.         ###
###                                                                         ###
###############################################################################


# --- Forward logic --------------------------------------------------------- #

class TestForwardLogic(TestForward):


   @pytest.mark.parametrize("valency",  [1,2,3])
   def test_eq(self, valency):

       adxs, out, args = self.args(valency)
       parents         = tpl.repeat(fake.ForwardNode, valency)

       logicA  = self.logic(parents, adxs, out, args)
       logicB  = self.logic(parents, adxs, out, args)

       assert logicA == logicB


   def test_ne(self):

       logicA = self.logic()
       logicB = self.logic()

       assert logicA != logicB


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_gate(self, valency):

       fun     = fake.Fun()
       jvpfuns = tpl.repeat(fake.Fun, valency) 

       tda.jvpmap.add(fun, *jvpfuns) # FIXME having to do this is an argument for keeping vjp/jvp 
                                     #       info in Fun classes instead of using global adjoint maps!
       parents = tpl.repeat(fake.ForwardNode, valency)  
       logic   = self.logic(parents)
       ans     = self.gate(parents, fun)

       assert logic.gate(fun) == ans


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_make_logic(self, valency):

       ans     = fake.ForwardLogic()
       parents = tpl.repeat(lambda: fake.ForwardNode(logic=ans), valency)   

       adxs, out, args = self.args(valency)

       assert tdnode.make_logic(parents, adxs, out, args) == ans




# --- Reverse logic --------------------------------------------------------- #

class TestReverseLogic(TestReverse):


   @pytest.mark.parametrize("valency",  [1,2,3])
   def test_eq(self, valency):

       adxs, out, args = self.args(valency)
       parents         = tpl.repeat(fake.ReverseNode, valency)

       logicA  = self.logic(parents, adxs, out, args)
       logicB  = self.logic(parents, adxs, out, args)

       assert logicA == logicB


   def test_ne(self):

       logicA = self.logic()
       logicB = self.logic()

       assert logicA != logicB


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_gate(self, valency):

       fun     = fake.Fun(valency)
       vjpfuns = tpl.repeat(fake.Fun, valency) 

       tda.vjpmap.add(fun, *vjpfuns) # FIXME having to do this is an argument for keeping vjp/jvp 
                                     #       info in Fun classes instead of using global adjoint maps!

       parents = tpl.repeat(fake.ReverseNode, valency)  
       logic   = self.logic(parents)
       ans     = self.gate(parents, fun, tda.vjpmap.get(fun)) # FIXME in the future we should also test vjp/jvp equality!

       assert logic.gate(fun) == ans


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_make_logic(self, valency):

       ans     = fake.ReverseLogic()
       parents = tpl.repeat(lambda: fake.ReverseNode(logic=ans), valency)  

       adxs, out, args = self.args(valency)

       assert tdnode.make_logic(parents, adxs, out, args) == ans




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Forward gate ---------------------------------------------------------- #

class TestForwardGate(TestForward):


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       parents = tpl.repeat(fake.ForwardNode, valency)
       fun     = fake.Fun()
       grad    = fake.Fun()

       gateA  = self.gate(parents, fun, grad)
       gateB  = self.gate(parents, fun, grad)

       assert gateA == gateB


   def test_ne(self):

       gateA = self.gate()
       gateB = self.gate()

       assert gateA != gateB


   def test_nodify(self):
 
       nodule = fake.Nodule()

       gate = self.gate()
       ans  = self.node(nodule, gate)

       assert gate.nodify(nodule) == ans  


   def test_grad(self):

       grad = fake.FunReturn()
       gate = self.gate(grad=grad)

       assert gate.grad() == grad




# --- Reverse gate ---------------------------------------------------------- #

class TestReverseGate(TestReverse):


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_eq(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       fun     = fake.Fun()
       vjp     = fake.Fun()

       gateA  = self.gate(parents, fun, vjp)
       gateB  = self.gate(parents, fun, vjp)

       assert gateA == gateB


   def test_ne(self):

       gateA = self.gate()
       gateB = self.gate()

       assert gateA != gateB


   def test_nodify(self):
 
       nodule = fake.Nodule()

       gate = self.gate()
       ans  = self.node(nodule, gate)

       assert gate.nodify(nodule) == ans  


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_accumulate_parent_grads(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       grads   = tpl.repeat(fake.FunReturn,   valency) 

       seed  = fake.FunReturn()
       vjp   = fake.Fun({(seed,): grads})
       gate  = self.gate(parents, vjp=vjp)

       accum = fake.GradAccum()
       gate.accumulate_parent_grads(seed, accum)
       
       assert tpl.amap(accum.accumulated, parents) == grads 


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_add_to_childcount(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       gate    = self.gate(parents)

       childcount = fake.ChildCount()
       gate.add_to_childcount(childcount)
       
       assert childcount.added() == parents 


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_add_to_toposort(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       gate    = self.gate(parents)

       toposort = fake.TopoSort()
       gate.add_to_toposort(toposort)

       assert tpl.link(toposort.added, valency) == parents




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Nodule: a node kernel ------------------------------------------------- #

class TestNodule:


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.mark.parametrize("source", [
      fake.ForwardNode(), 
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])
   @pytest.mark.parametrize("layer",  [0, 1, 2])
   def test_eq(self, source, layer):

       nodeA  = self.nodule(source, layer)
       nodeB  = self.nodule(source, layer)

       assert nodeA == nodeB


   def test_ne(self):

       nodeA = self.nodule()
       nodeB = self.nodule()

       assert nodeA != nodeB

  
   def test_tovalue(self):

       ans  = fake.FunReturn()
       node = self.nodule(source=fake.Point(tovalue=ans))

       assert node.tovalue() == ans 


   def test_attach(self): 

       ans   = fake.NodeTrain()               
       train = fake.NodeTrain(with_meta=ans)  

       node = self.nodule()

       assert node.attach(train) == ans




# --- Forward node ---------------------------------------------------------- #

class TestForwardNode(TestForward): 


   def test_eq(self):

       nodule = fake.Nodule()
       gate   = fake.ForwardGate()
       nodeA  = self.node(nodule, gate)
       nodeB  = self.node(nodule, gate)

       assert nodeA == nodeB


   def test_ne(self):

       nodeA = self.node()
       nodeB = self.node()

       assert nodeA != nodeB


   @pytest.mark.parametrize("x", [
      fake.ForwardNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])
   def test_nodify(self, x):

       ans = x
       if isinstance(ans, fake.FunReturn):
          ans = self.point(ans)
       
       assert tdnode.nodify(x) == ans 
       

   @pytest.mark.parametrize("source", [
      fake.ForwardNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])
   @pytest.mark.parametrize("layer",  [0, 1, 2])
   def test_make_node(self, source, layer):

       nodule = self.nodule(tdnode.nodify(source), layer)
       node   = fake.ForwardNode()
       gate   = fake.ForwardGate(nodify=fake.Map({nodule: node}))

       assert tdnode.make_node(source, layer, gate) == node


   def test_tovalue(self):

       ans  = fake.FunReturn()
       node = self.node(fake.Nodule(tovalue=ans))

       assert node.tovalue() == ans 

 
   def test_attach(self): 

       ans    = fake.NodeTrain()               
       train1 = fake.NodeTrain(with_meta=ans)  
       train2 = fake.NodeTrain(with_node=train1) 

       node = self.node(fake.Nodule(attach={train1: ans}))

       assert node.attach(train2) == ans


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_logic(self, valency):

       adxs, out, args = self.args(valency)

       others = tpl.repeat(fake.ForwardNode, valency-1) 
       node   = self.node()
       ans    = self.logic((node, *others), adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

    
   def test_grad(self):

       grad = fake.FunReturn()  
       node = self.node(gate=fake.ForwardGate(grad=grad))

       assert node.grad() == grad  




# --- Reverse node ---------------------------------------------------------- #

class TestReverseNode(TestReverse): 


   def test_eq(self):

       nodule = fake.Nodule()
       gate   = fake.ReverseGate()
       nodeA  = self.node(nodule, gate)
       nodeB  = self.node(nodule, gate)

       assert nodeA == nodeB


   def test_ne(self):

       nodeA = self.node()
       nodeB = self.node()

       assert nodeA != nodeB


   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])
   def test_nodify(self, x):

       ans = x
       if isinstance(ans, fake.FunReturn):
          ans = self.point(ans)
       
       assert tdnode.nodify(x) == ans 
       

   @pytest.mark.parametrize("source", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])
   @pytest.mark.parametrize("layer",  [0, 1, 2])
   def test_make_node(self, source, layer):

       nodule = self.nodule(tdnode.nodify(source), layer)
       node   = fake.ReverseNode()
       gate   = fake.ReverseGate(nodify=fake.Map({nodule: node}))

       assert tdnode.make_node(source, layer, gate) == node

  
   def test_tovalue(self):

       ans  = fake.FunReturn()
       node = self.node(fake.Nodule(tovalue=ans))

       assert node.tovalue() == ans 

 
   def test_attach(self): 

       ans    = fake.NodeTrain()               
       train1 = fake.NodeTrain(with_meta=ans)  
       train2 = fake.NodeTrain(with_node=train1) 

       node = self.node(fake.Nodule(attach={train1: ans}))

       assert node.attach(train2) == ans


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_logic(self, valency):

       adxs, out, args = self.args(valency)

       others = tpl.repeat(fake.ReverseNode, valency-1) 
       node   = self.node()
       ans    = self.logic((node, *others), adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

    
   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_accumulate_parent_grads(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       grads   = tpl.repeat(fake.FunReturn,   valency) 

       seed = fake.FunReturn()
       vjp  = fake.Fun({(seed,): grads})
       node = self.node(gate=self.gate(parents, vjp=vjp))

       accum = fake.GradAccum(pop={node: seed})
       node.accumulate_parent_grads(accum)
       
       assert tpl.amap(accum.accumulated, parents) == grads 


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_add_to_childcount(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       node    = self.node(gate=self.gate(parents))

       childcount = fake.ChildCount()
       node.add_to_childcount(childcount)
       
       assert all((
                   childcount.visited() == node,
                   childcount.added()   == parents,
                 ))


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_add_to_toposort(self, valency):

       parents = tpl.repeat(fake.ReverseNode, valency)
       node    = self.node(gate=self.gate(parents))

       toposort = fake.TopoSort()
       node.add_to_toposort(toposort)

       assert tpl.link(toposort.added, valency) == parents  




# --- Point (a disconnected node, only carries a value and no logic) -------- #

class TestPoint:


   @pytest.fixture(autouse=True)
   def request_point(self, point):

       self.point = point


   def test_eq(self):

       source = fake.FunReturn()
       nodeA  = self.point(source)
       nodeB  = self.point(source)

       assert nodeA == nodeB


   def test_ne(self):

       nodeA  = self.point()
       nodeB  = self.point()

       assert nodeA != nodeB

  
   def test_tovalue(self):

       ans  = fake.FunReturn()
       node = self.point(source=ans)

       assert node.tovalue() == ans 


   def test_attach(self): 

       ans    = fake.NodeTrain()               
       train1 = fake.NodeTrain(with_meta=ans)  
       train2 = fake.NodeTrain(with_node=train1) 

       node = self.point()

       assert node.attach(train2) == ans


















