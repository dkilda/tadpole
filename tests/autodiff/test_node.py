#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.util     as tdutil
import tadpole.autodiff.node     as tdnode
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.grad     as tdgrad
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl


"""
a) -- Make a cls that sets up all the fake data

b) -- Then wrap it with a cls that generates the actual object

c) -- So we can take cls from (a) and generate grads and parents 

d) -- Then we can define a seed (for another gate) using those grads

e) -- Create a real/fake gate with grads, parents, seed


class ReverseGateSetup:

   def with_seed(self, seed):

   def with_grads(self, grads):

   def with_parents(self, parents):


   def seed(self):
    
   def grads(self):

   def parents(self):


gradsA = gateA.grads()
gradsB = gateB.grads()

seed = gradsA[0] + gradsB[0]
   
gateC = gateC.with_seed(seed)
gateA = gateA.with_parents(gateC)

"""

"""
Altway: 

--- create separate classes for Parents, Grads

--- can set: grads = grads.with_seed(seed) 

--- pass'em to ReverseNode(gate=ReverseGate(parents, grads)) to create a new object

E.g.

class ReverseGate(Gate):

   @fakeit
   def accumulate_parent_grads(self, seed, grads):

       for parent, grad in zip(self._parents, self._grads.vjp(seed)): # NB Parents cls impls 
           grads.accumulate(parent, grad) 

       return self


--- Use a factory as follows:

grads = Grads()

... do some calcs to get seed ...

grads = grads.with_seed(seed)
setup = ReverseNodeSetup(ReverseGateSetup(parents, grads))
node  = setup.node()

which looks same as before, but parents/grads are Parents/Grads objects instead!
which we can setup before creating gate/node factory!


--- Let Setup/Factory cls take care of defaults... 


--- How to automate the above?

    def node_setup(..., parents, grads):
     
        return ReverseNodeSetup(ReverseGateSetup(parents, grads))

    takes care of defaults and sets it up in one go. The whole process becomes:

    grads = Grads()

    ... do some calcs to get seed ...

    grads = grads.with_seed(seed)
    setup = node_setup(..., parents, grads) 
    node  = setup.node()

    Much like before, but because grads = Grads obj, we can set seed on the same object
    plus keep grads-and-seed together.


SO THE MAIN CHANGES ARE:

--- Use Grads, ReverseParents, ForwardParents, Args, etc classes to manipulate groups of related objects


--- Use ReverseNodeSetup, ReverseGateSetup, etc factories as before, but their ctor arguments are 
    Grads, ReverseParents, etc objects instead of raw tuples/collections. Let these factories take care
    of generating defaults (e.g. parents/grads if they're not set).


--- For more specific test scenarios, we may wanna chain factories together, e.g. do
    ReverseNodeSetup(ReverseGateSetup(parents, grads)) in one go.

    The specific non-default args may vary from one test to another. But what we could do is define
    a factory function    

    def node_setup(parents, grads):

        return ReverseNodeSetup(ReverseGateSetup(parents, grads))
 
    or equivalently

    class ReverseNodeSetupEnvelope(ReverseNodeSetup):

       def __init__(self, parents, grads):

           super.__init__(ReverseGateSetup(parents, grads))
      
    to create a special setup for certain test cases (say, we could create multiple versions 
    of ReverseNodeSetupEnvelope for different test cases, like ReverseNodeSetupEnvelopeA, ReverseNodeSetupEnvelopeB, etc 
    which can take parents + grads, or source + layer, or source + parents depending on the test case).


--- The whole process looks like:

    grads = Grads()

    ... do some calcs to get seed ...

    grads = grads.with_seed(seed)
    setup = node_setup(..., parents, grads) / or / ReverseNodeSetupEnvelope(parents, grads)
    node  = setup.node()


--- We can probs abandon fixtures for all cases that return wrap(args) functions!


"""



class ReverseGateSetup:

   def __init__(self, parents=None, grads=None, seed=None):

       self._parents = parents
       self._grads   = grads
       self._seed    = seed


   @property
   @tdutil.cacheable
   def parents(self):

       if self._parents is None:
          return tuple()

       if isinstance(self._parents, int):
          return tpl.repeat(fake.ReverseNode, self._parents)

       return self._parents


   @property
   @tdutil.cacheable
   def grads(self):

       if self._grads is None:
          return tpl.repeat(fake.CumFunReturn, self.valency)

       return self._grads


   @property
   @tdutil.cacheable
   def valency(self):

       return len(self.parents)


   @property
   @tdutil.cacheable
   def seed(self):

       if self._seed is None:
          return fake.CumFunReturn()

       return self._seed


   @property
   @tdutil.cacheable
   def fake(self):

       grads = fake.Fun({(self.seed,): self.grads})

       return fake.ReverseGate(
                               parents=self.parents, 
                               grads=grads
                              )  

   @property
   @tdutil.cacheable
   def real(self):
 
       fun = fake.Fun()
       vjp = fake.Fun({(self.seed,): self.grads})

       return tdnode.ReverseGate(self.parents, fun, vjp)




@pytest.fixture
def reverse_gate_factory(): # FIXME could let fixtures take care of setting defaults

    def wrap(*args, **kwargs):
        return ReverseGateSetup(*args, **kwargs)

    return wrap




class ReverseNodeSetup:

   def __init__(self, gate_setup, source=None, layer=None):

       self._gate_setup = gate_setup
       self._source     = source
       self._layer      = seed


   @property
   @tdutil.cacheable
   def source(self):

       return self._source


   @property
   @tdutil.cacheable
   def layer(self):

       return self._layer


   @property
   @tdutil.cacheable
   def nodule(self):

       return tdnode.Nodule(self.source, self.layer)


   @property
   @tdutil.cacheable
   def gate(self):

       return self._gate_setup.fake


   @property
   @tdutil.cacheable
   def fake(self):

       return fake.ReverseNode(gate=self.gate)


   @property
   @tdutil.cacheable
   def fake(self):

       return tdnode.ReverseNode(self.nodule, self.gate)




@pytest.fixture
def reverse_node_factory():

    def wrap(*args, **kwargs):

        gate_setup = ReverseGateSetup(*args, **kwargs)

        return ReverseNodeSetup(gate_setup) 

    return wrap





"""
# TODO probs we should just use factories for fakes, and ctors/fixtures for real objects?

class ReverseGateFactory: # TODO could also make SeededReverseGateFactory decorator...

   def __init__(self, parents=tuple(), seed=None):

       if isinstance(parents, int):
          parents = tpl.repeat(fake.ReverseNode, parents)

       self._parents = parents
       self._seed    = seed


   @property
   @tdutil.cacheable
   def gradfun(self):

       if self._seed is None:
          return lambda seed: self.grads

       return lambda seed: {self._seed: self.grads}[seed]


   @property
   @tdutil.cacheable
   def parents(self):

       return self._parents


   @property
   @tdutil.cacheable
   def grads(self):

       return tpl.repeat(fake.CumFunReturn, self.valency)


   @property
   @tdutil.cacheable
   def valency(self):

       return len(self._parents)


   @property
   @tdutil.cacheable
   def gate(self):

       return fake.ReverseGate(
                               parents=self.parents, 
                               grads=self.gradfun
                              )  




@pytest.fixture
def reverse_gate_factory():

    def wrap(*args, **kwargs):
        return ReverseGateFactory(*args, **kwargs)

    return wrap
"""




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

   @pytest.fixture(autouse=True)
   def request_gate_factory(self, reverse_gate_factory):

       self.gate_factory = reverse_gate_factory


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

 
   """
   def test_attach(self): 

       ans    = fake.NodeTrain()               
       train1 = fake.NodeTrain(with_meta=ans)  
       train2 = fake.NodeTrain(with_node=train1) 

       node = self.node(fake.Nodule(attach={train1: ans}))

       assert node.attach(train2) == ans
   """

   @pytest.mark.parametrize("source, layer", [
      [fake.FunReturn(), fake.FunReturn()]
   ]) 
   def test_attach(self, source, layer):

       node = self.node(tdnode.Nodule(source, layer))

       train  = tdgraph.NodeTrain() # TODO make factory for NodeTrain to automate this
       train1 = tdgraph.NodeTrain(
                                  tdutil.Sequence([node], end=1), 
                                  tdutil.Sequence([(source, layer)], end=1) # FIXME Sequence should have auto-init of _end
                                 )                                          # i.e. we shouldn't be able to pass contradictory data

       assert node.attach(train) == train1 



   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_logic(self, valency):

       adxs, out, args = self.args(valency)

       others = tpl.repeat(fake.ReverseNode, valency-1) 
       node   = self.node()
       ans    = self.logic((node, *others), adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

       
   """    
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
   """


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_accumulate_parent_grads(self, valency):

       x    = self.gate_factory(valency) 
       node = self.node(gate=x.fake)
       
       accum = tdgrad.GradAccum()
       accum.push(node, x.seed)

       node.accumulate_parent_grads(accum) 

       assert tpl.amap(accum.pop, x.parents) == x.grads 


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


















