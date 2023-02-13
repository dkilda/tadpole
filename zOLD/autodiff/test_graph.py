#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.node     as tdnode
import tadpole.autodiff.graph    as tdgraph
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class TestGraph:

  
   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_graph(self, graph):

       self.graph = graph


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   # --- Helpers --- #

   def _setup_forward(self, x, layer):

       nodule = self.nodule(tdnode.nodify(x), layer)
       node   = fake.ForwardNode()
       gate   = fake.ForwardGate(nodify=fake.Map({nodule: node}))

       ans = fake.ForwardNode()
       fun = fake.Fun({(node,): ans})

       return fun, gate, ans


   def _setup_reverse(self, x, layer):

       nodule = self.nodule(tdnode.nodify(x), layer)
       node   = fake.ReverseNode()
       gate   = fake.ReverseGate(nodify=fake.Map({nodule: node}))

       ans = fake.ReverseNode()
       fun = fake.Fun({(node,): ans})

       return fun, gate, ans


   # --- Tests: minilayer --- #

   def test_minilayer(self):

       assert tdgraph.minlayer() == -1


   # --- Tests: build --- #

   @pytest.mark.parametrize("x", [
      fake.ForwardNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])   
   def test_build_forward(self, x): 

       fun, gate, ans = self._setup_forward(x, -1)  
  
       graph = self.graph(fun, x)
       assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])   
   def test_build_reverse(self, x):      # FIXME make_node() should be more testable: 
                                         # the internally created and never returned Nodule 
                                         # makes it impossible to test using fakes alone
       fun, gate, ans = self._setup_reverse(x, -1)  

       graph = self.graph(fun, x)
       assert graph.build(gate) == ans


   # --- Tests: enter --- #

   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])   
   def test_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ]) 
   def test_nested_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 1) 

       with self.graph() as graph1:
          with self.graph(fun, x) as graph:
             assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])  
   def test_extra_nested_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 2) 

       with self.graph() as graph2:
          with self.graph() as graph1:
             with self.graph(fun, x) as graph:
                assert graph.build(gate) == ans


   # --- Tests: exit --- #

   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])  
   def test_nested_exit(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          with self.graph() as graph1:
             pass
          assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
      fake.ReverseNode(), 
      fake.Point(), 
      fake.FunReturn(),
   ])  
   def test_extra_nested_exit(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          with self.graph() as graph1:
             with self.graph() as graph2:
                pass
          assert graph.build(gate) == ans




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Function with gate ---------------------------------------------------- #

class TestFunWithGate:

   @pytest.fixture(autouse=True)
   def request_fun_with_gate(self, fun_with_gate):

       self.fun_with_gate = fun_with_gate


   @pytest.mark.parametrize("valency", [0,1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()

       fun = self.fun_with_gate(raw_fun=fake.Fun({args: ans}))
 
       assert fun(*args) == ans


   @pytest.mark.parametrize("valency", [0,1,2,3])
   def test_gate(self, valency):

       gate  = fake.ReverseGate()
       logic = fake.ReverseLogic(gate=gate)

       fun     = fake.Fun(valency)
       vjpfuns = tpl.repeat(fake.Fun, valency) 

       tda.vjpmap.add(fun, *vjpfuns)

       assert self.fun_with_gate(diff_fun=fun).gate(logic) == gate




# --- Differentiable function decorator ------------------------------------- #

class TestDifferentiable:

   @pytest.fixture(autouse=True)
   def request_differentiable(self, differentiable):

       self.differentiable = differentiable


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()
       fun  = self.differentiable(fake.Fun({args: ans}))

       assert fun(*args) == ans




# --- Non-differentiable function decorator --------------------------------- #

class TestNonDifferentiable:

   @pytest.fixture(autouse=True)
   def request_nondifferentiable(self, nondifferentiable):

       self.nondifferentiable = nondifferentiable


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()
       fun  = self.nondifferentiable(fake.Fun({args: ans}))

       assert fun(*args) == ans       




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

class TestNodeTrain:

   @pytest.fixture(autouse=True)
   def request_node_train(self, node_train):

       self.node_train = node_train


   @pytest.mark.parametrize("node", [
      fake.ForwardNode(), 
      fake.ReverseNode(), 
      fake.Point(),
   ])   
   def test_with_node(self, node):

       ans   = fake.Sequence()
       nodes = fake.Sequence(push={node: ans})

       train  = self.node_train(nodes=nodes)
       train1 = self.node_train(nodes=ans)

       assert train.with_node(node) == train1


   @pytest.mark.parametrize("source", [
      fake.ForwardNode(), 
      fake.ReverseNode(), 
      fake.Point(),
   ])  
   @pytest.mark.parametrize("layer", [0,1,2]) 
   def test_with_meta(self, source, layer):

       ans  = fake.Sequence()
       meta = fake.Sequence(push={(source, layer): ans})

       train  = self.node_train(meta=meta)
       train1 = self.node_train(meta=ans)

       assert train.with_meta(source, layer) == train1


   @pytest.mark.parametrize("nodes, sources, layers", [
      [
       tuple(), 
       tuple(), 
       tuple(),
      ],
      [
       (fake.ReverseNode(), ),
       (fake.Point(),       ),
       (1,                  ),
      ],
      [
       (fake.ReverseNode(), fake.Point(), fake.ReverseNode()),
       (fake.Point(),       fake.Point(), fake.ReverseNode()), 
       (0,1,1),
      ],
   ])
   def test_concatenate(self, nodes, sources, layers):

       ans = tdgraph.ConcatArgsKernel(nodes, sources, layers)

       size  = len(nodes)
       nodes = fake.Sequence(iterate=iter(nodes),                size=size)
       meta  = fake.Sequence(iterate=iter(zip(sources, layers)), size=size)
       train = self.node_train(nodes, meta)

       assert train.concatenate() == ans




# --- Node glue ------------------------------------------------------------- #

class TestNodeGlue:

   @pytest.fixture(autouse=True)
   def request_node_glue(self, node_glue):

       self.node_glue = node_glue


   @pytest.fixture(autouse=True)
   def request_concat_args_kernel(self, concat_args_kernel):

       self.concat_args_kernel = concat_args_kernel


   @pytest.mark.parametrize("args", [
      tuple(),
      (fake.Node(),                                        ),
      (fake.Node(),        fake.Node()                     ),
      (fake.FunReturn(),   fake.FunReturn()                ),
      (fake.ReverseNode(), fake.ReverseNode(), fake.Point()),
      (fake.ReverseNode(), fake.FunReturn(),   fake.Point()),
   ])  
   def test_iterate(self, args):

       ans  = [tdnode.Point(x) if isinstance(x, fake.FunReturn) 
                               else x for x in args]
       glue = self.node_glue(args)

       assert list(glue.iterate()) == ans


   @pytest.mark.parametrize("valency", [0,1,2])
   def test_concatenate(self, valency):

       ans   = fake.ConcatArgs()  
       train = fake.NodeTrain(concatenate=ans) 
       args  = tpl.repeat(lambda: fake.ReverseNode(
                                     attach=fake.TrivMap(train)), valency)

       glue = self.node_glue(args) # FIXME testability is worse cuz 
                                   #       we create NodeTrain inside instead of injecting it!
       if   valency == 0:
            assert glue.concatenate() == self.concat_args_kernel(0)
       else:
            assert glue.concatenate() == ans




###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


# --- Concatenated arguments kernel ----------------------------------------- #

def parametrize_concat_args_kernel():

    labels = iter(range(5))

    return pytest.mark.parametrize("label, nodes, sources, layers", [
      [
       next(labels),
       tpl.repeat(fake.Node, 2),
       tpl.repeat(fake.Node, 2),
       (0,0),
      ],
      [
       next(labels),
       (fake.ReverseNode(), ),
       (fake.Point(),       ),
       (1,                  ),
      ],
      [
       next(labels),
       (fake.ReverseNode(), fake.Point(), fake.ReverseNode()),
       (fake.ReverseNode(), fake.Point(), fake.Point()), 
       (0,-1,1),
      ],
      [
       next(labels),
       (fake.ReverseNode(), fake.ReverseNode(), fake.Point()),
       (fake.ReverseNode(), fake.Point(),       fake.Point()), 
       (1,2,-1),
      ],
      [
       next(labels),
       (fake.ReverseNode(), fake.Point(), fake.ReverseNode()),
       (fake.Point(),       fake.Point(), fake.ReverseNode()), 
       (0,-1,0),
      ],
   ])




class TestConcatArgsKernel:

   @pytest.fixture(autouse=True)
   def request_concat_args_kernel(self, concat_args_kernel):

       self.concat_args_kernel = concat_args_kernel


   @parametrize_concat_args_kernel()
   def test_layer(self, label, nodes, sources, layers):

       ans  = (0, 1, 1, 2, 0)[label]
       args = self.concat_args_kernel(nodes, sources, layers)
       assert args.layer() == ans


   @parametrize_concat_args_kernel() 
   def test_adxs(self, label, nodes, sources, layers):

       ans  = ((0,1), (0,), (2,), (1,), (0,2))[label]
       args = self.concat_args_kernel(nodes, sources, layers)
       assert args.adxs() == ans


   @parametrize_concat_args_kernel() 
   def test_parents(self, label, nodes, sources, layers):

       ans = (
              lambda: (nodes[0], nodes[1]),
              lambda: (nodes[0],         ),
              lambda: (nodes[2],         ),
              lambda: (nodes[1],         ),
              lambda: (nodes[0], nodes[2]),
             )[label]()
       args = self.concat_args_kernel(nodes, sources, layers)
       assert args.parents() == ans


   @parametrize_concat_args_kernel() 
   def test_deshell(self, label, nodes, sources, layers):

       ans = (
              lambda: (sources[0], sources[1]            ),
              lambda: (sources[0],                       ),
              lambda: (nodes[0],   nodes[1],   sources[2]),
              lambda: (nodes[0],   sources[1], nodes[2]  ),
              lambda: (sources[0], nodes[1],   sources[2]),
             )[label]()
       args = self.concat_args_kernel(nodes, sources, layers)
       assert args.deshell() == ans


   def test_deshelled(self):

       ans   = fake.ConcatArgs()  
       train = fake.NodeTrain(concatenate=ans) 

       nA = fake.ReverseNode()
       nB = fake.Point(attach=fake.TrivMap(train))
       nC = fake.ReverseNode()

       sA = fake.Point(attach=fake.TrivMap(train))
       sB = fake.Point()
       sC = fake.ReverseNode(attach=fake.TrivMap(train))

       args = self.concat_args_kernel(
                                      (nA, nB, nC), 
                                      (sA, sB, sC), 
                                      (1, -1,  1)
                                     )

       assert args.deshelled() == ans                                     




# --- Concatenated arguments ------------------------------------------------ #

class TestConcatArgs:

   @pytest.fixture(autouse=True)
   def request_concat_args(self, concat_args):

       self.concat_args = concat_args


   def _setup(self, valency, *args, **kwargs):

       ans   = fake.ConcatArgs(*args, **kwargs)  
       train = fake.NodeTrain(concatenate=ans) 
       args  = tpl.repeat(lambda: fake.Node(
                                  attach=fake.TrivMap(train)), valency) # FIXME put this in the fixture?

       return self.concat_args(args)


   @pytest.mark.parametrize("valency, layer", [
      [2, 0], 
      [1, 1], 
      [3, 1], 
      [3, 2], 
      [3, 0],
   ])
   def test_layer(self, valency, layer):

       args = self._setup(valency, layer)

       assert args.layer() == layer


   @pytest.mark.parametrize("valency, adxs", [
      [2, (0,1)], 
      [1, (0,) ], 
      [3, (2,) ], 
      [3, (1,) ], 
      [3, (0,2)],
   ])
   def test_adxs(self, valency, adxs):

       args = self._setup(valency, adxs=adxs)

       assert args.adxs() == adxs


   @pytest.mark.parametrize("valency, parents", [
      [2, (fake.Node(), fake.Node())], 
      [1, (fake.Node(),            )], 
      [3, (fake.Node(),            )], 
      [3, (fake.Node(),            )], 
      [3, (fake.Node(), fake.Node())],
   ])
   def test_parents(self, valency, parents):

       args = self._setup(valency, parents=parents)

       assert args.parents() == parents


   @pytest.mark.parametrize("valency", [2,1,3,3,3])
   def test_deshell(self, valency):

       deshell = tpl.repeat(fake.Node, valency)
       args    = self._setup(valency, deshell=deshell)

       assert args.deshell() == deshell


   @pytest.mark.parametrize("valency", [2,1,3,3,3])
   def test_deshelled(self, valency):

       deshell = tpl.repeat(fake.Node, valency)
       args    = self._setup(valency, deshell=deshell)

       assert args.deshelled() == self.concat_args(deshell)




# --- Active concatenated arguments ----------------------------------------- #

class TestActive:

   @pytest.fixture(autouse=True)
   def request_active(self, active):

       self.active = active


   @pytest.fixture(autouse=True)
   def request_active_pack(self, active_pack):

       self.active_pack = active_pack


   @pytest.fixture(autouse=True)
   def request_point_pack(self, point_pack):

       self.point_pack = point_pack


   @pytest.mark.parametrize("valency, layer, adxs, parents", [
      [2, 0, (0,1), 2], 
      [1, 1, (0,),  1], 
      [3, 1, (2,),  1], 
      [3, 2, (1,),  1], 
      [3, 0, (0,2), 2],
   ])
   def test_pack(self, valency, layer, adxs, parents):

       logic   = fake.ReverseLogic()
       parents = tpl.repeat(lambda: fake.ReverseNode(logic=logic), parents) 
       deshell = tpl.repeat(fake.Node, valency)
   
       nodes     = tpl.repeat(fake.Node, valency) 
       source    = self.point_pack(nodes) # FIXME difficult to test cuz PointPack is created internally and never returned directly
       deshelled = fake.Active(adxs=tuple(), deshell=nodes, pack=source)

       args = self.active(fake.ConcatArgs(layer, adxs, 
                                          parents, deshell, deshelled))
       ans  = self.active_pack(source, layer, logic) 

       assert args.pack() == ans


   @pytest.mark.parametrize("valency, layer, adxs, parents", [
      [2, 0, (0,1), 2], 
      [1, 1, (0,),  1], 
      [3, 1, (2,),  1], 
      [3, 2, (1,),  1], 
      [3, 0, (0,2), 2],
   ])
   def test_default_pack(self, valency, layer, adxs, parents):

       parents = tpl.repeat(fake.ReverseNode, parents) 
       deshell = tpl.repeat(fake.Node, valency)

       nodes     = tpl.repeat(fake.Node, valency) 
       source    = fake.ActivePack(nodes) 
       deshelled = fake.Active(adxs=tuple(), deshell=nodes, pack=source)

       args = fake.Active(layer, adxs, parents, deshell, deshelled)
       ans  = fake.FunReturn()
       fun  = fake.Fun(out={(args, source): ans}) # TODO replace with fake.Fun.call(args1, source1), register .called = (args1, source1)
                                                  #      then fun.verify_call(args, source): assert (args, source) == (args1, source1)
       assert tdgraph.default_pack(fun)(args) == ans




# --- Passive concatenated arguments ---------------------------------------- #

class TestPassive:

   @pytest.fixture(autouse=True)
   def request_passive(self, passive):

       self.passive = passive


   @pytest.fixture(autouse=True)
   def request_passive_pack(self, passive_pack):

       self.passive_pack = passive_pack


   @pytest.fixture(autouse=True)
   def request_point_pack(self, point_pack):

       self.point_pack = point_pack


   @pytest.mark.parametrize("valency, layer, adxs, parents", [
      [2, 0, (0,1), 2], 
      [1, 1, (0,),  1], 
      [3, 1, (2,),  1], 
      [3, 2, (1,),  1], 
      [3, 0, (0,2), 2],
   ])
   def test_pack(self, valency, layer, adxs, parents):

       parents = tpl.repeat(fake.ReverseNode, parents) 
       deshell = tpl.repeat(fake.Node, valency)
   
       nodes     = tpl.repeat(fake.Node, valency) 
       source    = self.point_pack(nodes) # FIXME difficult to test cuz PointPack is created internally and never returned directly
       deshelled = fake.Passive(adxs=tuple(), deshell=nodes, pack=source)

       args = self.passive(fake.ConcatArgs(layer, adxs, 
                                           parents, deshell, deshelled))
       ans  = self.passive_pack(source) 

       assert args.pack() == ans


   @pytest.mark.parametrize("valency, layer, adxs, parents", [
      [2, 0, (0,1), 2], 
      [1, 1, (0,),  1], 
      [3, 1, (2,),  1], 
      [3, 2, (1,),  1], 
      [3, 0, (0,2), 2],
   ])
   def test_default_pack(self, valency, layer, adxs, parents):

       parents = tpl.repeat(fake.ReverseNode, parents) 
       deshell = tpl.repeat(fake.Node, valency)

       nodes     = tpl.repeat(fake.Node, valency) 
       source    = fake.PassivePack(nodes) 
       deshelled = fake.Passive(adxs=tuple(), deshell=nodes, pack=source)

       args = fake.Passive(layer, adxs, parents, deshell, deshelled)
       ans  = fake.FunReturn()
       fun  = fake.Fun(out={(args, source): ans}) # TODO replace with fake.Fun.call(args1, source1), register .called = (args1, source1)
                                                  #      then fun.verify_call(args, source): assert (args, source) == (args1, source1)
       assert tdgraph.default_pack(fun)(args) == ans




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Active pack ----------------------------------------------------------- #

class TestActivePack:

   @pytest.fixture(autouse=True)
   def request_pack(self, active_pack):

       self.pack = active_pack


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.fixture(autouse=True)
   def request_reverse_node(self, reverse_node):

       self.reverse_node = reverse_node


   @pytest.fixture(autouse=True)
   def request_forward_node(self, forward_node):

       self.forward_node = forward_node


   @pytest.mark.parametrize("srctype", [fake.ActivePack, fake.PointPack])
   @pytest.mark.parametrize("layer",   [0, 1])
   def test_reverse_pluginto(self, srctype, layer):

       ans    = fake.Node()   
       nodule = self.nodule(ans, layer)
       gate   = fake.ReverseGate(nodify=fake.Map({nodule: ans}))  

       logic = fake.ReverseLogic()
       fun   = fake.FunWithGate(gate={logic: gate})
       pack  = self.pack(
                         srctype(pluginto={fun: ans}), # FIXME there's a tight coupling between ActivePack.pluginto()
                         layer,                        # and tdnode.make_node(), cuz we've to mock out the impl of make_node()
                         logic                         # to test pluginto() -- i.e. the output of pluginto() is strongly coupled
                        )                              # to that of make_node(). Ideally, we would just pass a fake mock_node() to
                                                       # ActivePack instead! Try to fix it!
       assert pack.pluginto(fun) == ans


   @pytest.mark.parametrize("srctype", [fake.ActivePack, fake.PointPack])
   @pytest.mark.parametrize("layer", [0,1])
   def test_forward_pluginto(self, srctype, layer):

       ans    = fake.Node()
       nodule = self.nodule(ans, layer)
       gate   = fake.ForwardGate(nodify=fake.Map({nodule: ans}))   

       logic = fake.ForwardLogic()
       fun   = fake.FunWithGate(gate={logic: gate})
       pack  = self.pack(
                         srctype(pluginto={fun: ans}), 
                         layer, 
                         logic
                        )

       assert pack.pluginto(fun) == ans




# --- Passive pack ---------------------------------------------------------- #

class TestPassivePack:

   @pytest.fixture(autouse=True)
   def request_pack(self, passive_pack):

       self.pack = passive_pack


   @pytest.mark.parametrize("srctype", [fake.PassivePack, fake.PointPack])
   def test_pluginto(self, srctype):

       ans  = fake.Node()
       fun  = fake.Fun()
       pack = self.pack(srctype(pluginto={fun: ans}))

       assert pack.pluginto(fun) == ans




# --- Point pack ------------------------------------------------------------ #

class TestPointPack:

   @pytest.fixture(autouse=True)
   def request_pack(self, point_pack):

       self.pack = point_pack


   @pytest.mark.parametrize("valence", [0,1,2])
   def test_reverse_pluginto(self, valence):

       vals  = tpl.repeat(fake.FunReturn, valence)
       nodes = tpl.amap(lambda x: fake.Node(tovalue=x), vals)

       ans  = fake.FunReturn()
       fun  = fake.Fun(out={vals: ans})
       pack = self.pack(nodes)  

       assert pack.pluginto(fun) == ans























