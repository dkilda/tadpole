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

import tadpole.autodiff.wrappers   as aw 
import tadpole.autodiff.adjointmap as adj 




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

GraphData = collections.namedtuple("GraphData", [
               "evalop", "graph", "layer", "root", 
               "fun", "x", "out", "start", "end",
            ])




def graph_dat(which="REVERSE", layer=None, 
              x=None, out=None, start=None, end=None):

    root = {
            "REVERSE": an.GateReverse,
            "FORWARD": an.GateForward,
           }[which]()

    if layer is None:
       layer = 0

    if x is None:
       x = fake.Value()

    if out is None:
       out = fake.Value()

    if start is None:
       start = an.node(x, layer, root) 

    if end is None:
       end = an.node(out, layer, fake.Gate())

    fun    = fake.Fun(end, start)
    graph  = ag.Graph(root)
    evalop = ad.EvalOp(fun, x)

    return GraphData(evalop, graph, layer, root, 
                     fun, x, out, start, end)




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable/NonDifferentiable function wrappers -------------------- #

FunWrapData = collections.namedtuple("FunWrapData", [
                 "funwrap",        "fun", 
                 "make_envelope",  "envelope", 
                 "args",           "out"
              ])




def differentiable_funwrap_dat(args, vjpmap=None, jvpmap=None):

    if vjpmap is None:
       vjpmap = aw._VJPMAP

    if jvpmap is None:
       jvpmap = aw._JVPMAP

    out = fake.Node()

    applywrap     = fake.Fun(out)
    envelope      = fake.Envelope(applywrap=applywrap)
    make_envelope = fake.Fun(envelope, *args)

    fun     = fake.Fun(None)
    funwrap = ag.Differentiable(fun, make_envelope, vjpmap, jvpmap)

    applywrap.update_args(funwrap, fun)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




def nondifferentiable_funwrap_dat(args):

    out = fake.Node()

    applyfun      = fake.Fun(out)
    envelope      = fake.Envelope(**{"apply": applyfun})
    make_envelope = fake.Fun(envelope, *args)

    fun     = fake.Fun(None)
    funwrap = ag.NonDifferentiable(fun, make_envelope)
 
    applyfun.update_args(fun)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation.                            ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

ArgsData = collections.namedtuple("ArgsData", [
              "args", "concat", "pack",
              "adxs", 
              "nodes", "sources", "layers", "values", "rawargs",
           ])




def args_dat(n=1, adxs=(0,), layers=(0,)):

    info   = []
    layers = list(reversed(layers))

    for i in range(n):

        value = fake.Value()

        if   i in adxs: 
             source = fake.Node()
             layer  = layers.pop()
             node   = an.node(source, layer, fake.Gate())
             rawarg = node

        else:
             source = value
             layer  = misc.minlayer()           
             node   = an.point(value)
             rawarg = value

        info.append([rawarg, node, source, layer, value])

    rawargs, nodes, sources, layers, values = zip(*info)

    args   = ag.ArgsGen(*rawargs)
    concat = ag.ConcatArgs(
                           util.Sequence(nodes), 
                           util.Sequence(sources), 
                           util.Sequence(layers)
                          )
    pack = ag.PackArgs(concat)

    return ArgsData(
                    args,   concat, pack,
                    adxs,   nodes,  sources, 
                    layers, values, rawargs
                   )




def nodeargs_dat(n=1, layers=None):

    if layers is None:
       layers = arepeat(0, n)

    return args_dat(n, range(n), layers)



      
# --- Concatenation of nodes ------------------------------------------------ #

ConcatData = collections.namedtuple("ConcatData", [
                "concat", "concat_history", "attach_history", 
                "nodes", "sources", "layers",
             ])




def concat_dat(n, adxs, layers):

    x = args_dat(n, adxs, layers)

    attach_history = list(zip(x.nodes, x.sources, x.layers))
    concat_history = []

    for size in range(n):

        _nodes   = util.Sequence(x.nodes[:size])
        _sources = util.Sequence(x.sources[:size])
        _layers  = util.Sequence(x.layers[:size])

        _concat  = ag.ConcatArgs(_nodes, _sources, _layers)

        concat_history.append(_concat)
         
    nodes   = util.Sequence(x.nodes)
    sources = util.Sequence(x.sources)
    layers  = util.Sequence(x.layers)

    concat  = ag.ConcatArgs(nodes, sources, layers)

    return ConcatData(concat, concat_history, attach_history, 
                      nodes, sources, layers)




ConcatOutputData = collections.namedtuple("ConcatOutputData", [
                      "concat", "layer", "adxs", "parents", "deshell",
                   ])




def concat_output_dat(n, adxs, layers):

    x = args_dat(n, adxs, layers)

    def case_000():
        return (
           0, (0,  ), (x.nodes[0], ), (x.sources[0], )
        )

    def case_001():
        return (
           1, (0,  ), (x.nodes[0], ), (x.sources[0], x.nodes[1])
        )

    def case_002():
        return (
           0, (1,  ), (x.nodes[1], ), (x.nodes[0], x.sources[1])
        )

    def case_003():
        return (
           0, (0, 1), (x.nodes[0], x.nodes[1]), (x.sources[0], x.sources[1])
        )

    def case_004():
        return (
           2, (1,  ), (x.nodes[1], ), (x.nodes[0], x.sources[1])
        )

    def case_005():
        return (
           1, (2,  ), (x.nodes[2], ), (x.nodes[0], x.nodes[1], x.sources[2])
        )

    def case_006():
        return (
           2, (0,  ), (x.nodes[0], ), (x.sources[0], x.nodes[1], x.nodes[2])
        )

    def case_007():
        return (
           1, (0, 2), (x.nodes[0], x.nodes[2]), (x.sources[0], x.nodes[1], x.sources[2])
        )

    def case_008():
        return (
           -1, tuple(), tuple(), (x.sources[0], x.sources[1])
        )

    case = {
            (1,  (0,),    (0,)    ): case_000,
            (2,  (0,),    (1,)    ): case_001,
            (2,  (1,),    (0,)    ): case_002,
            (2,  (0,1),   (0,0)   ): case_003,
            (2,  (0,1),   (0,2)   ): case_004,
            (3,  (0,2),   (0,1)   ): case_005,
            (3,  (0,2),   (2,1)   ): case_006,
            (3,  (0,2),   (1,1)   ): case_007,
            (2,  tuple(), tuple() ): case_008,
           }[n, adxs, layers]()


    return ConcatOutputData(
              x.concat, 
              case[0], 
              case[1], 
              an.ParentsGen(*case[2]), 
              ag.ArgsGen(*case[3])
           )




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack (of concatenated nodes) --------------------------------- #

PackData = collections.namedtuple("PackData", [
              "pack",  "deshelled", "deshell", "concat", 
              "nodes", "sources",   "funwrap", 
           ])




def pack_dat(valency=1, layer=0):
 
    deshelled_pack = fake.Pack()
    deshelled_args = fake.Args(pack=fake.Fun(deshelled_pack))

    if   layer == misc.minlayer():
         source  = fake.Value()
         node    = an.point(source)         
    else:
         source  = fake.Node()
         node    = fake.Node()

    funwrap = fake.Fun(None)
    adxs    = arepeat(fake.Value, valency)
    op      = an.AdjointOpGen(funwrap, adxs, source, deshelled_args)
    parents = fake.Parents(next=fake.Fun(node, source, layer, op))  

    concat = fake.Cohesive(
                           layer=fake.Fun(layer),
                           adxs=fake.Fun(adxs),
                           parents=fake.Fun(parents),
                           deshell=fake.Fun(deshelled_args)
                          )
    pack = ag.PackArgs(concat)

    return PackData(
                    pack, 
                    deshelled_pack, 
                    deshelled_args, 
                    concat, 
                    node, 
                    source, 
                    funwrap
                   )




# --- Argument envelope ----------------------------------------------------- #

EnvelopeData = collections.namedtuple("EnvelopeData", [
                  "envelope", "args", 
                  "packs",    "nodes", 
                  "outnodes", "outvalues", 
                  "fun",      "funwrap"
               ])




def envelope_dat(stackdat):

    def make_packs():

        nodes = stackdat.nodes 
        packs = []

        for _nodes in nodes:

           _args  = ag.ArgsGen(*_nodes)
           _packs = _args.pack()

           packs.append(_packs)

        return tuple(reversed(packs))

    packs = make_packs()

    outnode  = stackdat.outnodes[-1]
    args     = ag.ArgsGen(*stackdat.nodes[-1])
    envelope = ag.EnvelopeArgs(args)

    return EnvelopeData(
                        envelope, 
                        args, 
                        packs, 
                        stackdat.nodes, 
                        outnode, 
                        stackdat.outvalue, 
                        stackdat.fun, 
                        stackdat.funwrap
                       )




# --- Node stack ------------------------------------------------------------ #

NodeStackData = collections.namedtuple("NodeStackData", [
                   "outnodes", "outparents", "outvalue",
                   "nodes",    "parents",    "values", 
                   "layers", 
                   "adxs", 
                   "fun", 
                   "funwrap",
                ])




def node_stack_dat_001(gatetype="REVERSE"):

    def gate(parents, op=None):

        if op is None:
           op = fake.AdjointOp()

        parents = an.ParentsGen(*parents)

        return {
                "REVERSE": an.GateReverse, 
                "FORWARD": an.GateForward,
               }[gatetype](parents, op)


    # --- Input nodes --- #

    values = arepeat(fake.Value, 3)

    layersA = (-1, -1, -1)
    layersB = (-1,  0, -1)
    layersC = (-1,  1,  1)
    layersD = ( 2,  1,  1)

    adxsA = tuple()
    adxsB = (1,)
    adxsC = (1,2)
    adxsD = (0,)

    parentsA = (tuple(),               tuple(),               tuple())
    parentsB = (tuple(),               arepeat(fake.Node, 3), tuple()) 
    parentsC = (parentsB[0],           arepeat(fake.Node, 1), arepeat(fake.Node, 2))
    parentsD = (arepeat(fake.Node, 2), parentsC[1],           parentsC[2])

    nodesA = amap(an.point, values)
    nodesB = (
              nodesA[0],  
              an.node(nodesA[1], layersB[1], gate(parentsB[1])), 
              nodesA[2], 
             )
    nodesC = (
              nodesB[0],
              an.node(nodesB[1], layersC[1], gate(parentsC[1])),          
              an.node(nodesB[2], layersC[2], gate(parentsC[2])),       
             )
    nodesD = (
              an.node(nodesC[0], layersD[0], gate(parentsD[0])),  
              nodesC[1],
              nodesC[2],
             )

    nodes   = (nodesA,   nodesB,   nodesC,   nodesD)
    parents = (parentsA, parentsB, parentsC, parentsD)
    adxs    = (adxsA,    adxsB,    adxsC,    adxsD)
    layers  = (layersA,  layersB,  layersC,  layersD)


    # --- Output node --- #

    outvalue = fake.Value()

    outparentsA = tuple()
    outparentsB = (nodesB[1],          )
    outparentsC = (nodesC[1], nodesC[2])
    outparentsD = (nodesD[0],          )

    fun     = fake.Fun(outvalue, *values)
    funwrap = fake.Fun(None)

    outnodeA = an.point(outvalue)
    outnodeB = an.node(outnodeA, 0, gate(outparentsB, an.AdjointOpGen(funwrap, adxsB, outnodeA, ag.ArgsGen(*nodesA)))) 
    outnodeC = an.node(outnodeB, 1, gate(outparentsC, an.AdjointOpGen(funwrap, adxsC, outnodeB, ag.ArgsGen(*nodesB))))
    outnodeD = an.node(outnodeC, 2, gate(outparentsD, an.AdjointOpGen(funwrap, adxsD, outnodeC, ag.ArgsGen(*nodesC))))

    outnodes   = (outnodeA,    outnodeB,    outnodeC,    outnodeD)
    outparents = (outparentsA, outparentsB, outparentsC, outparentsD)

    return NodeStackData(
                         outnodes, outparents, outvalue, 
                         nodes,    parents,    values,
                         layers,   adxs,
                         fun,      funwrap,
                        )




def node_stack_dat_002(gatetype="REVERSE"):

    # --- Input nodes --- #

    values = arepeat(fake.Value, 3)

    layersA  = (-1, -1, -1)
    adxsA    = tuple()
    parentsA = (tuple(), tuple(), tuple())
    nodesA   = amap(an.point, values)

    nodes   = (nodesA,   )
    parents = (parentsA, )
    adxs    = (adxsA,    )
    layers  = (layersA,  )

    # --- Output node --- #

    outvalue    = fake.Value()
    outparentsA = tuple()
    outnodeA    = an.point(outvalue)

    outnodes   = (outnodeA,    )
    outparents = (outparentsA, )

    fun     = fake.Fun(outvalue, *values)
    funwrap = fake.Fun(None)

    return NodeStackData(
                         outnodes, outparents, outvalue, 
                         nodes,    parents,    values,
                         layers,   adxs,
                         fun,      funwrap,
                        )




