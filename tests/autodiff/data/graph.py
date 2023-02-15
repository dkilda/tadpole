#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tests.autodiff.data.node as nd

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

GraphData = collections.namedtuple("GraphData", ["graph", "root"])




def graph(which="REVERSE"):

    root = {
            "REVERSE": tdnode.ReverseGate,
            "FORWARD": tdnode.ForwardGate,
           }[which]()

    graph = tdgraph.Graph(root)

    return GraphData(graph, root)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

ArgsData = collections.namedtuple("ArgsData", [
              "args", "adxs", "nodes", "sources", "layers", "values"
           ])




def args(n=1, adxs=(0,), layers=(0,)):

    info   = []
    layers = reversed(layers)

    for i in range(n):

        value = fake.Value()

        if   i in adxs: 
             source = fake.NodeLike(tovalue=fake.Fun(value))
             layer  = layers.pop()
             node   = td.Node(source, layer, fake.GateLike())

        else:
             layer  = tdgraph.minlayer()           
             node   = td.Point(value)
             source = node

        info.append([node, source, layer, value])

    nodes, sources, layers, values = zip(*info)
    args = tdgraph.Args(args)

    return ArgsData(args, adxs, nodes, sources, layers, values)




def nodeargs(n=1, layers=None):

    if layers is None:
       layers = common.arepeat(0, n)

    return args(n, range(n), layers)



      
# --- Concatenation of nodes ------------------------------------------------ #

ConcatenationData = collections.namedtuple("ConcatenationData", [
                       "concat", "nodes", "sources", "layers"
                    ])




def concatenation(args):

    n = len(args.nodes)

    nodes   = tdutil.Sequence(args.nodes,   n)
    sources = tdutil.Sequence(args.sources, n)
    layers  = tdutil.Sequence(args.layers,  n)

    concat = tdgraph.Concatenation(nodes, sources, layers)

    return ConcatenationData(concat, nodes, sources, layers)




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack and envelope -------------------------------------------- #

PackData = collections.namedtuple("PackData", [
              "envelope", "pack", "concat", "args"
           ])




def pack(args):

    concat   = concatenation(args.args)
    pack     = pack(concat)
    envelope = tdgraph.Envelope(args.args)

    return EnvelopeData(envelope, pack, concat, args.args)




