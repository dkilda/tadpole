#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad
import tests.autodiff.fakes   as fake
import tests.common.ntuple    as tpl

from tests.autodiff.fixtures.misc import (
   node_args,
   op_args,
)




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Forward differential operator ----------------------------------------- #

@pytest.fixture
def forward_diff_op():

    def wrap(fun=None, x=None):
        return tdgrad.ForwardDiffOp(*op_args(fun, x))

    return wrap




# --- Reverse differential operator ----------------------------------------- #

@pytest.fixture
def reverse_diff_op():

    def wrap(fun=None, x=None):
        return tdgrad.ReverseDiffOp(*op_args(fun, x))

    return wrap




###############################################################################
###                                                                         ###
###  Backpropagation through the computation graph.                         ###
###                                                                         ###
###############################################################################


# --- Helpers: create a top reverse node ------------------------------------ #

def _make_top_node(top_node=None, valency=2):

    if top_node is None:

       parents = tpl.repeat(fake.ReverseNode, valency)
       gate    = fake.ReverseGate(parents=parents)

       return fake.ReverseNode(gate=gate)   

    return top_node 




# --- Child-node counter ---------------------------------------------------- #

@pytest.fixture
def childcount():

    def wrap(top_node=None, valency=2):
        return tdgrad.ChildCount(_make_top_node(top_node, valency))

    return wrap




# --- Topological sort ------------------------------------------------------ #

@pytest.fixture
def toposort():

    def wrap(count=None, top_node=None):

        if count is None:
           count = {}

        if top_node is None:
           top_node = fake.ReverseNode()

        return tdgrad.TopoSort(count, top_node)

    return wrap




# --- Gradient accumulation ------------------------------------------------- #

@pytest.fixture
def gradaccum():

    def wrap():
        return tdgrad.GradAccum()

    return wrap




# --- Backpropagation ------------------------------------------------------- # 

@pytest.fixture
def backprop():

    def wrap(top_node=None, valency=2):
        return tdgrad.Backprop(_make_top_node(top_node, valency))

    return wrap



















