#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tests.autodiff.fakes   as fake
import tests.common.ntuple    as tpl

from tests.autodiff.fixtures.misc import (
   node_args,
   op_args,
)




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

@pytest.fixture
def graph():

    def wrap(fun=None, x=None):
        return tdgraph.Graph(*op_args(fun, x))

    return wrap




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Function with gate ---------------------------------------------------- #

@pytest.fixture
def fun_with_gate():

    def wrap(diff_fun=None, raw_fun=None):

        if diff_fun is None:
           diff_fun = fake.Fun()

        if raw_fun is None:
           raw_fun = fake.Fun()

        return tdgraph.FunWithGate(diff_fun, raw_fun)

    return wrap




# --- Differentiable function decorator ------------------------------------- #

@pytest.fixture
def differentiable():

    def wrap(fun=None): 

        if fun is None:
           fun = fake.Fun()

        return tdgraph.Differentiable(fun)

    return wrap




# --- Non-differentiable function decorator --------------------------------- #

@pytest.fixture
def nondifferentiable():

    def wrap(fun=None): 

        if fun is None:
           fun = fake.Fun()

        return tdgraph.NonDifferentiable(fun)

    return wrap




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

@pytest.fixture
def node_train():

    def wrap(nodes=None, meta=None):

        return tdgraph.NodeTrain(nodes, meta)

    return wrap




# --- Node glue ------------------------------------------------------------- #

@pytest.fixture
def node_glue():

    def wrap(args=None):
        return tdgraph.NodeGlue(node_args(args))

    return wrap




###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


# --- Concatenated arguments kernel ----------------------------------------- #

@pytest.fixture
def concat_args_kernel():

    def wrap(nodes=2, sources=None, layers=None):

        if isinstance(nodes, int):
           nodes = tpl.repeat(fake.Node, nodes)

        if sources is None:
           sources = tpl.repeat(fake.Node, len(nodes))

        if layers is None:
           layers = (0,)*len(nodes)

        return tdgraph.ConcatArgsKernel(nodes, sources, layers)

    return wrap




# --- Concatenated arguments ------------------------------------------------ #

@pytest.fixture
def concat_args():

    def wrap(args=None):
        return tdgraph.ConcatArgs(node_args(args))

    return wrap




# --- Active concatenated arguments ----------------------------------------- #

@pytest.fixture
def active():

    def wrap(args=None):
        return tdgraph.Active(node_args(args))

    return wrap




# --- Passive concatenated arguments ---------------------------------------- #

@pytest.fixture
def passive():

    def wrap(args=None):
        return tdgraph.Passive(node_args(args))

    return wrap




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Active pack ----------------------------------------------------------- #

@pytest.fixture
def active_pack():

    def wrap(source=None, layer=0, logic=None):

        if source is None:
           source = fake.Node()

        if logic is None:
           logic = fake.Logic()

        return tdgraph.ActivePack(source, layer, logic)

    return wrap




# --- Passive pack ---------------------------------------------------------- #

@pytest.fixture
def passive_pack():

    def wrap(source=None):

        if source is None:
           source = fake.Node()

        return tdgraph.PassivePack(source)

    return wrap




# --- Point pack ------------------------------------------------------------ #

@pytest.fixture
def point_pack():

    def wrap(nodes=None):
        return tdgraph.PointPack(node_args(nodes))

    return wrap







































