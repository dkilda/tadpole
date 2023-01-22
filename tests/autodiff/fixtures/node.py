#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.node as tdnode
import tests.autodiff.fakes  as fake

from tests.autodiff.fixtures.misc import (
   randn,
   jvpfun_args,
   vjpfun_args,
)




###############################################################################
###                                                                         ###
###  Logic of forward and reverse propagation, creates logic gates.         ###
###                                                                         ###
###############################################################################


# --- Forward logic --------------------------------------------------------- #

@pytest.fixture
def forward_logic(jvpfun_args):

    def wrap(parents=None, valency=2, adxs=None, out=None, args=None):

        adxs, out, args = jvpfun_args(valency, adxs, out, args)

        if parents is None:
           parents = tuple([fake.ForwardNode()]*valency)

        return tdnode.ForwardLogic(parents, adxs, out, args)

    return wrap




# --- Reverse logic --------------------------------------------------------- #

@pytest.fixture
def reverse_logic(vjpfun_args):

    def wrap(parents=None, valency=2, adxs=None, out=None, args=None):

        adxs, out, args = vjpfun_args(valency, adxs, out, args)

        if parents is None:
           parents = tuple([fake.ReverseNode()]*valency)

        return tdnode.ReverseLogic(parents, adxs, out, args)

    return wrap




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Forward logic gate ---------------------------------------------------- #

@pytest.fixture
def forward_gate(randn):

    def wrap(valency=2, fun=None, grad=None):

        if grad is None:
           grad = randn()

        if fun is None:
           fun = fake.Fun()

        parents = tuple([fake.ForwardNode()]*valency)

        return tdnode.ForwardGate(parents, fun, grad)

    return wrap 




# --- Reverse logic gate ---------------------------------------------------- #

@pytest.fixture
def reverse_gate():

    def wrap(valency=2, fun=None, vjp=None):

        if vjp is None:
           vjp = fake.Fun(valency=valency)

        if fun is None:
           fun = fake.Fun()

        parents = tuple([fake.ReverseNode()]*valency)

        return tdnode.ReverseGate(parents, fun, vjp)

    return wrap 




###############################################################################
###                                                                         ###
###  Nodes of the autodiff computation graph                                ###
###                                                                         ###
###############################################################################


# --- Nodule: a node kernel ------------------------------------------------- #

@pytest.fixture
def nodule():

    def wrap(source=None, layer=0):

        if source is None and layer == 0:
           source = fake.Point()

        if source is None and layer > 0:
           source = fake.ActiveNode()

        return tdnode.Nodule(source, layer) 

    return wrap




# --- Forward node ---------------------------------------------------------- #

@pytest.fixture
def forward_node():

    def wrap(nodule=None, gate=None):

        if nodule is None:
           nodule = fake.Nodule()

        if gate is None:
           gate = fake.ForwardGate()

        return tdnode.ForwardNode(nodule, gate) 
                                  
    return wrap




# --- Reverse node ---------------------------------------------------------- #

@pytest.fixture
def reverse_node():

    def wrap(nodule=None, gate=None):

        if nodule is None:
           nodule = fake.Nodule()

        if gate is None:
           gate = fake.ReverseGate()

        return tdnode.ReverseNode(nodule, gate) 
                                  
    return wrap




# --- Point (a disconnected node, only carries a value and no logic) -------- #

@pytest.fixture
def point():

    def wrap(source=None):

        if source is None:
           source = fake.Point()

        return tdnode.Point(source)
                                  
    return wrap




