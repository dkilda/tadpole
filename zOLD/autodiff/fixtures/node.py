#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.node as tdnode
import tests.autodiff.fakes  as fake
import tests.common.ntuple   as tpl

from tests.autodiff.fixtures.misc import (
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

    def wrap(parents=2, adxs=None, out=None, args=None):

        if isinstance(parents, int):
           parents = tpl.repeat(fake.ForwardNode, parents)  

        adxs, out, args = jvpfun_args(len(parents), adxs, out, args)

        return tdnode.ForwardLogic(parents, adxs, out, args)

    return wrap




# --- Reverse logic --------------------------------------------------------- #

@pytest.fixture
def reverse_logic(vjpfun_args):

    def wrap(parents=2, adxs=None, out=None, args=None):

        if isinstance(parents, int):
           parents = tpl.repeat(fake.ReverseNode, parents)  

        adxs, out, args = vjpfun_args(len(parents), adxs, out, args)

        return tdnode.ReverseLogic(parents, adxs, out, args)

    return wrap




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Forward logic gate ---------------------------------------------------- #

@pytest.fixture
def forward_gate():

    def wrap(parents=2, fun=None, grad=None):

        if isinstance(parents, int):
           parents = tpl.repeat(fake.ForwardNode, parents) 

        if grad is None:
           grad = fake.FunReturn()

        if fun is None:
           fun = fake.Fun()

        return tdnode.ForwardGate(parents, fun, grad)

    return wrap 




# --- Reverse logic gate ---------------------------------------------------- #

@pytest.fixture
def reverse_gate():

    def wrap(parents=2, fun=None, vjp=None):

        if isinstance(parents, int):
           parents = tpl.repeat(fake.ReverseNode, parents) 

        if vjp is None:
           vjp = fake.Fun(len(parents))

        if fun is None:
           fun = fake.Fun()

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
           source = fake.FunReturn()

        return tdnode.Point(source)
                                  
    return wrap




