#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import tests.autodiff.fakes as fake
import tests.common.ntuple  as tpl




# --- Random value ---------------------------------------------------------- #

@pytest.fixture
def randn():

    def wrap(seed=1):

        np.random.seed(seed)    
        return np.random.randn()

    return wrap




# --- Node arguments -------------------------------------------------------- #

def node_args(args=None):

    if args is None:
       args = 2

    if isinstance(args, int):
       return tpl.repeat(fake.Node, args)

    return args




# --- Operator arguments ---------------------------------------------------- #

def op_args(fun=None, x=None):

    if fun is None:
       fun = fake.Fun() 

    if x is None:
       x = fake.FunReturn()

    return fun, x




# --- Default arguments of any adjoint function ----------------------------- #

def adjfun_args(nodetype):

    def wrap(valency, adxs, out, args):

        if adxs is None:
           adxs = list(range(valency))

        if out is None:
           out = nodetype()

        if args is None:
           args = tpl.repeat(fake.Node, max(adxs)+1) 

        return adxs, out, args

    return wrap




# --- Default JVP function arguments ---------------------------------------- #

@pytest.fixture
def jvpfun_args():       

    def wrap(valency=2, adxs=None, out=None, args=None):

        return adjfun_args(fake.ForwardNode)(valency, adxs, out, args)

    return wrap




# --- Default VJP function arguments ---------------------------------------- #

@pytest.fixture
def vjpfun_args():

    def wrap(valency=2, adxs=None, out=None, args=None):

        return adjfun_args(fake.ReverseNode)(valency, adxs, out, args)

    return wrap




