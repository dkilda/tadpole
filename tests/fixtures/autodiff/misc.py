#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import mocks as mock



# --- Random value ---------------------------------------------------------- #

@pytest.fixture
def randn():

    def wrap(seed=1):

        np.random.seed(seed)    
        return np.random.randn()

    return wrap




# --- Default arguments of any adjoint function ----------------------------- #

def adjfun_args(nodetype):

    def wrap(valency, adxs, out, args):

        if adxs is None:
           adxs = list(range(valency))

        if out is None:
           out = nodetype()

        if args is None:
           args = tuple([mock.Node()]*max(adxs))

        return adxs, out, args

    return wrap




# --- Default JVP function arguments ---------------------------------------- #

@pytest.fixture
def jvpfun_args():       

    def wrap(valency=2, adxs=None, out=None, args=None):

        return adjfun_args(mock.ForwardNode)(valency, adxs, out, args)

    return wrap




# --- Default VJP function arguments ---------------------------------------- #

@pytest.fixture
def vjpfun_args():

    def wrap(valency=2, adxs=None, out=None, args=None):

        return adjfun_args(mock.ReverseNode)(valency, adxs, out, args)

    return wrap





