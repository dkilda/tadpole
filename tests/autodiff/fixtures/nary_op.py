#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.nary_op as tdnary
import tests.common.ntuple      as tpl




###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Nary operator --------------------------------------------------------- #

@pytest.fixture
def nary_op():

    def wrap(unary_op, fun, argproxy):
        return tdnary.NaryOp(unary_op, fun, argproxy)

    return wrap
































































