#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util as util




###############################################################################
###                                                                         ###
###  Logical operations with arrays                                         ###
###                                                                         ###
###############################################################################


# --- Array equality -------------------------------------------------------- # 

def allequal(x, y):

    return x.allequal(y)


def allclose(x, y, **opts):

    return x.allclose(y, **opts)


def allallequal(xs, ys):

    return all(allequal(x, y) for x, y in zip(xs, ys))


def allallclose(xs, ys, **opts):

    return all(allclose(x, y, **opts) for x, y in zip(xs, ys))








