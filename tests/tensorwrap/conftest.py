#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


# --- Backend fixtures ------------------------------------------------------ #

@pytest.fixture
def current_backend(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def numpy_backend():
    return "numpy"

available_backends = ["numpy_backend"]
















































