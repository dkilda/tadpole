#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid
import tadpole.linalg   as la

import tadpole.array.backends as backends
import tadpole.tensor.engine  as tne 

import tests.linalg.fakes as fake
import tests.linalg.data  as data

from tests.common import (
   available_backends,
)

from tadpole.linalg.decomp import (
   SIndexFun, 
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)





















































