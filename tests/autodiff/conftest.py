#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


from tests.autodiff.fixtures.misc import (
   randn,
   jvpfun_args,
   vjpfun_args,
)   


from tests.autodiff.fixtures.node import (
   forward_logic,
   forward_gate,
   forward_node,
   nodule,
   point,
   reverse_logic,
   reverse_gate,
   reverse_node,
)


from tests.autodiff.fixtures.graph import (
   graph,
   fun_with_gate,
   differentiable,
   nondifferentiable,
   node_train,
   node_glue,
   concat_args_kernel,
   concat_args,
   active,
   passive,
   active_pack,
   passive_pack,
   point_pack,
)


from tests.autodiff.fixtures.grad import (
   forward_diff_op,
   reverse_diff_op,
   childcount,
   toposort,
   gradaccum,
   backprop,
)












































































































