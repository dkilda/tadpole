#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .adjoints import (
   jvpmap_dat,
   vjpmap_dat,
)


from .nary import (
   nary_op_dat,
   nary_op_creator_dat,
   singular_argproxy_dat,
   singular_argproxy_dat_001,
   plural_argproxy_dat,
   plural_argproxy_dat_001,
)


from .node import (
   forward_adjfun_dat,  reverse_adjfun_dat,  adjointop_dat,
   forward_flow_dat,    reverse_flow_dat,    null_flow_dat,
   forward_gate_dat,    reverse_gate_dat,
   forward_node_dat,    reverse_node_dat,    node_dat, point_dat,
   forward_parents_dat, reverse_parents_dat,
)


from .graph import (
   graph_dat,
   differentiable_funwrap_dat, 
   nondifferentiable_funwrap_dat,
   args_dat, nodeargs_dat,
   concat_dat, concat_output_dat, 
   pack_dat, 
   envelope_dat,
   node_stack_dat_001, 
   node_stack_dat_002,
)


from .grad import (
   diffop_dat,
   childcount_dat,
   forward_node_network_dat,
   reverse_node_network_dat,
)




