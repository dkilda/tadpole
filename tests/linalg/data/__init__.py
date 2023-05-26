#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tests.array.data import (
   randn,
   randn_pos,
   array_dat,
   narray_dat,
   arrayspace_dat,   
)


from tests.tensor.data import (
   indices_dat,
   nindices_dat,
   tensor_basis_dat,
   tensor_sample_dat,
   sparsegrad_dat,
   sparsegrad_dat_001,
   sparsegrad_dat_002,
   sparsegrad_dat_003,
   sparsegrad_dat_004,
   sparsegrad_dat_005,
   nullgrad_dat,
   nullgrad_dat_001,
   tensor_dat,
   ntensor_dat,
   tensorspace_dat,
   tensorspace_dat_001,
)


from tests.tensor.data import (
   unary_wrappedfun_dat_001,
   unary_wrappedfun_dat_002,
   binary_wrappedfun_dat_001,
   binary_wrappedfun_dat_002,
   binary_wrappedfun_dat_003,
   binary_wrappedfun_dat_004,
)


from tests.tensor.data import (
   train_dat,
)


from .decomp import (
   svd_trunc_dat,
   randn_decomp_000,
   decomp_input_000,
   decomp_input_001,
   decomp_input_002,
   decomp_input_003,
   qr_tensor_dat,
   lq_tensor_dat,
   svd_tensor_dat,
   eig_tensor_dat,
   eigh_tensor_dat,
)


from .properties import (
   property_input_001,
   property_input_002,
   property_input_003,
   property_linalg_dat,
)


from .solvers import (
   solver_input_000,
   solver_input_001,
   solver_input_002,
   solve_linalg_dat,
   trisolve_upper_linalg_dat,
   trisolve_lower_linalg_dat,
)




