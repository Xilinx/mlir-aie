# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from aie.extras import types as T
from aie.extras.util import bfloat16, _pseudo_bfloat16, np_dtype_to_mlir_type
from util import construct_and_print_module

# RUN: %PYTHON %s | FileCheck %s


# CHECK-LABEL: bfloat16Conversion
# CHECK: PASS!
@construct_and_print_module
def bfloat16Conversion():
    assert np.dtype(bfloat16).itemsize == 2
    assert _pseudo_bfloat16 != np.float16
    assert np_dtype_to_mlir_type(_pseudo_bfloat16) == T.bf16()
    try:
        from bfloat16 import bfloat16 as real_bfloat16

        assert real_bfloat16 == bfloat16
        assert np_dtype_to_mlir_type(real_bfloat16) == T.bf16()
        assert np_dtype_to_mlir_type(bfloat16) == T.bf16()
        assert np_dtype_to_mlir_type(real_bfloat16) == np_dtype_to_mlir_type(
            _pseudo_bfloat16
        )
    except ModuleNotFoundError:
        pass
    print("PASS!")
