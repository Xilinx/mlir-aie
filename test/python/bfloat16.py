# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np

# RUN: %PYTHON %s | FileCheck %s
from aie.extras import types as T
from aie.extras.util import bfloat16, _pseudo_bfloat16, np_dtype_to_mlir_type


def bfloat16Conversion():
    assert np.dtype(bfloat16).itemsize == 2
    assert _pseudo_bfloat16 != np.float16
    assert np_dtype_to_mlir_type(_pseudo_bfloat16) == T.bf16()
    try:
        # Checks if bfloat16 library is available
        from bfloat16 import real_bfloat16

        assert real_bfloat16 == bfloat16
        assert np_dtype_to_mlir_type(real_bfloat16) == T.bf16()
        assert np_dtype_to_mlir_type(bfloat16) == T.bf16()
        assert np_dtype_to_mlir_type(real_bfloat16) == np_dtype_to_mlir_type(
            _pseudo_bfloat16
        )
    except:
        # CHECK: Warning! bfloat16 python package not available, so using placeholder bfloat16 type
        pass
    # CHECK: PASS
    print("PASS")
