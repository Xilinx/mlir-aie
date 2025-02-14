# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

from aie.dialects.aiex import GetTileOp
from aie.extras.dialects.ext import arith
from aie.extras import types as T
from util import construct_and_print_module


# CHECK-LABEL: getTileOp
# CHECK: aiex.getTile
@construct_and_print_module
def getTileOp():
    four = arith.constant(4, index=True)
    two = arith.constant(2, index=True)
    GetTileOp(T.index(), four, two)
