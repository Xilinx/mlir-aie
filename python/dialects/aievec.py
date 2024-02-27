# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# noinspection PyUnresolvedReferences
from ._aievec_ops_gen import *

from .._mlir_libs._aie import *
from .._mlir_libs import get_dialect_registry

# noinspection PyUnresolvedReferences
from ..extras.dialects.ext import memref

# Comes from _aie
register_dialect(get_dialect_registry())
