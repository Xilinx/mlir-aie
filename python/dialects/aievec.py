# Copyright (C) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# noinspection PyUnresolvedReferences
from ._aievec_ops_gen import *

from .._mlir_libs._aie import *
from .._mlir_libs import get_dialect_registry

# noinspection PyUnresolvedReferences
from ..extras.dialects import memref

# Comes from _aie
register_dialect(get_dialect_registry())
