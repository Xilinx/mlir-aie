# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path

# RUN: %python %s | FileCheck %s
# REQUIRES: xrt_python_bindings


from aie.xrt import load_xclbin

# CHECK: MLIR_AIE
print(load_xclbin(str(Path(__file__).parent / "final.xclbin")))
