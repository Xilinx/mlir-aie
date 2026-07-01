#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(LLVM_TARGET_DEFINITIONS AIEVecTransformOps.td)
mlir_tablegen(AIEVecTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AIEVecTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAIEVecTransformOpsIncGen)

