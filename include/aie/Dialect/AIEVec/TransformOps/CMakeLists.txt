#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023, Advanced Micro Devices, Inc.

set(LLVM_TARGET_DEFINITIONS AIEVecTransformOps.td)
mlir_tablegen(AIEVecTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AIEVecTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAIEVecTransformOpsIncGen)

add_mlir_doc(AIEVecTransformOps AIEVecTransformOps ./ -gen-op-doc)
