#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates

set(LLVM_TARGET_DEFINITIONS AIEVecTransformOps.td)
mlir_tablegen(AIEVecTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AIEVecTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAIEVecTransformOpsIncGen)

add_mlir_doc(AIEVecTransformOps AIEVecTransformOps ./ -gen-op-doc)
