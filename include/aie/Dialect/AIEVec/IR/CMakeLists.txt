#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates

add_mlir_dialect(AIEVecOps aievec)
add_mlir_doc(AIEVecOps AIEVecDialect ./ -gen-dialect-doc -dialect=aievec)

# Add AIEVec attributes
set(LLVM_TARGET_DEFINITIONS AIEVecAttributes.td)
mlir_tablegen(AIEVecEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEVecEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AIEVecAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(AIEVecAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAIEVecAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRAIEVecAttributesIncGen)