#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(AIEVecOps aievec)
add_mlir_doc(AIEVecOps AIEVecDialect ./ -gen-dialect-doc -dialect=aievec)

# AIEVecOps.td pulls in AIEAttrs.td which relies on generated event enums
add_dependencies(MLIRAIEVecOpsIncGen GenerateAIEEventsTD)

# Add AIEVec attributes
set(LLVM_TARGET_DEFINITIONS AIEVecAttributes.td)
mlir_tablegen(AIEVecEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEVecEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AIEVecAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(AIEVecAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAIEVecAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRAIEVecAttributesIncGen)
