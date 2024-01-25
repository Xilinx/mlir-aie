#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect(AIE aie)
add_mlir_doc(AIEDocs AIE ./ -gen-dialect-doc)

# Add AIE interfaces
set(LLVM_TARGET_DEFINITIONS AIEInterfaces.td)
mlir_tablegen(AIEInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIEInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIEInterfacesIncGen)
add_dependencies(aie-headers MLIRAIEInterfacesIncGen)

# Add AIE operations
set(LLVM_TARGET_DEFINITIONS AIEOps.td)
mlir_tablegen(AIEOps.h.inc -gen-op-decls)
mlir_tablegen(AIEOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAIEOpsIncGen)
add_dependencies(aie-headers MLIRAIEOpsIncGen)

# Add AIE enums
set(LLVM_TARGET_DEFINITIONS AIEAttrs.td)
mlir_tablegen(AIEEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEEnumsIncGen)
add_dependencies(aie-headers MLIRAIEEnumsIncGen)

# Add AIE attributes
set(LLVM_TARGET_DEFINITIONS AIEAttrs.td)
mlir_tablegen(AIEAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AIEAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAIEAttrDefsIncGen)
add_dependencies(aie-headers MLIRAIEAttrDefsIncGen)
