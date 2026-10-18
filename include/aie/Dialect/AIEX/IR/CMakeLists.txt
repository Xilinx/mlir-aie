#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(AIEX aiex)
add_mlir_doc(AIEX AIEXDialect ./ -gen-dialect-doc -dialect=aiex)

# AIEX.td pulls in AIEAttrs.td which includes generated event enums
add_dependencies(MLIRAIEXIncGen GenerateAIEEventsTD)

# Add AIEX enums
set(LLVM_TARGET_DEFINITIONS AIEXAttrs.td)
mlir_tablegen(AIEXEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEXEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEXEnumsIncGen)
add_dependencies(MLIRAIEXIncGen MLIRAIEXEnumsIncGen)
add_dependencies(aie-headers MLIRAIEXEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AIEX.td)
mlir_tablegen(AIEXAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=aiex)
mlir_tablegen(AIEXAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=aiex)
add_public_tablegen_target(MLIRAIEXAttrDefsIncGen)
add_dependencies(MLIRAIEXAttrDefsIncGen GenerateAIEEventsTD)
add_dependencies(MLIRAIEXIncGen MLIRAIEXAttrDefsIncGen)
add_dependencies(aie-headers MLIRAIEXAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS AIEXInterfaces.td)
mlir_tablegen(AIEXInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIEXInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIEXInterfacesIncGen)
add_dependencies(MLIRAIEXIncGen MLIRAIEXInterfacesIncGen)
add_dependencies(aie-headers MLIRAIEXInterfacesIncGen)
