#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# Use a simplified version AIEDocs.td to generate the AIEDialect.md ending up as
# https://xilinx.github.io/mlir-aie/AIEDialect.html
add_mlir_doc(AIEDocs AIEDialect ./ -gen-dialect-doc -dialect=aie)

# Add AIE dialect
set(LLVM_TARGET_DEFINITIONS AIE.td)
mlir_tablegen(AIEDialect.h.inc -gen-dialect-decls -dialect=aie)
mlir_tablegen(AIEDialect.cpp.inc -gen-dialect-defs -dialect=aie)
add_public_tablegen_target(MLIRAIEIncGen)
add_dependencies(aie-headers MLIRAIEIncGen)

# Add AIE Types
set(LLVM_TARGET_DEFINITIONS AIETypes.td)
mlir_tablegen(AIETypes.h.inc -gen-typedef-decls -typedefs-dialect=aie)
mlir_tablegen(AIETypes.cpp.inc -gen-typedef-defs -typedefs-dialect=aie)
add_public_tablegen_target(MLIRAIETypesIncGen)
add_dependencies(aie-headers MLIRAIETypesIncGen)

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
add_dependencies(MLIRAIEEnumsIncGen GenerateAIEEventsTD)

# Generate AIE event enums from aie-rt headers
set(AIE_RT_EVENTS_DIR "${AIE_SOURCE_DIR}/third_party/aie-rt/driver/src/events")
set(EVENTS_HEADER_FILES
  "${AIE_RT_EVENTS_DIR}/xaie_events_aie.h"
  "${AIE_RT_EVENTS_DIR}/xaie_events_aieml.h"
  "${AIE_RT_EVENTS_DIR}/xaie_events_aie2p.h"
)

set(GENERATED_EVENTS_TD_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/AIEEvents.td.inc"
  "${CMAKE_CURRENT_BINARY_DIR}/AIEEventsAIE2.td.inc"
  "${CMAKE_CURRENT_BINARY_DIR}/AIEEventsAIE2P.td.inc"
)

add_custom_command(
  OUTPUT ${GENERATED_EVENTS_TD_FILES}
  COMMAND ${Python3_EXECUTABLE}
    ${AIE_SOURCE_DIR}/utils/generate_events_tablegen.py
    -i ${EVENTS_HEADER_FILES}
    -o ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${AIE_SOURCE_DIR}/utils/generate_events_tablegen.py
    ${EVENTS_HEADER_FILES}
  COMMENT "Generating AIE event enum tablegen files from aie-rt headers"
  VERBATIM
)

add_custom_target(GenerateAIEEventsTD
  DEPENDS ${GENERATED_EVENTS_TD_FILES}
)

# Add AIE attributes
set(LLVM_TARGET_DEFINITIONS AIEAttrs.td)
mlir_tablegen(AIEAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AIEAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAIEAttrDefsIncGen)
add_dependencies(aie-headers MLIRAIEAttrDefsIncGen)
add_dependencies(MLIRAIEAttrDefsIncGen GenerateAIEEventsTD)
