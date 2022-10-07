##===- AddPhy.txt - adding Phy to CMake -----------------------*- cmake -*-===//
##
## Adapted from llvm/circt/cmake/modules/AddCIRCT.cmake
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

include_guard()

function(add_phy_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(phy-headers MLIR${dialect}IncGen)
endfunction()

function(add_phy_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(phy-headers MLIR${interface}IncGen)
endfunction()

# Additional parameters are forwarded to tablegen.
function(add_phy_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${PHY_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(phy-doc ${output_id}DocGen)
endfunction()

function(add_phy_library name)
  add_mlir_library(${ARGV})
  add_phy_library_install(${name})
endfunction()

# Adds a phy library target for installation.  This should normally only be
# called from add_phy_library().
function(add_phy_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT PhyTargets)
  set_property(GLOBAL APPEND PROPERTY PHY_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY PHY_EXPORTS ${name})
endfunction()

function(add_phy_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY PHY_DIALECT_LIBS ${name})
  add_phy_library(${ARGV} DEPENDS phy-headers)
endfunction()

function(add_phy_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY PHY_CONVERSION_LIBS ${name})
  add_phy_library(${ARGV} DEPENDS phy-headers)
endfunction()

function(add_phy_translation_library name)
  set_property(GLOBAL APPEND PROPERTY PHY_TRANSLATION_LIBS ${name})
  add_phy_library(${ARGV} DEPENDS phy-headers)
endfunction()

set(PHY_TABLE_GEN_DEF "")

macro(add_phy_definition)
  add_definitions(${ARGV})
  list(APPEND PHY_TABLE_GEN_DEF ${ARGV})
endmacro()

macro(phy_tablegen)
  mlir_tablegen(${ARGV} ${PHY_TABLE_GEN_DEF})
endmacro()
