#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
#

set(AIE_REGDB_BUILD_DIR ${AIE_BINARY_DIR}/lib/regdb)
set(AIE_REGDB_FILES
  aie_registers_aie2.json)

set(AIE_REGDB_OUTPUTS)
foreach(regdb_file ${AIE_REGDB_FILES})
  set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${regdb_file})
  set(dst_file ${AIE_REGDB_BUILD_DIR}/${regdb_file})
  add_custom_command(
    OUTPUT ${dst_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${AIE_REGDB_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src_file} ${dst_file}
    DEPENDS ${src_file}
    COMMENT "Copying ${regdb_file} to ${AIE_REGDB_BUILD_DIR}"
  )
  list(APPEND AIE_REGDB_OUTPUTS ${dst_file})
endforeach()

add_custom_target(MLIRAIERegDBResources ALL DEPENDS ${AIE_REGDB_OUTPUTS})

add_mlir_dialect_library(MLIRAIEUtil
  AIERegisterDatabase.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRSupport
)

add_dependencies(MLIRAIEUtil MLIRAIERegDBResources GenerateTraceEventsEnum)

install(FILES
  ${AIE_BINARY_DIR}/lib/regdb/aie_registers_aie2.json
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/regdb)
