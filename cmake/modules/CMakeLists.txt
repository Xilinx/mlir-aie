#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc. (c) Copyright 2023 Advanced Micro Devices Inc.

# Generate a list of CMake library targets so that other CMake projects can link
# against them. LLVM calls its version of this file LLVMExports.cmake, but the
# usual CMake convention seems to be ${Project}Targets.cmake.

set(AIE_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/aie)
set(aie_cmake_builddir "${CMAKE_BINARY_DIR}/${AIE_INSTALL_PACKAGE_DIR}")

# Export targets for the build directory
export(EXPORT AIETargets FILE ${aie_cmake_builddir}/AIETargets.cmake)

# Generate AIEConfig.cmake for the build tree.
set(AIE_CONFIG_CMAKE_DIR "${aie_cmake_builddir}")
set(AIE_CONFIG_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(AIE_CONFIG_TOOLS_BINARY_DIR "${PROJECT_BINARY_DIR}/bin")
set(AIE_CONFIG_PYTHON_INSTALL_DIR "${PROJECT_BINARY_DIR}/${AIE_PYTHON_INSTALL_DIR}")
set(AIE_CONFIG_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include"
                            "${PROJECT_BINARY_DIR}/include")
set(AIE_CONFIG_EXPORTS_FILE "\${AIE_CMAKE_DIR}/AIETargets.cmake")
set(AIE_CONFIG_LibXAIE_x86_64_DIR)
if("x86_64" IN_LIST AIE_RUNTIME_TARGETS)
  set(AIE_CONFIG_LibXAIE_x86_64_DIR
      "${PROJECT_BINARY_DIR}/runtime_lib/x86_64/xaiengine")
endif()
set(AIE_CONFIG_LibXAIE_aarch64_DIR)
if("aarch64" IN_LIST AIE_RUNTIME_TARGETS)
  set(AIE_CONFIG_LibXAIE_aarch64_DIR
      "${PROJECT_BINARY_DIR}/runtime_lib/aarch64/xaiengine")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AIEConfig.cmake.in
               ${aie_cmake_builddir}/AIEConfig.cmake @ONLY)
set(AIE_CONFIG_CMAKE_DIR)
set(AIE_CONFIG_BINARY_DIR)
set(AIE_CONFIG_TOOLS_BINARY_DIR)
set(AIE_CONFIG_INCLUDE_DIRS)
set(AIE_CONFIG_EXPORTS_FILE)

# Generate AIEConfig.cmake for the install tree.
set(AIE_CONFIG_CODE
    "
# Compute the installation prefix from this file location.
get_filename_component(AIE_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)"
)
# Construct the proper number of get_filename_component(... PATH) calls to
# compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${AIE_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(AIE_CONFIG_CODE "${AIE_CONFIG_CODE}
get_filename_component(AIE_INSTALL_PREFIX \"\${AIE_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(AIE_CONFIG_CMAKE_DIR "\${AIE_INSTALL_PREFIX}/${AIE_INSTALL_PACKAGE_DIR}")
set(AIE_CONFIG_BINARY_DIR "\${AIE_INSTALL_PREFIX}")
set(AIE_CONFIG_TOOLS_BINARY_DIR "\${AIE_INSTALL_PREFIX}/bin")
set(AIE_CONFIG_INCLUDE_DIRS "\${AIE_INSTALL_PREFIX}/include")
set(AIE_CONFIG_PYTHON_INSTALL_DIR "\${AIE_INSTALL_PREFIX}/${AIE_PYTHON_INSTALL_DIR}")
set(AIE_CONFIG_EXPORTS_FILE "\${AIE_CMAKE_DIR}/AIETargets.cmake")
set(MLIR_CONFIG_INCLUDE_EXPORTS_FILE "\${AIE_CMAKE_DIR}/MLIRTargets.cmake")
set(XAIENGINECDO_CONFIG_INCLUDE_EXPORTS_FILE "\${AIE_CMAKE_DIR}/xaienginecdo_static.cmake")
set(CDO_DRIVER_CONFIG_INCLUDE_EXPORTS_FILE "\${AIE_CMAKE_DIR}/cdo_driver_mlir_aie.cmake")
set(AIE_CONFIG_LibXAIE_x86_64_DIR)
if("x86_64" IN_LIST AIE_RUNTIME_TARGETS)
  set(AIE_CONFIG_LibXAIE_x86_64_DIR
      "\${AIE_INSTALL_PREFIX}/runtime_lib/x86_64/xaiengine")
endif()
set(AIE_CONFIG_LibXAIE_aarch64_DIR)
if("aarch64" IN_LIST AIE_RUNTIME_TARGETS)
  set(AIE_CONFIG_LibXAIE_aarch64_DIR
      "\${AIE_INSTALL_PREFIX}/runtime_lib/aarch64/xaiengine")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AIEConfig.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIEConfig.cmake @ONLY)
set(AIE_CONFIG_CODE)
set(AIE_CONFIG_CMAKE_DIR)
set(AIE_CONFIG_BINARY_DIR)
set(AIE_CONFIG_TOOLS_BINARY_DIR)
set(AIE_CONFIG_INCLUDE_DIRS)
set(AIE_CONFIG_EXPORTS_FILE)
set(AIE_CONFIG_LibXAIE_x86_64_DIR)
set(AIE_CONFIG_LibXAIE_aarch64_DIR)

# export targets for the install directory
install(
  EXPORT AIETargets
  DESTINATION ${AIE_INSTALL_PACKAGE_DIR}
  COMPONENT aie-cmake-exports)
install(
  EXPORT MLIRTargets
  DESTINATION ${AIE_INSTALL_PACKAGE_DIR}
  COMPONENT aie-cmake-exports)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIEConfig.cmake
  DESTINATION ${AIE_INSTALL_PACKAGE_DIR}
  COMPONENT aie-cmake-exports)
