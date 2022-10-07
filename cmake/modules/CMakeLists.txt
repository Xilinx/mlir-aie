#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.

set(AIE_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/aie)
set(aie_cmake_builddir "${CMAKE_BINARY_DIR}/${AIE_INSTALL_PACKAGE_DIR}")

# Export targets for the build directory
export(EXPORT AIETargets FILE ${aie_cmake_builddir}/AIETargets.cmake)

# Generate AIEConfig.cmake for the build tree.
set(AIE_CONFIG_CMAKE_DIR "${aie_cmake_builddir}")
set(AIE_CONFIG_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(AIE_CONFIG_TOOLS_BINARY_DIR "${PROJECT_BINARY_DIR}/bin")
set(AIE_CONFIG_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include"
  )
set(AIE_CONFIG_EXPORTS_FILE "\${AIE_CMAKE_DIR}/AIETargets.cmake")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIEConfig.cmake.in
  ${aie_cmake_builddir}/AIEConfig.cmake
  @ONLY)
set(AIE_CONFIG_CMAKE_DIR)
set(AIE_CONFIG_BINARY_DIR)
set(AIE_CONFIG_TOOLS_BINARY_DIR)
set(AIE_CONFIG_INCLUDE_DIRS)
set(AIE_CONFIG_EXPORTS_FILE)

# Generate AIEConfig.cmake for the install tree.
set(AIE_CONFIG_CODE "
# Compute the installation prefix from this file location.
get_filename_component(AIE_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${AIE_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(AIE_CONFIG_CODE "${AIE_CONFIG_CODE}
get_filename_component(AIE_INSTALL_PREFIX \"\${AIE_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(AIE_CONFIG_CMAKE_DIR "\${AIE_INSTALL_PREFIX}/${AIE_INSTALL_PACKAGE_DIR}")
set(AIE_CONFIG_BINARY_DIR "\${AIE_INSTALL_PREFIX}")
set(AIE_CONFIG_TOOLS_BINARY_DIR "\${AIE_INSTALL_PREFIX}/bin")
set(AIE_CONFIG_INCLUDE_DIRS
  "\${AIE_INSTALL_PREFIX}/include"
  )
set(AIE_CONFIG_EXPORTS_FILE "\${AIE_CMAKE_DIR}/AIETargets.cmake")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AIEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIEConfig.cmake
  @ONLY)
set(AIE_CONFIG_CODE)
set(AIE_CONFIG_CMAKE_DIR)
set(AIE_CONFIG_BINARY_DIR)
set(AIE_CONFIG_TOOLS_BINARY_DIR)
set(AIE_CONFIG_INCLUDE_DIRS)
set(AIE_CONFIG_EXPORTS_FILE)

# export targets for the install directory
install(EXPORT AIETargets DESTINATION ${AIE_INSTALL_PACKAGE_DIR}
        COMPONENT aie-cmake-exports)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIEConfig.cmake
  DESTINATION ${AIE_INSTALL_PACKAGE_DIR}
  COMPONENT aie-cmake-exports)


set(PHY_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/phy)
set(phy_cmake_builddir "${CMAKE_BINARY_DIR}/${PHY_INSTALL_PACKAGE_DIR}")

# Keep this in sync with mlir/cmake/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${MLIR_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(PHY_EXPORTS GLOBAL PROPERTY PHY_EXPORTS)
export(TARGETS ${PHY_EXPORTS} FILE ${phy_cmake_builddir}/PhyTargets.cmake)

get_property(PHY_ALL_LIBS GLOBAL PROPERTY PHY_ALL_LIBS)
get_property(PHY_DIALECT_LIBS GLOBAL PROPERTY PHY_DIALECT_LIBS)
get_property(PHY_CONVERSION_LIBS GLOBAL PROPERTY PHY_CONVERSION_LIBS)
get_property(PHY_TRANSLATION_LIBS GLOBAL PROPERTY PHY_TRANSLATION_LIBS)
get_property(PHY_ANALYSIS_LIBS GLOBAL PROPERTY PHY_ANALYSIS_LIBS)

# Generate PHYConfig.cmake for the build tree.
set(PHY_CONFIG_CMAKE_DIR "${phy_cmake_builddir}")
set(PHY_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(PHY_CONFIG_MLIR_CMAKE_DIR "${mlir_cmake_builddir}")
set(PHY_CONFIG_LIBRARY_DIRS "${PHY_LIBRARY_DIR}")
set(PHY_CONFIG_BINARY_DIR "${PHY_BINARY_DIR}")
set(PHY_CONFIG_TOOLS_DIR "${PHY_TOOLS_DIR}")
set(PHY_CONFIG_INCLUDE_EXPORTS "include(\"\${PHY_CMAKE_DIR}/PhyTargets.cmake\")")
set(PHY_CONFIG_INCLUDE_DIRS
  "${PHY_SOURCE_DIR}/include"
  "${PHY_BINARY_DIR}/include"
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PhyConfig.cmake.in
  ${phy_cmake_builddir}/PhyConfig.cmake
  @ONLY
)
set(PHY_CONFIG_CMAKE_DIR)
set(PHY_CONFIG_LLVM_CMAKE_DIR)
set(PHY_CONFIG_MLIR_CMAKE_DIR)
set(PHY_CONFIG_LIBRARY_DIRS)
set(PHY_CONFIG_BINARY_DIR)
set(PHY_CONFIG_TOOLS_DIR)
set(PHY_CONFIG_INCLUDE_EXPORTS)
set(PHY_CONFIG_INCLUDE_DIRS)

# Generate PhyConfig.cmake for the install tree.
set(PHY_CONFIG_CODE "
# Compute the installation prefix from this PhyConfig.cmake file location.
get_filename_component(PHY_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${PHY_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(PHY_CONFIG_CODE "${PHY_CONFIG_CODE}
get_filename_component(PHY_INSTALL_PREFIX \"\${PHY_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(PHY_CONFIG_CMAKE_DIR "\${PHY_INSTALL_PREFIX}/${PHY_INSTALL_PACKAGE_DIR}")
set(PHY_CONFIG_LLVM_CMAKE_DIR "\${PHY_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(PHY_CONFIG_MLIR_CMAKE_DIR "\${PHY_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(PHY_CONFIG_LIBRARY_DIRS "\${PHY_INSTALL_PREFIX}/lib")
set(PHY_CONFIG_BINARY_DIR "\${PHY_INSTALL_PREFIX}")
set(PHY_CONFIG_TOOLS_DIR "\${PHY_INSTALL_PREFIX}/bin")
set(PHY_CONFIG_INCLUDE_EXPORTS "include(\"\${PHY_CMAKE_DIR}/PhyTargets.cmake\")")
set(PHY_CONFIG_INCLUDE_DIRS
  "\${PHY_INSTALL_PREFIX}/include"
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/PhyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PhyConfig.cmake
    @ONLY
)
set(PHY_CONFIG_CODE)
set(PHY_CONFIG_CMAKE_DIR)
set(PHY_CONFIG_LLVM_CMAKE_DIR)
set(PHY_CONFIG_MLIR_CMAKE_DIR)
set(PHY_CONFIG_LIBRARY_DIRS)
set(PHY_CONFIG_BINARY_DIR)
set(PHY_CONFIG_TOOLS_DIR)
set(PHY_CONFIG_INCLUDE_EXPORTS)
set(PHY_CONFIG_INCLUDE_DIRS)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the Phy parts as well
  # Include the cmake files so other tools can use phy-tblgen, etc.
  install(TARGETS ${PHY_EXPORTS} DESTINATION ${PHY_INSTALL_PACKAGE_DIR}
          COMPONENT phy-cmake-exports)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PhyConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddPhy.cmake
    DESTINATION ${PHY_INSTALL_PACKAGE_DIR}
    COMPONENT phy-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(phy-cmake-exports)
    add_llvm_install_targets(install-phy-cmake-exports
                             COMPONENT phy-cmake-exports)
  endif()
endif()
