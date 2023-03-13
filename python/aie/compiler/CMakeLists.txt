# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

declare_mlir_python_sources(AieCompilerPythonSources)

declare_mlir_python_sources(AieCompilerPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT AieCompilerPythonSources
  SOURCES_GLOB
    aiecc/*.py
    *.py
)

add_mlir_python_modules(AieCompilerPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie/compiler"
  INSTALL_PREFIX "${AIE_PYTHON_INSTALL_DIR}/aie/compiler"
  DECLARED_SOURCES AieCompilerPythonSources
  )
