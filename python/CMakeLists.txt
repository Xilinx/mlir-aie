# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(AIE_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aie")

# The AIE copy of the MLIR bindings is in the `aie.mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AiePythonSources
  ROOT_DIR "${AIE_PYTHON_ROOT_DIR}"
  SOURCES
    dialects/_ods_common.py)

declare_mlir_python_sources(AiePythonExtensions)

declare_mlir_python_sources(AiePythonSources.Dialects
  ADD_TO_PARENT AiePythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AiePythonSources.Dialects
  ROOT_DIR "${AIE_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AieBinding.td
  SOURCES
    dialects/aie/__init__.py
    dialects/_AIE_ops_ext.py
  DIALECT_NAME AIE
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(AiePythonExtensions.MLIR
  MODULE_NAME _aieMlir
  ADD_TO_PARENT AiePythonExtensions
  ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    AIEMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    AIECAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_common_capi_library(AieAggregateCAPI
  INSTALL_COMPONENT AiePythonModules
  INSTALL_DESTINATION python/aie/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${AIE_PYTHON_PACKAGES_DIR}/aie/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AiePythonSources
    AiePythonExtensions
)

add_mlir_python_modules(AieMLIRPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie/mlir"
  INSTALL_PREFIX "python/aie/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AiePythonExtensions
  COMMON_CAPI_LINK_LIBS
    AieAggregateCAPI
    AIECAPI
  )

add_mlir_python_modules(AiePythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    AiePythonSources
  COMMON_CAPI_LINK_LIBS
    AieAggregateCAPI
)

add_dependencies(AiePythonModules AieCompilerPythonModules)
add_subdirectory(aie/compiler)
