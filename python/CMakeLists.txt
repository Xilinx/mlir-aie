# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(AIE_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# The AIE copy of the MLIR bindings is in the `aie.mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AIEPythonSources
  ROOT_DIR "${AIE_PYTHON_ROOT_DIR}"
)

declare_mlir_python_sources(AIEPythonExtensions)

declare_mlir_python_sources(AIEPythonSources.Dialects
  ADD_TO_PARENT AIEPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${AIE_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AIEBinding.td
  SOURCES
    dialects/aie.py
    types.py
    util.py
  DIALECT_NAME AIE
  GEN_ENUM_BINDINGS_TD_FILE
    "dialects/AIEBinding.td"
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${AIE_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AIEXBinding.td
  SOURCES
    dialects/aiex.py
  DIALECT_NAME AIEX
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(AIEPythonExtensions.MLIR
  MODULE_NAME _aieMlir
  ADD_TO_PARENT AIEPythonExtensions
  ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    AIEMLIRModule.cpp
    PythonPass.cpp
  EMBED_CAPI_LINK_LIBS
    AIECAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_common_capi_library(AIEAggregateCAPI
  INSTALL_COMPONENT AIEPythonModules
  INSTALL_DESTINATION python/aie/_mlir_libs
  OUTPUT_DIRECTORY "${AIE_PYTHON_PACKAGES_DIR}/aie/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AIEPythonSources
    AIEPythonExtensions
)

add_mlir_python_modules(AIEMLIRPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AIEPythonExtensions
  COMMON_CAPI_LINK_LIBS
    AIEAggregateCAPI
    AIECAPI
  )

add_mlir_python_modules(AIEPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    AIEPythonSources
  COMMON_CAPI_LINK_LIBS
    AIEAggregateCAPI
)

add_subdirectory(compiler)
add_dependencies(AIEPythonModules AIECompilerPythonModules)
