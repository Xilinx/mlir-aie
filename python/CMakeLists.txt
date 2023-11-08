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

get_target_property(CoreSources MLIRPythonExtension.Core INTERFACE_SOURCES)

get_target_property(DialectsLinalgPybindSources MLIRPythonExtension.Dialects.Linalg.Pybind INTERFACE_SOURCES)
get_target_property(DialectsQuantPybindSources MLIRPythonExtension.Dialects.Quant.Pybind INTERFACE_SOURCES)
get_target_property(DialectsPDLPybindSources MLIRPythonExtension.Dialects.PDL.Pybind INTERFACE_SOURCES)
get_target_property(DialectsSparseTensorPybindSources MLIRPythonExtension.Dialects.SparseTensor.Pybind INTERFACE_SOURCES)
get_target_property(DialectsTransformPybindSources MLIRPythonExtension.Dialects.Transform.Pybind INTERFACE_SOURCES)

get_target_property(AsyncDialectPassesSources MLIRPythonExtension.AsyncDialectPasses INTERFACE_SOURCES)
get_target_property(ExecutionEngineSources MLIRPythonExtension.ExecutionEngine INTERFACE_SOURCES)
get_target_property(GPUDialectPassesSources MLIRPythonExtension.GPUDialectPasses INTERFACE_SOURCES)
get_target_property(SparseTensorDialectPassesSources MLIRPythonExtension.SparseTensorDialectPasses INTERFACE_SOURCES)
get_target_property(RegisterEverythingSources MLIRPythonExtension.RegisterEverything INTERFACE_SOURCES)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  LLVMSupport

  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  ${MLIR_EXTENSION_LIBS}
  ${MLIR_TRANSLATION_LIBS}

  ADF
  AIE
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEVec
  MLIRAIEVecToLLVM
  MLIRAIEVecTransformOps
  MLIRAIEVecTransforms

  AIECAPI

  MLIRCAPIAsync
  MLIRCAPIConversion
  MLIRCAPIDebug
  MLIRCAPIExecutionEngine
  MLIRCAPIFunc
  MLIRCAPIGPU
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILinalg
  MLIRCAPIPDL
  MLIRCAPIQuant
  MLIRCAPIRegisterEverything
  MLIRCAPISparseTensor
  MLIRCAPITransformDialect
  MLIRCAPITransforms
)

declare_mlir_python_extension(AIEPythonExtensionsOmni.MLIR
  MODULE_NAME _aieMlirOmni
  ADD_TO_PARENT AIEPythonExtensions
  ROOT_DIR "/"

  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/AIEMLIRModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PythonPass.cpp

  ${CoreSources}

  ${DialectsLinalgPybindSources}
  ${DialectsQuantPybindSources}
  ${DialectsPDLPybindSources}
  ${DialectsSparseTensorPybindSources}
  ${DialectsTransformPybindSources}

  ${AsyncDialectPassesSources}
  ${ExecutionEngineSources}
  ${GPUDialectPassesSources}
  ${SparseTensorDialectPassesSources}
  ${RegisterEverythingSources}

  PRIVATE_LINK_LIBS
  ${LIBS}
)

_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
foreach(t ${mlir_python_sources_deps})
  set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
endforeach()

add_mlir_python_modules(AIEMLIRPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    ${mlir_python_sources_deps}
    MLIRPythonSources.Core.Python
    AIEPythonExtensions
)

add_mlir_python_modules(AIEPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    AIEPythonSources
)

add_subdirectory(compiler)
add_dependencies(AIEPythonModules AIECompilerPythonModules)

set(_extensions
  _aieMlir
  _mlir
  _mlirAsyncPasses
  _mlirDialectsLinalg
  _mlirDialectsPDL
  _mlirDialectsQuant
  _mlirDialectsSparseTensor
  _mlirDialectsTransform
  _mlirExecutionEngine
  _mlirGPUPasses
  _mlirLinalgPasses
  _mlirPythonTest
  _mlirRegisterEverything
  _mlirSparseTensorPasses
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python/aie/_mlir_libs")

foreach(ext ${_extensions})
  add_custom_target("symlink_${ext}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "_aieMlirOmni${PYTHON_MODULE_EXTENSION}"
      "${ext}${PYTHON_MODULE_EXTENSION}"
    DEPENDS AIEPythonExtensionsOmni.MLIR
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/aie/_mlir_libs"
  )
  add_dependencies(AIEPythonModules "symlink_${ext}")
  install(CODE "execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            _aieMlirOmni${PYTHON_MODULE_EXTENSION}
            ${ext}${PYTHON_MODULE_EXTENSION}
            WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/python/aie/_mlir_libs
    )"
  )

endforeach()
