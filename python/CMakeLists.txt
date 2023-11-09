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

get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(MLIR_UPSTREAM_CAPI_LIBS GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

get_target_property(RegisterEverythingSources MLIRPythonExtension.RegisterEverything INTERFACE_SOURCES)

_flatten_mlir_python_targets(mlir_python_sources MLIRPythonSources)
set(_mlir_python_cpp_sources)
foreach(s ${mlir_python_sources})
  get_target_property(_sources ${s} INTERFACE_SOURCES)
  list(FILTER _sources INCLUDE REGEX "(\.cpp|\.h)$")
  if(_sources)
    list(APPEND _mlir_python_cpp_sources ${_sources})
  endif()
endforeach()

list(SORT _mlir_python_cpp_sources)

declare_mlir_python_extension(AIEPythonExtensions
  MODULE_NAME _aie
  ROOT_DIR "/"

  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/AIEMLIRModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PythonPass.cpp

  ${_mlir_python_cpp_sources}
  ${RegisterEverythingSources}

  PRIVATE_LINK_LIBS
  LLVMSupport
  AIECAPI
  ${MLIR_CONVERSION_LIBS}
  ${MLIR_DIALECT_LIBS}
  ${MLIR_EXTENSION_LIBS}
  ${MLIR_TRANSLATION_LIBS}
  ${MLIR_UPSTREAM_CAPI_LIBS}
)

_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
foreach(t ${mlir_python_sources_deps})
  set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
endforeach()
set_property(TARGET MLIRPythonSources.ExecutionEngine PROPERTY mlir_python_DEPENDS "")

add_mlir_python_modules(AIEMLIRPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "python/aie"
  DECLARED_SOURCES
    ${mlir_python_sources_deps}
    MLIRPythonSources.Core.Python
    MLIRPythonSources.ExecutionEngine
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

set(_other_extensions
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

foreach(ext ${_other_extensions})
  add_custom_target("symlink_${ext}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "_aie${PYTHON_MODULE_EXTENSION}"
      "${ext}${PYTHON_MODULE_EXTENSION}"
    DEPENDS AIEPythonExtensions
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/aie/_mlir_libs"
  )
  add_dependencies(AIEPythonModules "symlink_${ext}")
  install(CODE "execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            _aie${PYTHON_MODULE_EXTENSION}
            ${ext}${PYTHON_MODULE_EXTENSION}
            WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/python/aie/_mlir_libs
    )"
  )
endforeach()

set(_runtimes
  mlir_async_runtime
  mlir_c_runner_utils
  mlir_float16_utils
  mlir_runner_utils
)

if (TARGET mlir_rocm_runtime)
  list(APPEND _runtimes mlir_rocm_runtime)
endif()

foreach(r ${_runtimes})
  add_custom_command(
    TARGET AIEPythonModules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${r}>
    ${CMAKE_BINARY_DIR}/lib
  )
endforeach()

install(IMPORTED_RUNTIME_ARTIFACTS
  ${_runtimes}
  COMPONENT aie-python
  LIBRARY DESTINATION "python/aie/_mlir_libs"
)

