# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

# The AIE copy of the MLIR bindings is in the `aie.mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AIEPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
)

declare_mlir_python_sources(AIEPythonExtensions)

declare_mlir_python_sources(AIEPythonSources.Dialects
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    dialects/__init__.py
)

declare_mlir_python_sources(AIEPythonSources.Tools
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    tools/__init__.py
)

declare_mlir_python_sources(AIEPythonSources.Utils
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    utils/__init__.py
    utils/benchmark.py
    utils/config.py
    utils/test.py
    utils/ml.py
    utils/npukernel.py
    utils/regdb.py
    utils/verify.py
    utils/hostruntime/__init__.py
    utils/hostruntime/argparse.py
    utils/hostruntime/cli.py
    utils/hostruntime/hostruntime.py
    utils/hostruntime/tensor_class.py
    utils/hostruntime/xrtruntime/__init__.py
    utils/hostruntime/xrtruntime/hostruntime.py
    utils/hostruntime/xrtruntime/parameter_scratchpad.py
    utils/hostruntime/xrtruntime/tensor.py
    utils/compile/__init__.py
    utils/compile/utils.py
    utils/compile/cache/utils.py
    utils/compile/jit/__init__.py
    utils/compile/jit/compilabledesign.py
    utils/compile/jit/_dma_size_parser.py
    utils/compile/jit/_hash.py
    utils/compile/jit/_introspect.py
    utils/compile/jit/_serialization.py
    utils/compile/jit/markers.py
    utils/compile/jit/context.py
    utils/compile/jit/compileconfig.py
    utils/callabledesign.py
    utils/jit.py
    utils/trace/__init__.py
    utils/trace/config.py
    utils/trace/events/__init__.py
    utils/trace/event_ir.py
    utils/trace/get_trace_summary.py
    utils/trace/parse.py
    utils/trace/setup.py
    utils/trace/utils.py
)

declare_mlir_python_sources(AIEPythonSources.Helpers
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    helpers/*.py
    helpers/taplib/*.py
)

declare_mlir_python_sources(AIEPythonSources.Iron
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    iron/*.py
    iron/compile/*.py
    iron/hostruntime/*.py
    iron/algorithms/*.py
    iron/kernels/*.py
    iron/dataflow/*.py
    iron/device/*.py
    iron/runtime/*.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEBinding.td
  SOURCES
    dialects/aie.py
    _mlir_libs/_aie.pyi
  DIALECT_NAME aie
  GEN_ENUM_BINDINGS_TD_FILE
    "dialects/AIEBinding.td"
  DEPENDS
    aie-headers
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEXBinding.td
  SOURCES
    dialects/aiex.py
  DIALECT_NAME aiex
  DEPENDS
    aie-headers
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEVecBinding.td
  SOURCES
    dialects/aievec.py
  DIALECT_NAME aievec
  DEPENDS
    aie-headers
)

configure_file(compiler/aiecc/configure.py.in aie/compiler/aiecc/configure.py)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_BINARY_DIR}/aie/compiler/aiecc/configure.py"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/aie/compiler/aiecc/configure.py"
  ${CMAKE_CURRENT_IN}
  DESTINATION ${AIE_PYTHON_INSTALL_DIR}/aie/compiler/aiecc
)

# Note that compiler/ is not globbed but compiler/aiecc/ and compiler/txn2mlir/
# this is so that compiler/aiecc.py and compiler/txn2mlir.py do not get distributed
# through this dir but actually through bin (see all the way at the bottom of this file)
declare_mlir_python_sources(AIEPythonSources.Compiler
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    compiler/aiecc/*.py
    compiler/txn2mlir/*.py
)

################################################################################
# Extensions
################################################################################

# Standard Python C extension bindings.
declare_mlir_python_extension(AIEPythonExtensions.MLIR
  MODULE_NAME _aie
  ADD_TO_PARENT AIEPythonExtensions
  ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}

  PARTIAL_SOURCES_INTENDED
  SOURCES
    AIEMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    AIECAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_mlir_python_common_capi_library(AIEAggregateCAPI
  INSTALL_COMPONENT AIEPythonModules
  INSTALL_DESTINATION ${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs
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

add_mlir_python_modules(AIEPythonModules
  ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
  INSTALL_PREFIX "${AIE_PYTHON_INSTALL_DIR}/aie"
  DECLARED_SOURCES
    AIEPythonSources
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AIEPythonExtensions
  COMMON_CAPI_LINK_LIBS
    AIEAggregateCAPI
)

# Vendor eudsl-python-extras into the build directory
add_custom_command(
  TARGET AIEPythonModules POST_BUILD
  COMMAND ${Python3_EXECUTABLE} ${AIE_SOURCE_DIR}/utils/mlir_aie_wheels/vendor_eudsl.py
    --requirements ${AIE_SOURCE_DIR}/python/requirements.txt
    --target ${AIE_PYTHON_PACKAGES_DIR}
  COMMENT "Vendoring eudsl-python-extras"
)

install(
  DIRECTORY "${AIE_PYTHON_PACKAGES_DIR}/aie/extras"
  DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie"
  COMPONENT aie-python
)

# Note: events_database.json is generated at build time by the
# GenerateAIEEventsTD target (include/aie/Dialect/AIE/IR/CMakeLists.txt).
# Python trace event enums are sourced from the TableGen-generated
# _aie_enum_gen.py bindings (no separate generation step needed).

# Stage the MLIR core type stubs (Context, Location, Module, ...) next to the
# compiled extension. aie/ir.py re-exports those symbols via
# `from ._mlir_libs._mlir.ir import *`; the .py sources we assemble don't carry
# their types, so a static type checker can only resolve them from these stubs.
# They ship with the MLIR distro but aren't copied by add_mlir_python_modules.
add_custom_command(
  TARGET AIEPythonModules POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core.type_stub_gen/_mlir_libs/_mlir/__init__.pyi"
    "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core.type_stub_gen/_mlir_libs/_mlir/ir.pyi"
    "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core.type_stub_gen/_mlir_libs/_mlir/passmanager.pyi"
    "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core.type_stub_gen/_mlir_libs/_mlir/rewrite.pyi"
    "${AIE_PYTHON_PACKAGES_DIR}/aie/_mlir_libs/_mlir/"
  COMMENT "Staging MLIR core type stubs for pyright"
)

################################################################################
# ParameterScratchpad pybind11 module (uses a raw buffer)
################################################################################
if(AIE_ENABLE_XRT_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      RESULT_VARIABLE _pybind11_cmakedir_result
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(NOT _pybind11_cmakedir_result EQUAL 0)
      message(FATAL_ERROR
        "AIE_ENABLE_XRT_PYTHON_BINDINGS requires pybind11 in ${Python3_EXECUTABLE}."
      )
    endif()
    find_package(pybind11 CONFIG REQUIRED
      PATHS "${_pybind11_cmakedir}"
      NO_DEFAULT_PATH
    )
  endif()

  pybind11_add_module(_parameter_scratchpad
    ${CMAKE_CURRENT_SOURCE_DIR}/ParameterScratchpadModule.cpp
  )
  target_compile_definitions(_parameter_scratchpad PRIVATE TEST_UTILS_USE_XRT=0)
  target_include_directories(_parameter_scratchpad PRIVATE
    ${AIE_SOURCE_DIR}/runtime_lib/test_lib
  )
  set_target_properties(_parameter_scratchpad PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${AIE_PYTHON_PACKAGES_DIR}/aie/_mlir_libs"
    RUNTIME_OUTPUT_DIRECTORY "${AIE_PYTHON_PACKAGES_DIR}/aie/_mlir_libs"
  )
  add_dependencies(AIEPythonModules _parameter_scratchpad)
  install(TARGETS _parameter_scratchpad
    LIBRARY DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
    RUNTIME DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
    COMPONENT aie-python
  )
endif()

target_include_directories(AIEPythonModules.extension._aie.dso PUBLIC ${VITIS_AIETOOLS_DIR}/include)

# Copy the runtime libs into the _mlir_libs directory for convenience.
set(_runtime_deps
  mlir_apfloat_wrappers
  mlir_async_runtime
  mlir_c_runner_utils
  mlir_float16_utils
  mlir_runner_utils
)

if (TARGET mlir_rocm_runtime)
  list(APPEND _runtime_deps mlir_rocm_runtime)
endif()

set(HAS_MLIR_RUNTIME_LIBRARIES ON PARENT_SCOPE)
foreach(r ${_runtime_deps})
  if(NOT TARGET ${r})
    set(HAS_MLIR_RUNTIME_LIBRARIES OFF PARENT_SCOPE)
    continue()
  endif()
  # build dir lib/
  add_custom_command(
    TARGET AIEPythonModules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${r}>
    "${CMAKE_BINARY_DIR}/lib"
  )
  # build dir aie/_mlir_libs
  add_custom_command(
    TARGET AIEPythonModules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${r}>
    "${CMAKE_CURRENT_BINARY_DIR}/aie/_mlir_libs"
  )
  # install dir
  install(FILES
    $<TARGET_FILE:${r}>
    COMPONENT aie-python
    DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
  )
endforeach()

# during build
add_custom_command(
  TARGET AIEPythonModules PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler/aiecc.py
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler/txn2mlir.py
  ${CMAKE_BINARY_DIR}/bin
)
# during install
install(PROGRAMS compiler/aiecc.py compiler/txn2mlir.py DESTINATION bin)
