# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

# The AIE copy of the MLIR bindings is in the `aie.mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.")

if(NOT AIE_PYTHON_INSTALL_DIR)
  set(AIE_PYTHON_INSTALL_DIR "python")
endif()

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AIEPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
)

declare_mlir_python_sources(AIEPythonExtensions)

declare_mlir_python_sources(AIEPythonSources.Dialects
  ADD_TO_PARENT AIEPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEBinding.td
  SOURCES
    dialects/aie.py
  DIALECT_NAME AIE
  GEN_ENUM_BINDINGS_TD_FILE
    "dialects/AIEBinding.td"
)

declare_mlir_python_sources(AIEPythonSources.Extras
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    extras/meta.py
    extras/passes.py
    extras/util.py
)

declare_mlir_python_sources(AIEPythonSources.DialectExtras
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    extras/dialects/*.py
    extras/dialects/**/*.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEXBinding.td
  SOURCES
    dialects/aiex.py
  DIALECT_NAME AIEX
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AIEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/AIEVecBinding.td
  SOURCES
    dialects/aievec.py
  DIALECT_NAME aievec
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

# Note that compiler/ is not globbed but compiler/aiecc/
# this is so that compiler/aiecc.py does not get distributed
# through this dir but actually through bin (see all the way at the bottom of this file)
declare_mlir_python_sources(AIEPythonSources.Compiler
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    compiler/aiecc/*.py
)

################################################################################
# Extensions
################################################################################

if (AIE_ENABLE_PYTHON_PASSES)
  # What's going on here? This branch builds a single shared library Python C extension module
  # with absolutely everything, including the kitchen sink. I.e, this "omnibus" shared library
  # will have absolutely all the MLIR code (including C APIs), all the upstream C extension modules,
  # our main C extension module (AIEMLIRModule.cpp) and the PythonPass.cpp C extension module.
  #
  # Why do this? This is a way to call arbitrary MLIR code from the python bindings. Such as, for example,
  # PythonPass.cpp does.
  #
  # Note, this depends on MLIR being compiled with RTTI (i.e., LLVM_ENABLE_RTTI=ON) and the
  # guard includes this.

  get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
  get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
  get_property(MLIR_UPSTREAM_CAPI_LIBS GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

  # Dig out all of the upstream cpp source files associated with the C extension modules.
  _flatten_mlir_python_targets(mlir_python_sources MLIRPythonSources)
  set(_mlir_python_cpp_sources)
  foreach(s ${mlir_python_sources})
    get_target_property(_sources ${s} INTERFACE_SOURCES)
    list(FILTER _sources INCLUDE REGEX "(\.cpp|\.h)$")
    if(_sources)
      list(APPEND _mlir_python_cpp_sources ${_sources})
    endif()
  endforeach()

  # Missed one...
  get_target_property(RegisterEverythingSources MLIRPythonExtension.RegisterEverything INTERFACE_SOURCES)

  set(PYBINDINGS_SRC "${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core")

  declare_mlir_python_extension(AIEPythonExtensions.MLIR
    MODULE_NAME _aie
    ADD_TO_PARENT AIEPythonExtensions
    ROOT_DIR "/"

    SOURCES

    ${CMAKE_CURRENT_SOURCE_DIR}/AIEMLIRModule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PybindTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PythonPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RouterPass.cpp

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
  target_include_directories(
    AIEPythonExtensions.MLIR
    INTERFACE $<BUILD_INTERFACE:${PYBINDINGS_SRC}>
  )

  # Kill all of the dependencies (mlir_python_DEPENDS) on cpp sources in the dialect build targets
  # so the C extension modules don't get built but the Python sources do get installed.
  _flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
  list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
  foreach(t ${mlir_python_sources_deps})
    set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
  endforeach()
  set_property(TARGET MLIRPythonSources.ExecutionEngine PROPERTY mlir_python_DEPENDS "")

  add_mlir_python_modules(AIEPythonModules
    ROOT_PREFIX "${AIE_PYTHON_PACKAGES_DIR}/aie"
    INSTALL_PREFIX "${AIE_PYTHON_INSTALL_DIR}/aie"
    DECLARED_SOURCES
      ${mlir_python_sources_deps}
      MLIRPythonSources.Core.Python
      MLIRPythonSources.ExecutionEngine
      AIEPythonExtensions.MLIR
      AIEPythonSources
  )

  set(_other_extensions
    _aie_python_passes
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

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs")

  # This is the trick that makes multiple Python C extension modules packaged into one shared library work;
  # Python importlib machinery looks for PyInit__your_module based on the filename _your_module.so.
  # Thus, if you have a symlink _your_module.so -> BIGOMNIBUS.so, it will look for PyInit__your_module in
  # BIGOMNIBUS.so.
  foreach(ext ${_other_extensions})
    # Build time/directory creation.
    add_custom_target("symlink_${ext}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "_aie${PYTHON_MODULE_EXTENSION}"
        "${ext}${PYTHON_MODULE_EXTENSION}"
      DEPENDS AIEPythonExtensions
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
    )
    add_dependencies(AIEPythonModules "symlink_${ext}")
    # Install time/directory creation.
    install(CODE "execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              _aie${PYTHON_MODULE_EXTENSION}
              ${ext}${PYTHON_MODULE_EXTENSION}
              WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs
      )"
    )
  endforeach()

else ()

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
      AIECAPI
  )

endif ()

# Copy the runtime libs into the _mlir_libs directory for convenience.
set(_runtimes
  mlir_async_runtime
  mlir_c_runner_utils
  mlir_float16_utils
  mlir_runner_utils
)

if (TARGET mlir_rocm_runtime)
  list(APPEND _runtimes mlir_rocm_runtime)
endif()

set(HAS_MLIR_RUNTIME_LIBRARIES ON PARENT_SCOPE)
foreach(r ${_runtimes})
  if(NOT TARGET ${r})
    set(HAS_MLIR_RUNTIME_LIBRARIES OFF PARENT_SCOPE)
    break()
  endif()
  add_custom_command(
    TARGET AIEPythonModules PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${r}>
    ${CMAKE_BINARY_DIR}/lib
  )
  install(IMPORTED_RUNTIME_ARTIFACTS
    ${r}
    COMPONENT aie-python
    LIBRARY DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
  )
endforeach()

# during build
add_custom_command(
  TARGET AIEPythonModules PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler/aiecc.py
  ${CMAKE_BINARY_DIR}/bin
)
# during install
install(PROGRAMS compiler/aiecc.py DESTINATION bin)

