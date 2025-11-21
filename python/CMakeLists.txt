# Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
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
)

declare_mlir_python_sources(AIEPythonSources.Util
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    util.py
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
    utils/config.py
    utils/test.py
    utils/jit.py
    utils/ml.py
    utils/npukernel.py
    utils/regdb.py
    utils/hostruntime/__init__.py
    utils/hostruntime/hostruntime.py
    utils/hostruntime/tensor_class.py
    utils/hostruntime/xrtruntime/__init__.py
    utils/hostruntime/xrtruntime/hostruntime.py
    utils/hostruntime/xrtruntime/tensor.py
    utils/compile/__init__.py
    utils/compile/link.py
    utils/compile/utils.py
    utils/compile/cache/circular_cache.py
    utils/compile/cache/utils.py
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
    helpers/dialects/ext/*.py
    helpers/runtime/*.py
    helpers/taplib/*.py
)

declare_mlir_python_sources(AIEPythonSources.Iron
  ADD_TO_PARENT AIEPythonSources
  SOURCES_GLOB
    iron/*.py
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

if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
  declare_mlir_python_sources(AIEPythonSources.XRT
    ADD_TO_PARENT AIEPythonSources
    SOURCES
      xrt.py
      _mlir_libs/_xrt.pyi
  )
endif()

declare_mlir_python_sources(AIEPythonSources.AIERT
  ADD_TO_PARENT AIEPythonSources
  SOURCES
    aiert.py
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

  set(_py_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/AIEMLIRModule.cpp
    # Python passes
    ${CMAKE_CURRENT_SOURCE_DIR}/PybindTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PythonPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RouterPass.cpp

    ${_mlir_python_cpp_sources}
    ${RegisterEverythingSources}
  )

  set(_py_libs
    LLVMSupport
    AIECAPI
    AIETargets
    ${MLIR_CONVERSION_LIBS}
    ${MLIR_DIALECT_LIBS}
    ${MLIR_EXTENSION_LIBS}
    ${MLIR_TRANSLATION_LIBS}
    ${MLIR_UPSTREAM_CAPI_LIBS}
  )

  if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
    list(APPEND _py_srcs ${CMAKE_CURRENT_SOURCE_DIR}/XRTModule.cpp)
    list(APPEND _py_libs xrt_coreutil uuid)
  endif()
  list(APPEND _py_srcs ${CMAKE_CURRENT_SOURCE_DIR}/AIERTModule.cpp)

  declare_mlir_python_extension(AIEPythonExtensions.MLIR
    MODULE_NAME _aie
    ADD_TO_PARENT AIEPythonExtensions
    ROOT_DIR "/"

    SOURCES
      ${_py_srcs}

    PRIVATE_LINK_LIBS
      ${_py_libs}

    PYTHON_BINDINGS_LIBRARY
      nanobind
  )
  target_include_directories(
    AIEPythonExtensions.MLIR
    INTERFACE $<BUILD_INTERFACE:${PYBINDINGS_SRC}>
  )
  if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
    target_include_directories(AIEPythonExtensions.MLIR INTERFACE ${XRT_INCLUDE_DIR})
    target_link_directories(AIEPythonExtensions.MLIR INTERFACE ${XRT_LIB_DIR})
  endif()

  # Kill all of the dependencies (mlir_python_DEPENDS) on cpp sources in the dialect build targets
  # so the C extension modules don't get built but the Python sources do get installed.
  _flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
  list(FILTER mlir_python_sources_deps INCLUDE REGEX "^MLIRPythonSources.Dialects")
  foreach(t ${mlir_python_sources_deps})
    set_property(TARGET ${t} PROPERTY mlir_python_DEPENDS "")
  endforeach()
  set_property(TARGET MLIRPythonSources.ExecutionEngine PROPERTY mlir_python_DEPENDS "")

  # Set NB_DOMAIN for nanobind, to avoid domain conflict with other LLVM projects.
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "_aie")

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
  if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
    list(APPEND _other_extensions _xrt)
  endif()

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
    PYTHON_BINDINGS_LIBRARY
      nanobind
  )

  if(AIE_ENABLE_XRT_PYTHON_BINDINGS)
    declare_mlir_python_extension(AIEPythonExtensions.XRT
      MODULE_NAME _xrt
      ADD_TO_PARENT AIEPythonExtensions
      ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}

      PARTIAL_SOURCES_INTENDED
      SOURCES
        XRTModule.cpp

      PRIVATE_LINK_LIBS
        LLVMSupport
        xrt_coreutil
        uuid
      PYTHON_BINDINGS_LIBRARY
        nanobind
    )
    target_include_directories(AIEPythonExtensions.XRT INTERFACE ${XRT_INCLUDE_DIR})
    target_link_directories(AIEPythonExtensions.XRT INTERFACE ${XRT_LIB_DIR})
  endif()

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

  ################################################################################
  # Generated trace events enum files and copy to Python package directories
  ################################################################################

  # Input header files from aie-rt
  set(AIE_RT_EVENTS_DIR "${AIE_SOURCE_DIR}/third_party/aie-rt/driver/src/events")
  set(EVENTS_HEADER_FILES
    "${AIE_RT_EVENTS_DIR}/xaie_events_aie.h"
    "${AIE_RT_EVENTS_DIR}/xaie_events_aieml.h"
    "${AIE_RT_EVENTS_DIR}/xaie_events_aie2p.h"
  )

  # Output directory for generated files in build directory
  set(GENERATED_EVENTS_DIR "${CMAKE_BINARY_DIR}/lib/regdb")

  # Generated Python enum files in the build directory
  set(GENERATED_EVENTS_ENUM_FILES
    "${GENERATED_EVENTS_DIR}/aie.py"
    "${GENERATED_EVENTS_DIR}/aie2.py"
    "${GENERATED_EVENTS_DIR}/aie2p.py"
  )

  # Generated JSON database in the build directory
  set(GENERATED_EVENTS_JSON_FILENAME "events_database.json")
  set(GENERATED_EVENTS_JSON "${GENERATED_EVENTS_DIR}/${GENERATED_EVENTS_JSON_FILENAME}")

  # Add custom command to generate the event enum files
  add_custom_command(
    OUTPUT ${GENERATED_EVENTS_ENUM_FILES} ${GENERATED_EVENTS_JSON}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_EVENTS_DIR}
    COMMAND ${Python3_EXECUTABLE} 
      ${AIE_SOURCE_DIR}/utils/generate_events_enum.py
      -i ${EVENTS_HEADER_FILES}
      -o ${GENERATED_EVENTS_DIR}
      --json ${GENERATED_EVENTS_JSON_FILENAME}
      --python-prefix ""
    DEPENDS 
      ${AIE_SOURCE_DIR}/utils/generate_events_enum.py
      ${EVENTS_HEADER_FILES}
    COMMENT "Generating trace events enum files from aie-rt headers"
    VERBATIM
  )

  # Install the generated json to the lib dir
  install(
    FILES 
      ${GENERATED_EVENTS_JSON}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/regdb
    COMPONENT aie-python
  )

  # Create a custom target for the generated files
  add_custom_target(GenerateTraceEventsEnum ALL
    DEPENDS ${GENERATED_EVENTS_ENUM_FILES} ${GENERATED_EVENTS_JSON}
  )

  # Ensure AIEPythonModules depends on GenerateTraceEventsEnum
  add_dependencies(AIEPythonModules GenerateTraceEventsEnum)

  # Copy generated event enum files to Python package build directory
  foreach(_file ${GENERATED_EVENTS_ENUM_FILES})
    get_filename_component(_filename ${_file} NAME)
    add_custom_command(
      TARGET AIEPythonModules POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${AIE_PYTHON_PACKAGES_DIR}/aie/utils/trace/events
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_file}
        ${AIE_PYTHON_PACKAGES_DIR}/aie/utils/trace/events/${_filename}
      COMMENT "Copying ${_filename} to build Python package"
    )
  endforeach()

  # Install generated event enum files
  install(
    FILES 
      ${GENERATED_EVENTS_ENUM_FILES}
    DESTINATION ${AIE_PYTHON_INSTALL_DIR}/aie/utils/trace/events
    COMPONENT aie-python
  )


endif ()

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
    break()
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
  install(IMPORTED_RUNTIME_ARTIFACTS
    ${r}
    COMPONENT aie-python
    LIBRARY
    DESTINATION "${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
  )
endforeach()

# libxrt_coreutil has to be handled differentled because it's versioned...
if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
  # for build dir
  add_custom_command(
    TARGET AIEPythonModules.extension._xrt.dso PRE_BUILD
    COMMENT "Copying libxrt_coreutil into _mlir_libs during build"
    DEPENDS "${XRT_COREUTIL}"
    COMMAND "${CMAKE_COMMAND}" -E copy
    "${XRT_COREUTIL}" "${CMAKE_CURRENT_BINARY_DIR}/aie/_mlir_libs/libxrt_coreutil.so"
  )
  # for install
  install(FILES
    "${XRT_COREUTIL}"
    COMPONENT xrt_coreutil
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${AIE_PYTHON_INSTALL_DIR}/aie/_mlir_libs"
    # hack
    RENAME libxrt_coreutil.so.2
  )
endif()

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
