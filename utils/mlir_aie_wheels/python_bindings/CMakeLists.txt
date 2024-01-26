cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0116 OLD)
endif()

project(PI LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

if(MSVC)
  add_compile_options(
    $<$<CONFIG:>:/MT> # ---------|
    $<$<CONFIG:Debug>:/MTd> # ---|-- Statically link the runtime libraries
    $<$<CONFIG:Release>:/MT> # --|
  )
endif()

find_package(AIE REQUIRED CONFIG)

message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${AIE_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.")

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  # TODO(max): AIEConfig.cmake should add this to the CXXFLAGS...
  add_compile_definitions(AIE_ENABLE_GENERATE_CDO_DIRECT)
  # for cdo_driver.h
  find_package(Vitis 2023.2 COMPONENTS AIE AIE2)
  include_directories(${VITIS_AIETOOLS_DIR}/include)
endif()

set(_sources
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  MLIRPythonExtension.ExecutionEngine
  AIEPythonExtensions
  AIEPythonSources
  AIEPythonSources.Compiler
)

if(AIE_ENABLE_XRT_PYTHON_BINDINGS)
  find_package(XRT)
  list(APPEND _sources AIEPythonExtensions.XRT)
  find_library(XRT_COREUTIL xrt_coreutil HINTS ${XRT_LIB_DIR})
  add_library(xrt_coreutil SHARED IMPORTED)
  set_property(TARGET xrt_coreutil PROPERTY IMPORTED_LOCATION "${XRT_COREUTIL}")
  install(IMPORTED_RUNTIME_ARTIFACTS
    xrt_coreutil
    COMPONENT xrt_coreutil
    LIBRARY
    DESTINATION "${CMAKE_INSTALL_PREFIX}/aie/_mlir_libs"
  )
endif()

add_mlir_python_common_capi_library(AIEAggregateCAPI
  INSTALL_COMPONENT AIEPythonModules
  INSTALL_DESTINATION aie/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/aie/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_sources}
)

# ##############################################################################
# Custom targets.
# ##############################################################################

_flatten_mlir_python_targets(aie_python_sources_deps
  AIEPythonSources
  AIEPythonSources.Compiler
)
add_custom_target("aie-python-sources" DEPENDS ${aie_python_sources_deps})
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-aie-python-sources DEPENDS
                           aie-python-sources COMPONENT aie-python-sources)
endif()

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  find_library(CDO_DRIVER_LIBRARY cdo_driver
    HINTS "${VITIS_AIETOOLS_DIR}/lib/lnx64.o"
  )

  # TODO(max): this is not DRY - should be able to fetch it from mlir_aie_install/lib
  add_library(cdo_driver SHARED IMPORTED)
  set_property(TARGET cdo_driver PROPERTY IMPORTED_LOCATION "${CDO_DRIVER_LIBRARY}")
  install(IMPORTED_RUNTIME_ARTIFACTS
    cdo_driver
    COMPONENT aie-python-cdo
    LIBRARY
    DESTINATION "${CMAKE_INSTALL_PREFIX}/aie/_mlir_libs"
  )
endif()

file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/aie/compiler/aiecc"
  TYPE FILE
  FILES "${AIE_PYTHON_INSTALL_DIR}/aie/compiler/aiecc/configure.py"
)

# ##############################################################################
# The fully assembled package of modules. This must come last.
# ##############################################################################

add_mlir_python_modules(AIEMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/aie"
  INSTALL_PREFIX aie
  DECLARED_SOURCES ${_sources}
  COMMON_CAPI_LINK_LIBS
    AIEAggregateCAPI
)