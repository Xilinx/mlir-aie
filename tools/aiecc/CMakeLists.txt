# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Build aiecc compiler driver
add_llvm_executable(aiecc aiecc.cpp aiecc_aiesim.cpp)
llvm_update_compile_flags(aiecc)

# Refresh AIECCVersion.h each build so --version reflects the current SHA.
# GenVersion.cmake only rewrites the file when the SHA changes, so aiecc.cpp
# is not pointlessly rebuilt.
set(AIECC_VERSION_HEADER ${CMAKE_CURRENT_BINARY_DIR}/AIECCVersion.h)
add_custom_target(aiecc_version ALL
  BYPRODUCTS ${AIECC_VERSION_HEADER}
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
    -DOUTPUT_FILE=${AIECC_VERSION_HEADER}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/GenVersion.cmake
  COMMENT "Updating aiecc git SHA"
  VERBATIM)
add_dependencies(aiecc aiecc_version)
target_include_directories(aiecc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
# aiecc.cpp uses __DATE__ / __TIME__ for the --version banner; that's
# intentional (we want the compile timestamp) so silence -Wdate-time.
# The flag is GCC/Clang-only; MSVC rejects it as an invalid argument.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(aiecc.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-date-time")
endif()
install(TARGETS aiecc
  EXPORT AIETargets
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aiecc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransformDialect
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  AIE
  AIETargets
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEToConfiguration
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecTransformOps
  MLIRAIEVecTransforms
  MLIRAIEVecToLLVM
  MLIRXLLVMDialect
  MLIRXLLVMToLLVMIRTranslation
  ${llvm_libs}
  )

target_link_libraries(aiecc PUBLIC ${LIBS})

# Optional: AIEBU library for direct ELF generation (avoids subprocess to aiebu-asm)
if(aiebu_FOUND)
  target_link_libraries(aiecc PRIVATE AIEBU::aiebu_static)
  target_include_directories(aiecc PRIVATE ${AIEBU_INCLUDE_DIRS})
  target_compile_definitions(aiecc PRIVATE AIECC_HAS_AIEBU_LIBRARY)
  message(STATUS "aiecc: Enabled AIEBU library integration")
endif()

# bootgen library for direct PDI generation (avoids subprocess to bootgen)
# Uses C API wrapper that handles exceptions internally
if(TARGET bootgen-lib)
  find_package(OpenSSL REQUIRED)
  target_link_libraries(aiecc PRIVATE bootgen-lib OpenSSL::SSL OpenSSL::Crypto)
  target_compile_definitions(aiecc PRIVATE AIECC_HAS_BOOTGEN_LIBRARY)
  message(STATUS "aiecc: Enabled bootgen library integration")
endif()
