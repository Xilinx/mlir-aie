#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# OR-Tools dependency for CP-SAT placer (REQUIRED for this branch)
# User should set ORTOOLS_ROOT environment variable before running build script
# Standard build: ORTOOLS_ROOT=/scratch/jayen/or-tools-source bash utils/build-mlir-aie-from-wheels.sh
#
# NOTE: ortoolsConfig.cmake automatically finds its dependencies (protobuf, absl)
# so we only need to point to ortools_DIR
if(DEFINED ENV{ORTOOLS_ROOT})
  set(ortools_DIR "$ENV{ORTOOLS_ROOT}/lib/cmake/ortools")
endif()
find_package(ortools CONFIG REQUIRED)

add_mlir_dialect_library(
  AIETransforms
  AIEPlacer.cpp
  AIEPlaceTiles.cpp
  AIEAssignBuffers.cpp
  AIEAssignBufferDescriptorIDs.cpp
  AIEAssignLockIDs.cpp
  AIEFindFlows.cpp
  AIEPathFinder.cpp
  AIECreatePathFindFlows.cpp
  AIECoreToStandard.cpp
  AIECanonicalizeDevice.cpp
  AIELocalizeLocks.cpp
  AIENormalizeAddressSpaces.cpp
  AIEVectorOpt.cpp
  AIEVectorToPointerLoops.cpp
  AIEVectorTransferLowering.cpp
  AIEHoistVectorTransferPointers.cpp
  AIEObjectFifoStatefulTransform.cpp
  AIEObjectFifoRegisterProcess.cpp
  AIELowerCascadeFlows.cpp
  AIEGenerateColumnControlOverlay.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIENormalizeAddressSpacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect)

# Link OR-Tools
target_link_libraries(AIETransforms PRIVATE ortools::ortools)

# Mark OR-Tools includes as SYSTEM to suppress warnings from their protobuf headers
# (OR_PROTO_DLL macro causes struct/class mismatch warnings that are harmless on Unix)
get_target_property(ORTOOLS_INCLUDE_DIRS ortools::ortools INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(AIETransforms SYSTEM PRIVATE ${ORTOOLS_INCLUDE_DIRS})

# Disable ALL warnings-as-errors for AIEPlacer.cpp due to OR-Tools headers
# (OR-Tools protobuf-generated code has incomplete types and other issues)
# Also define OR_PROTO_DLL as empty (required by OR-Tools protobuf headers)
set_source_files_properties(AIEPlacer.cpp PROPERTIES COMPILE_FLAGS "-w -DOR_PROTO_DLL=")
