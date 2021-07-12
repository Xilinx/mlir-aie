#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect_library(AIE
  AIEAssignBuffers.cpp
  AIEDialect.cpp
  AIETokenAnalysis.cpp
  AIENetlistAnalysis.cpp
  AIEFindFlows.cpp
  AIECreateFlows.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIECoreToLLVM.cpp
  AIECoreToStandard.cpp
  AIEHerdRouting.cpp
  AIECreatePacketFlows.cpp
  AIELowerMemcpy.cpp
  AIENormalizeAddressSpaces.cpp
  AIEVectorOpt.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEPassIncGen
  MLIRAIENormalizeAddressSpacesIncGen
  )

target_link_libraries(AIE
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )

add_subdirectory(Targets)
