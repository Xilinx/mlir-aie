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
  AIEPathfinder.cpp
  AIECreatePathfindFlows.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIECoreToStandard.cpp
  AIEHerdRouting.cpp
  AIECreateBroadcastPacket.cpp
  AIELowerMulticast.cpp
  AIECreatePacketFlows.cpp
  AIELowerMemcpy.cpp
  AIELocalizeLocks.cpp
  AIENormalizeAddressSpaces.cpp
  AIEVectorOpt.cpp
  AIEObjectFifoStatefulTransform.cpp
  AIEObjectFifoLoopUnroll.cpp
  AIEObjectFifoRegisterProcess.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEPassIncGen
  MLIRAIENormalizeAddressSpacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )

add_subdirectory(Targets)
add_subdirectory(Dialect)
