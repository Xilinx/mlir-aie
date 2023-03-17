#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect_library(AIETransforms
  AIEAssignBuffers.cpp
  AIEAssignLockIDs.cpp
  AIEFindFlows.cpp
  AIEPathfinder.cpp
  AIECreatePathfindFlows.cpp
  AIECoreToStandard.cpp
  AIECreatePacketFlows.cpp
  AIECanonicalizeDevice.cpp
  AIELocalizeLocks.cpp
  AIENormalizeAddressSpaces.cpp
  AIEVectorOpt.cpp
  AIEObjectFifoStatefulTransform.cpp
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
