#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect_library(AIEXTransforms
  AIEXToStandard.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIEHerdRouting.cpp
  AIECreateBroadcastPacket.cpp
  AIELowerMulticast.cpp
  AIELowerMemcpy.cpp
  AIEDmaToNpu.cpp
  AIEMaterializeBDChains.cpp
  AIEAssignRuntimeSequenceBDIDs.cpp
  AIEDMATasksToNPU.cpp
  AIESubstituteShimDMAAllocations.cpp
  AIECtrlPacketToDma.cpp
  AIELowerSetLock.cpp
  AIETransformBfpTypes.cpp
  AIETxnToControlPacket.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEXIncGen
  MLIRAIEXPassIncGen

  LINK_LIBS PUBLIC
  AIE
  AIEXUtils
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
