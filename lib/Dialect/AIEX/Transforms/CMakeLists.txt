#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(AIEXTransforms
  AIEXToStandard.cpp
  AIESCFToControlFlow.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIECreateBroadcastPacket.cpp
  AIELowerMulticast.cpp
  AIELowerMemcpy.cpp
  AIEDmaToNpu.cpp
  AIENpuToCert.cpp
  AIEMaterializeBDChains.cpp
  AIEMaterializeRuntimeSequences.cpp
  AIEAssignRuntimeSequenceBDIDs.cpp
  AIEUnrollRuntimeSequenceLoops.cpp
  AIELowerDynamicBDPool.cpp
  AIEDMATasksToNPU.cpp
  AIESubstituteShimDMAAllocations.cpp
  AIECtrlPacketToDma.cpp
  AIELowerSetLock.cpp
  AIELowerDmaChannelReset.cpp
  AIELowerCoreReset.cpp
  AIELowerScratchpadParameters.cpp
  AIETransformBfpTypes.cpp
  AIETxnToControlPacket.cpp
  AIEExpandLoadPdi.cpp
  AIEInlineTraceConfig.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEXIncGen
  MLIRAIEXPassIncGen

  LINK_LIBS PUBLIC
  AIE
  MLIRAIEToConfiguration
  AIEXUtils
  MLIRIR
  MLIRPass
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTransformUtils
  MLIRSCFUtils
  )
