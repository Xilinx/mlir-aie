//===- AIENpuLowering.h - Shared NPU lowering pipeline ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared NPU lowering pipeline used by both aiecc and aie-translate.
// This is the canonical pass sequence for lowering high-level DMA task ops
// (dma_configure_task_for, dma_start_task, dma_await_task, etc.) to flat
// npu.write32/blockwrite/sync/address_patch ops.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIENPULOWERING_H
#define AIE_TARGETS_AIENPULOWERING_H

namespace mlir {
class PassManager;
} // namespace mlir

namespace xilinx::AIE {

/// Populate the pass manager with the NPU lowering pipeline.
///
/// Adds the following passes:
/// - AIEMaterializeRuntimeSequences (module-level, skipped if
///   \p skipMaterialize is true)
/// - AIEMaterializeBDChains (device-level)
/// - AIESubstituteShimDMAAllocations (device-level)
/// - AIEAssignRuntimeSequenceBDIDs (device-level)
/// - AIEDMATasksToNPU (device-level)
/// - AIEDmaToNpu (device-level)
/// - AIELowerSetLock (device-level)
void populateNpuLoweringPipeline(mlir::PassManager &pm,
                                 bool skipMaterialize = false);

} // namespace xilinx::AIE

#endif // AIE_TARGETS_AIENPULOWERING_H
