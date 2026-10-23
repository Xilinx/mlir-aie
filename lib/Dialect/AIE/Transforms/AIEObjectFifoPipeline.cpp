//===- AIEObjectFifoPipeline.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {
struct ObjectFifoLoweringOptions
    : public PassPipelineOptions<ObjectFifoLoweringOptions> {
  Option<bool> packetSwitched{
      *this, "packet-sw-objFifos",
      llvm::cl::desc(
          "Flag to enable aie.packetflow lowering from objectfifos."),
      llvm::cl::init(false)};
  Option<bool> skipVerify{
      *this, "skip-verify",
      llvm::cl::desc("Skip structural verification of split objectFifo IR."),
      llvm::cl::init(false)};
  Option<bool> checkQueueSpace{
      *this, "check-queue-space",
      llvm::cl::desc("Wait for room on the task queue before each push a core "
                     "makes onto an out-of-order channel."),
      llvm::cl::init(true)};
};
} // namespace

void xilinx::AIE::registerAIEObjectFifoPipeline() {
  PassPipelineRegistration<ObjectFifoLoweringOptions>(
      "aie-objectFifo-stateful-transform",
      "Lower aie.objectfifo to buffers, locks, flows and DMA programs",
      [](OpPassManager &pm, const ObjectFifoLoweringOptions &options) {
        pm.addPass(createAIEObjectFifoSplitPass());
        if (!options.skipVerify) {
          pm.addPass(createAIEObjectFifoVerifyPass());
        }
        pm.addPass(createAIEObjectFifoAllocatePass(options.packetSwitched));
        // The pushes name the channels the allocation above assigns, and
        // they go where the releases are, which the core lowering below
        // turns into locks.
        AIEObjectFifoInsertCoreQueuePushesOptions pushOptions;
        pushOptions.checkQueueSpace = options.checkQueueSpace;
        pm.addPass(
            createAIEObjectFifoInsertCoreQueuePushesPass(pushOptions));
        pm.addPass(createAIEObjectFifoLowerDMAsPass());
        pm.addPass(createAIEObjectFifoLowerCoresPass());
        pm.addPass(createAIEObjectFifoErasePoolsPass());
      });
}
