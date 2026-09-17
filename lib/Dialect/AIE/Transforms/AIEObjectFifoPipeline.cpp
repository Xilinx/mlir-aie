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
  Option<bool> reserveControlIds{
      *this, "reserve-control-ids",
      llvm::cl::desc(
          "Reserve ctrl-pkt controller ids during packet id assignment."),
      llvm::cl::init(false)};
  Option<bool> dmaFenceSharedMem{
      *this, "dma-fence-shared-mem",
      llvm::cl::desc("Opt-in: carry every cross-tile core-to-core lock-only "
                     "shared-memory objectfifo on DMA (write-completion "
                     "barrier). Off by default."),
      llvm::cl::init(false)};
  Option<bool> warnUnfencedSharedOverlay{
      *this, "warn-unfenced-shared-overlay",
      llvm::cl::desc("Warn when a cross-tile core-to-core shared-memory "
                     "objectfifo is left on the lock-only path (no "
                     "write-completion barrier under the ctrl-pkt overlay)."),
      llvm::cl::init(false)};
};
} // namespace

void xilinx::AIE::registerAIEObjectFifoPipeline() {
  PassPipelineRegistration<ObjectFifoLoweringOptions>(
      "aie-objectFifo-stateful-transform",
      "Lower aie.objectfifo to buffers, locks, flows and DMA programs",
      [](OpPassManager &pm, const ObjectFifoLoweringOptions &options) {
        pm.addPass(createAIEObjectFifoSplitPass(
            options.dmaFenceSharedMem, options.warnUnfencedSharedOverlay));
        if (!options.skipVerify) {
          pm.addPass(createAIEObjectFifoVerifyPass());
        }
        pm.addPass(createAIEObjectFifoAllocatePass(options.packetSwitched,
                                                   options.reserveControlIds));
        pm.addPass(createAIEObjectFifoLowerDMAsPass());
        pm.addPass(createAIEObjectFifoLowerCoresPass());
        pm.addPass(createAIEObjectFifoErasePoolsPass());
      });
}
