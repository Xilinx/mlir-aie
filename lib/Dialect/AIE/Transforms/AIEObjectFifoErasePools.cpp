//===- AIEObjectFifoErasePools.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOERASEPOOLS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct AIEObjectFifoErasePoolsPass
    : public xilinx::AIE::impl::AIEObjectFifoErasePoolsBase<
          AIEObjectFifoErasePoolsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();

    // Pool users are open-ended because any symbol user may retain the pool.
    SymbolTableCollection symbolTables;
    SymbolUserMap users(symbolTables, device);

    for (auto pool :
         llvm::make_early_inc_range(device.getOps<ObjectFifoPoolOp>())) {
      if (users.getUsers(pool).empty()) {
        pool.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEObjectFifoErasePoolsPass() {
  return std::make_unique<AIEObjectFifoErasePoolsPass>();
}
