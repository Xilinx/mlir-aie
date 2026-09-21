//===- AIEReserveRuntimeBDIDs.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A tile's buffer-descriptor ids are one shared pool, but the static tile
// program and the runtime sequence get their ids from two separate passes that
// do not see each other. This prepass records those compile-time-pinned runtime
// ids per tile as an aiex.reserved_bd_ids attribute on each aie.tile.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <set>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIERESERVERUNTIMEBDIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

struct AIEReserveRuntimeBDIDsPass
    : xilinx::AIEX::impl::AIEReserveRuntimeBDIDsBase<
          AIEReserveRuntimeBDIDsPass> {

  void runOnOperation() override {
    DeviceOp device = getOperation();

    llvm::DenseMap<TileOp, std::set<int32_t>> reserved;
    device.walk([&](DMAConfigureTaskOp cfg) {
      TileOp tile = cfg.tryGetTileOp();
      if (!tile)
        return;
      cfg.walk([&](DMABDOp bd) {
        if (bd.getBdId().has_value())
          reserved[tile].insert(bd.getBdId().value());
      });
    });

    for (auto &[tile, ids] : reserved) {
      SmallVector<int32_t> sorted(ids.begin(), ids.end());
      tile->setAttr("aiex.reserved_bd_ids",
                    DenseI32ArrayAttr::get(&getContext(), sorted));
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEReserveRuntimeBDIDsPass() {
  return std::make_unique<AIEReserveRuntimeBDIDsPass>();
}
