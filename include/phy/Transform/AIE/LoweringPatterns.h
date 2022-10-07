//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "phy/Transform/Base/LoweringPatterns.h"

#include "aie/AIEDialect.h"

#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#ifndef MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
#define MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H

namespace xilinx {
namespace phy {
namespace transform {
namespace aie {

class AIELoweringPatternSets : public LoweringPatternSets {
  ModuleOp module;

  // dmas/shim_dmas[{col, row}] == DMAOp/ShimDMAOp
  std::map<std::pair<int, int>, AIE::MemOp> dmas;
  std::map<std::pair<int, int>, AIE::ShimDMAOp> shim_dmas;

  // locks[{TileOp, id}] == LockOp
  std::map<std::pair<AIE::TileOp, int>, AIE::LockOp> locks;

  // tiles[{col, row}] == TileOp
  std::map<std::pair<int, int>, AIE::TileOp> tiles;

public:
  AIELoweringPatternSets(ModuleOp &module) : module(module) {}
  ~AIELoweringPatternSets() override {}

  // Lists of lowering pattern sets
  std::list<std::list<std::unique_ptr<LoweringPatternSet>>>
  getPatternSets() override;

  // Shared resources constructors and getters.
  AIE::MemOp getDma(std::pair<int, int> index);
  AIE::LockOp getLock(AIE::TileOp tile, int id);
  AIE::ShimDMAOp getShimDma(std::pair<int, int> index);
  AIE::TileOp getTile(mlir::OpState &op);
  AIE::TileOp getTile(std::pair<int, int> index);

  // Common attribute getters.
  AIE::DMAChannel getChannel(mlir::OpState &op, phy::physical::StreamOp stream);
  int getId(mlir::OpState &op);
  std::string getImpl(mlir::OpState &op);
  std::pair<int, int> getTileIndex(mlir::OpState &op);
  AIE::WireBundle getWireBundle(phy::physical::StreamOp &op);
};

} // namespace aie
} // namespace transform
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
