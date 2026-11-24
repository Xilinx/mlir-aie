//===- dma_channel_capacity.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every public target has as many DMA channels into a tile as out of it, so
// code that swaps the two directions still passes on all of them. This test
// double gives core and shim tiles 2 S2MM channels and only 1 MM2S channel,
// and checks the per-direction counts that placement and channel allocation
// share.

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include <cstdio>
#include <optional>

using namespace xilinx::AIE;

namespace {

class AsymmetricDMAModel : public VE2802TargetModel {
public:
  uint32_t getNumDestSwitchboxConnections(int col, int row,
                                          WireBundle bundle) const override {
    if (bundle == WireBundle::DMA)
      return 2;
    return VE2802TargetModel::getNumDestSwitchboxConnections(col, row, bundle);
  }
  uint32_t getNumSourceSwitchboxConnections(int col, int row,
                                            WireBundle bundle) const override {
    if (bundle == WireBundle::DMA)
      return 1;
    return VE2802TargetModel::getNumSourceSwitchboxConnections(col, row,
                                                               bundle);
  }
  uint32_t getNumDestShimMuxConnections(int col, int row,
                                        WireBundle bundle) const override {
    if (bundle == WireBundle::DMA)
      return 2;
    return VE2802TargetModel::getNumDestShimMuxConnections(col, row, bundle);
  }
  uint32_t getNumSourceShimMuxConnections(int col, int row,
                                          WireBundle bundle) const override {
    if (bundle == WireBundle::DMA)
      return 1;
    return VE2802TargetModel::getNumSourceShimMuxConnections(col, row, bundle);
  }
};

int failures = 0;

void expect(const char *what, int actual, int expected) {
  if (actual != expected) {
    std::fprintf(stderr, "FAIL: %s: got %d, expected %d\n", what, actual,
                 expected);
    ++failures;
  }
}

std::optional<TileID> findTile(const AIETargetModel &tm, bool shim) {
  for (int col = 0; col < tm.columns(); ++col)
    for (int row = 0; row < tm.rows(); ++row)
      if (shim ? tm.isShimNOCTile(col, row) : tm.isCoreTile(col, row))
        return TileID{col, row};
  return std::nullopt;
}

} // namespace

int main() {
  AsymmetricDMAModel tm;
  for (bool shim : {false, true}) {
    std::optional<TileID> tile = findTile(tm, shim);
    if (!tile) {
      std::fprintf(stderr, "FAIL: no %s tile\n", shim ? "shim" : "core");
      return 1;
    }
    const char *kind = shim ? "shim" : "core";
    std::printf("%s tile (%d, %d)\n", kind, tile->col, tile->row);
    expect("S2MM channels",
           tm.getNumDMAChannels(tile->col, tile->row, DMAChannelDir::S2MM), 2);
    expect("MM2S channels",
           tm.getNumDMAChannels(tile->col, tile->row, DMAChannelDir::MM2S), 1);
    auto [in, out] = getDMACapacity(tm, *tile);
    expect("getDMACapacity input", in, 2);
    expect("getDMACapacity output", out, 1);
  }
  return failures ? 1 : 0;
}
