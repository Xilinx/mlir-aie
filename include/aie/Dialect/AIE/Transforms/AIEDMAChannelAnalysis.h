//===- AIEDMAChannelAnalysis.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"

namespace xilinx::AIE {

/// Which DMA channels of each tile are already spoken for, so that a channel is
/// handed out at most once across everything that programs one.
class DMAChannelAnalysis {
  /// A channel or stream port is either spoken for or free, so membership is
  /// the whole state.
  mlir::DenseSet<std::tuple<mlir::Value, DMAChannelDir, int>> usedChannels;
  mlir::DenseSet<std::tuple<mlir::Value, DMAChannelDir, int>> usedStreams;

public:
  DMAChannelAnalysis(DeviceOp &device);

  /// Next free channel of `tile` in `dir`, or -1 when the tile has none left.
  /// A channel reaching an adjacent MemTile's memory must come from the lower
  /// half of the range, which only a placed tile can bound.
  int getDMAChannelIndex(TileLike tile, DMAChannelDir dir,
                         bool requiresAdjacentTileAccessChannels);

  /// Claim `channel` for (`tile`, `dir`) so first-free assignment cannot take
  /// it. Returns the channel, or -1 when it is out of range or already
  /// claimed; the caller reports, since it knows which endpoint asked.
  int reservePinnedChannel(TileLike tile, DMAChannelDir dir, int channel);

  /// Claim a raw stream port, reporting on `tile` when it is already taken.
  void checkAIEStreamIndex(TileLike tile, DMAChannel chan);
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
