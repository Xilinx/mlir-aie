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
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, int>
      channelsPerTile;
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, int>
      aieStreamsPerTile;

public:
  DMAChannelAnalysis(DeviceOp &device);

  /// Next free channel of `tileOp` in `dir`, or -1 when the tile has none left.
  /// A channel reaching an adjacent MemTile's memory must come from the lower
  /// half of the range.
  int getDMAChannelIndex(TileOp tileOp, DMAChannelDir dir,
                         bool requiresAdjacentTileAccessChannels);

  /// Claim `channel` for (`tileOp`, `dir`) so first-free assignment cannot take
  /// it. Returns -1 when the channel is out of range or already claimed.
  int reservePinnedChannel(TileOp tileOp, DMAChannelDir dir, int channel);

  /// Claim a raw stream port, reporting on `tileOp` when it is already taken.
  void checkAIEStreamIndex(TileOp tileOp, DMAChannel chan);
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
