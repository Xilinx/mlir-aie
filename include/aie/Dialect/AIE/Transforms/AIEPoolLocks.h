//===- AIEPoolLocks.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEPOOLLOCKS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEPOOLLOCKS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace xilinx::AIE {

/// Which of a device's objectFIFO pools get locks, read once from its
/// endpoints and its `aiex.dma_channel_reset_for` ops.
class PoolLocks {
public:
  explicit PoolLocks(DeviceOp device);

  /// Whether `pool` gets locks. One with synchronization disabled does not,
  /// nor one that starts full, is never refilled and is not re-armed each
  /// launch: it holds constants its readers have nothing to wait for.
  bool needsLocks(ObjectFifoPoolOp pool) const;
  /// The same rule on a pool's facts: whether synchronization is disabled,
  /// the objects it starts full with, its depth, whether an endpoint fills
  /// it, and whether a reset re-arms its fifo. The placers ask it of the
  /// pools the split will make.
  static bool needsLocks(bool unsynchronized, int64_t full, int64_t depth,
                         bool filled, bool rearmed) {
    // Locks would let a pool of constants be read only as many times as they
    // count, so a second launch would hang, unless a reset re-arms them.
    return !unsynchronized && (full != depth || full <= 0 || filled || rearmed);
  }

  /// The `aiex.dma_channel_reset_for` ops re-arming each fifo, by fifo name.
  const llvm::StringMap<llvm::SmallVector<AIEX::DmaChannelResetForOp>> &
  rearms() const {
    return rearmsByFifo;
  }

private:
  llvm::DenseSet<mlir::Operation *> filled;
  llvm::StringMap<llvm::SmallVector<AIEX::DmaChannelResetForOp>> rearmsByFifo;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEPOOLLOCKS_H
