//===- AIEPacketIdSpace.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEPACKETIDSPACE_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEPACKETIDSPACE_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace xilinx::AIE {

/// The packet ids a device has already spoken for. Every switchbox rule on a
/// stream matches on the id, so two flows sharing one is a routing hazard, and
/// whoever hands out ids reads this first. It counts every holder, whatever
/// stage the design is at: an `aie.packet_flow`, the pinned header of an
/// `aie.route` or of an `aie.objectfifo` not yet split, and an
/// `aie.trace.packet` with an id.
class PacketIdSpace {
public:
  explicit PacketIdSpace(DeviceOp device);

  /// The largest id the target's header field holds.
  int maxId() const { return max; }

  bool isTaken(int id) const { return taken.contains(id); }
  void take(int id) { taken.insert(id); }

  /// Takes and returns the lowest free id at or above `from`, or nothing when
  /// the field is exhausted.
  std::optional<int> takeLowestFrom(int from = 0);

private:
  llvm::SmallDenseSet<int> taken;
  int max;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEPACKETIDSPACE_H
