//===- CtrlPacketDevices.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the control packet delivery passes read off the devices control packet
// groups configure: the MM2S channels and cores marked aiex.started_by, and
// the switch arbiters the routed streams take. A packet holds each arbiter on
// its way until its last word.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETDEVICES_H
#define AIEX_TRANSFORMS_CTRLPACKETDEVICES_H

#include "CtrlPacketClassifier.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <set>
#include <utility>

namespace xilinx::AIEX {

// An arbiter, by its switchbox and index.
using Arbiter = std::pair<AIE::TileID, int>;

// What sends a stream: an MM2S channel of a tile, or its core for channel -1.
using StreamSource = std::pair<AIE::TileID, int>;

// Each configured device, read once.
class ConfiguredDevices {
public:
  explicit ConfiguredDevices(mlir::ModuleOp module) : module(module) {}

  // Whether `source` is marked aiex.started_by.
  bool startGated(mlir::FlatSymbolRefAttr device, StreamSource source) {
    return infoOf(device).gated.contains(source);
  }

  // The arbiters the streams `write`, to `tile`, may start take: a push to
  // an MM2S channel not proved gated starts the channel; a core enable not
  // proved gated starts the core, what aiex.started_by lists it for, and what
  // that lists the gated cores among those for.
  std::set<Arbiter> startedArbiters(mlir::FlatSymbolRefAttr device,
                                    AIE::TileID tile, const Classified &write);

  // The first arbiter of `held` the control packets to `tile` on its
  // controller id take, if any.
  std::optional<Arbiter> controlArbiterIn(mlir::FlatSymbolRefAttr device,
                                          AIE::TileID tile,
                                          const std::set<Arbiter> &held);

private:
  struct Info {
    llvm::DenseSet<StreamSource> gated;
    // The gated sources each core's enable may start.
    llvm::DenseMap<AIE::TileID, llvm::SmallVector<StreamSource>> startedBy;
    llvm::DenseMap<StreamSource, std::set<Arbiter>> sent;
    // The arbiters on each tile's control route, in route order.
    llvm::DenseMap<AIE::TileID, llvm::SmallVector<Arbiter>> control;
  };

  Info &infoOf(mlir::FlatSymbolRefAttr device);

  mlir::ModuleOp module;
  llvm::DenseMap<mlir::StringAttr, std::unique_ptr<Info>> infos;
};

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETDEVICES_H
