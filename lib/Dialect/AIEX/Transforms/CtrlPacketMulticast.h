//===- CtrlPacketMulticast.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The control multicasts of a configured device's overlay, as
// aie-generate-column-control-overlay records them in aie.ctrl_pkt_multicast,
// with the routes the overlay gives them.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETMULTICAST_H
#define AIEX_TRANSFORMS_CTRLPACKETMULTICAST_H

#include "CtrlPacketDevices.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <memory>
#include <set>
#include <utility>

namespace xilinx::AIEX {

class OverlayMulticasts {
public:
  struct Multicast {
    int32_t id;
    // The column whose control channel it leaves by.
    int column;
    llvm::SmallVector<AIE::TileID> tiles;
  };

  // Multicasts may share an id, so each is known by its column.
  struct Info {
    std::map<int, Multicast> multicasts;
    // The column of each tile's multicast.
    std::map<AIE::TileID, int> byTile;
    // The switchboxes and input ports a packet passes from the control channel
    // of its tile's column, per (tile, packet id).
    std::map<std::pair<AIE::TileID, int32_t>,
             llvm::SmallVector<std::pair<AIE::TileID, AIE::Port>>>
        trails;
    // The arbiters each multicast's packets hold, by its column.
    std::map<int, std::set<Arbiter>> trees;
    // The controller id each tile's own packets carry.
    std::map<AIE::TileID, int32_t> controllerIds;

    // Whether `tile`'s own packets reach it in order after `m`'s: both leave
    // by one control channel and pass the same input ports.
    bool sharesTrail(const Multicast &m, AIE::TileID tile) const;

    // The multicast `packet` goes on: the one on its multicast_id whose lowest
    // tile it names. Null if none is.
    const Multicast *multicastOf(NpuControlPacketOp packet) const;
  };

  explicit OverlayMulticasts(mlir::ModuleOp module) : module(module) {}

  // The multicasts of the overlay `configured` references; null when it has
  // none.
  const Info *of(mlir::FlatSymbolRefAttr configured);

private:
  mlir::ModuleOp module;
  // By overlay symbol; null when the overlay records no multicast.
  llvm::DenseMap<mlir::StringAttr, std::unique_ptr<Info>> infos;
};

// The tiles `packet` reaches: its multicast's in `info`, or its own.
llvm::SmallVector<AIE::TileID>
tilesReached(NpuControlPacketOp packet, const OverlayMulticasts::Info *info);

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETMULTICAST_H
