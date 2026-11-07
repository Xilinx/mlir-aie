//===- CtrlPacketDevices.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the control packet delivery passes read off the devices control packet
// groups configure: the MM2S channels and cores marked aiex.started_by.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETDEVICES_H
#define AIEX_TRANSFORMS_CTRLPACKETDEVICES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
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

private:
  struct Info {
    llvm::DenseSet<StreamSource> gated;
  };

  Info &infoOf(mlir::FlatSymbolRefAttr device);

  mlir::ModuleOp module;
  llvm::DenseMap<mlir::StringAttr, std::unique_ptr<Info>> infos;
};

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETDEVICES_H
