//===- AIEPlacer.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains the interface for tile placement algorithms.
// Placers assign physical tile coordinates to logical tiles.
//===----------------------------------------------------------------------===//

#ifndef AIE_PLACER_H
#define AIE_PLACER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

namespace xilinx::AIE {

// maps logical tile operations to physical coordinates
using PlacementResult = llvm::DenseMap<mlir::Operation *, TileID>;

// Track available tiles and resource usage
struct TileAvailability {
  std::vector<TileID> compTiles;
  std::vector<TileID> nonCompTiles; // Memory and shim tiles

  llvm::DenseMap<TileID, int> inputChannelsUsed;
  llvm::DenseMap<TileID, int> outputChannelsUsed;

  void removeTile(TileID tile, AIETileType type);
};

// Abstract placer interface
class Placer {
public:
  Placer() = default;
  virtual ~Placer() = default;

  virtual void initialize(DeviceOp device,
                          const AIETargetModel &targetModel) = 0;

  virtual mlir::LogicalResult
  place(llvm::ArrayRef<mlir::Operation *> logicalTiles,
        llvm::ArrayRef<mlir::Operation *> objectFifos,
        llvm::ArrayRef<mlir::Operation *> cores, PlacementResult &result) = 0;

  virtual llvm::StringRef getName() const = 0;
};

// Sequential placement algorithm
//
// Places logical tiles to physical tiles using a simple strategy:
// - Compute tiles: Sequential row-major placement
// - Memory/shim tiles: Channel capacity placement near common column
//
// Core-to-core connections are NOT validated because SequentialPlacer
// doesn't account for shared memory optimization.
//
// Shim/Mem tiles with identical placement constraints and sufficient
// DMA capacity are merged to the same physical tile.
class SequentialPlacer : public Placer {
public:
  SequentialPlacer(std::optional<int> coresPerCol = std::nullopt)
      : coresPerCol(coresPerCol) {}

  void initialize(DeviceOp device, const AIETargetModel &targetModel) override;

  mlir::LogicalResult place(llvm::ArrayRef<mlir::Operation *> logicalTiles,
                            llvm::ArrayRef<mlir::Operation *> objectFifos,
                            llvm::ArrayRef<mlir::Operation *> cores,
                            PlacementResult &result) override;

  llvm::StringRef getName() const override { return "sequential_placer"; }

private:
  std::optional<int> coresPerCol;
  TileAvailability availability;
  const AIETargetModel *targetModel;
  DeviceOp device;

  int getCommonColumn(const PlacementResult &result);

  std::optional<TileID> findTileWithCapacity(int targetCol,
                                             std::vector<TileID> &tiles,
                                             int requiredInputChannels,
                                             int requiredOutputChannels);

  void updateChannelUsage(TileID tile, bool isOutput, int numChannels);

  bool hasAvailableChannels(TileID tile, int inputChannels, int outputChannels);

  mlir::LogicalResult validateAndUpdateChannelUsage(
      LogicalTileOp logicalTile, TileID tile,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements,
      bool isConstrained);
};

// PlacementAnalysis integrates the Pathfinder class into the MLIR
// environment.
class PlacementAnalysis {
public:
  PlacementAnalysis() : placer(std::make_shared<SequentialPlacer>()) {}
  explicit PlacementAnalysis(std::shared_ptr<Placer> p)
      : placer(std::move(p)) {}

  mlir::LogicalResult runAnalysis(DeviceOp &device);

  std::optional<TileID> getPlacement(mlir::Operation *logicalTile) const;

  Placer &getPlacer() { return *placer; }

private:
  std::shared_ptr<Placer> placer;
  PlacementResult result;
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
