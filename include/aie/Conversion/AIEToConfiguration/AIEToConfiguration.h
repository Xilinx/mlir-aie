//===- AIEToConfiguration.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
#define AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx::AIE {

class DeviceOp;

// Enum for specifying which tile types to reset
enum class ResetTileType : unsigned {
  None = 0,
  ShimNOC = 1 << 0,
  MemTile = 1 << 1,
  CoreTile = 1 << 2,
  All = ShimNOC | MemTile | CoreTile
};

// Allow bitwise OR operations on ResetTileType
inline ResetTileType operator|(ResetTileType lhs, ResetTileType rhs) {
  return static_cast<ResetTileType>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline ResetTileType operator&(ResetTileType lhs, ResetTileType rhs) {
  return static_cast<ResetTileType>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

inline bool hasFlag(ResetTileType value, ResetTileType flag) {
  return static_cast<unsigned>(value & flag) != 0;
}

// Enum for specifying when to reset
enum class ResetMode {
  Never,     // Never perform reset
  IfUsed,    // Reset only if the tile is used in the device
  IfChanged, // Reset only if the tile configuration changed from previous
  Always     // Reset all tiles of the specified type
};

// Configuration for different reset operations
struct ResetConfig {
  ResetTileType tileType;
  ResetMode mode;
  
  ResetConfig(ResetTileType tt = ResetTileType::None, ResetMode m = ResetMode::Never)
    : tileType(tt), mode(m) {}
};

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToTransactionPass();

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToControlPacketsPass();

std::optional<mlir::ModuleOp>
convertTransactionBinaryToMLIR(mlir::MLIRContext *ctx,
                               std::vector<uint8_t> &binary);

// Generate transaction binary and insert configuration operations at a specific point
mlir::LogicalResult
generateAndInsertConfigOps(xilinx::AIE::DeviceOp device,
                          mlir::Operation *insertionPoint,
                          llvm::StringRef clElfDir = "");

// Generate reset operations for tiles in a device and insert them at a specific point
mlir::LogicalResult
generateAndInsertResetOps(xilinx::AIE::DeviceOp device,
                         mlir::Operation *insertionPoint,
                         ResetConfig dmaConfig = ResetConfig(),
                         ResetConfig switchConfig = ResetConfig(),
                         ResetConfig lockConfig = ResetConfig());

// Version with previous device for change detection
mlir::LogicalResult
generateAndInsertResetOps(xilinx::AIE::DeviceOp device,
                         mlir::Operation *insertionPoint,
                         ResetConfig dmaConfig,
                         ResetConfig switchConfig,
                         ResetConfig lockConfig,
                         xilinx::AIE::DeviceOp previousDevice);

} // namespace xilinx::AIE

#endif // AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
