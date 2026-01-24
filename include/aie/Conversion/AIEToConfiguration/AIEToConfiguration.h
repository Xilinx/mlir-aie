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

// --------------------------------------------------------------------------
// Device configuration
// --------------------------------------------------------------------------

// an enum to represent the output type of the transaction binary
enum AIEToConfigurationOutputType {
  Transaction,
  ControlPacket,
};

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToTransactionPass();

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createConvertAIEToControlPacketsPass();

std::optional<mlir::ModuleOp>
convertTransactionBinaryToMLIR(mlir::MLIRContext *ctx,
                               std::vector<uint8_t> &binary);

// Generate transaction binary and insert configuration operations at the
// current insertion point
mlir::LogicalResult
generateAndInsertConfigOps(mlir::OpBuilder &builder,
                           xilinx::AIE::DeviceOp device,
                           llvm::StringRef clElfDir = "",
                           AIEToConfigurationOutputType outputType =
                               AIEToConfigurationOutputType::Transaction,
                           std::string blockwrite_prefix = "config_blockwrite_data_");

// --------------------------------------------------------------------------
// Device reset
// --------------------------------------------------------------------------

// Enum for specifying which tile types to reset
enum class ResetTileType : unsigned {
  None = 0,
  ShimNOC = 1 << 0,
  MemTile = 1 << 1,
  CoreTile = 1 << 2,
  All = ShimNOC | MemTile | CoreTile
};

inline bool hasFlag(ResetTileType value, ResetTileType flag) {
  return (static_cast<unsigned>(value) & static_cast<unsigned>(flag)) != 0;
}

// Enum for specifying when to reset
enum class ResetMode {
  Never,             // Never perform reset
  IfUsed,            // Reset only if the tile is used in the device
  IfUsedFineGrained, // Reset only individual locks/connections that are used
  IfChanged, // Reset only if the tile configuration changed from previous
  IfChangedFineGrained, // Reset only individual locks/connections that changed
  Always                // Reset all tiles of the specified type
};

// Configuration for different reset operations
struct ResetConfig {
  ResetTileType tileType;
  ResetMode mode;

  ResetConfig(ResetTileType tt = ResetTileType::None,
              ResetMode m = ResetMode::Never)
      : tileType(tt), mode(m) {}
};

// Insert reset operations at the current insertion point
mlir::LogicalResult generateAndInsertResetOps(
    mlir::OpBuilder &builder, xilinx::AIE::DeviceOp device,
    ResetConfig dmaConfig, ResetConfig switchConfig, ResetConfig lockConfig,
    ResetConfig coreConfig, xilinx::AIE::DeviceOp previousDevice);

} // namespace xilinx::AIE

#endif // AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
