//===- AIEToConfiguration.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
#define AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <tuple>

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
mlir::LogicalResult generateAndInsertConfigOps(
    mlir::OpBuilder &builder, xilinx::AIE::DeviceOp device,
    llvm::StringRef clElfDir = "",
    AIEToConfigurationOutputType outputType =
        AIEToConfigurationOutputType::Transaction,
    const std::string &blockwrite_prefix = "config_blockwrite_data_",
    bool skipCtrlPktOverlay = false);

// Emit switch-port DISABLE configuration ops (reset value 0) at the current
// insertion point for every data-plane (untagged) master/slave packet-switch
// port of `device` whose (col, row, bundle, index, isSlave) key is NOT in
// `excludePorts`. Routes the disable transactions through the same
// transaction->op conversion `generateAndInsertConfigOps` uses, so with
// `ControlPacket` output each disable becomes an `aiex.npu.control_packet`
// {data = [0]} write at the port's config register, and with `Transaction`
// output each becomes a `write32`/`blockwrite` direct write. Pass the resident
// overlay's ports as `excludePorts` to restrict the teardown to
// exclusively-data ports.
mlir::LogicalResult generateAndInsertSwitchDisableOps(
    mlir::OpBuilder &builder, xilinx::AIE::DeviceOp device,
    const llvm::DenseSet<std::tuple<int, int, int, int, int>> &excludePorts,
    AIEToConfigurationOutputType outputType =
        AIEToConfigurationOutputType::ControlPacket,
    const std::string &blockwrite_prefix = "selfclear_disable_data_",
    bool disableCircuit = false);

// Emit DMA channel RESET configuration ops at the current insertion point for
// every non-shim tile DMA (MemOp/MemTileDMAOp) of `device` -- assert then
// deassert the Ctrl.Reset bit for all channels of the tile. Routes through the
// same transaction->op conversion as generateAndInsertSwitchDisableOps, so
// `ControlPacket` output emits `aiex.npu.control_packet` writes (in-band) and
// `Transaction` output emits `write32`/`maskwrite32` direct writes (OOB).
mlir::LogicalResult generateAndInsertDmaChannelResetOps(
    mlir::OpBuilder &builder, xilinx::AIE::DeviceOp device,
    AIEToConfigurationOutputType outputType =
        AIEToConfigurationOutputType::ControlPacket,
    const std::string &blockwrite_prefix = "selfclear_dma_reset_");

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
