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
                         mlir::Operation *insertionPoint);

} // namespace xilinx::AIE

#endif // AIE_CONVERSION_AIETOCONFIGURATION_AIETOCONFIGURATION_H
