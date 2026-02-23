//===- aiecc_aiesim.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// AIE Simulation work folder generation for aiecc.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_AIESIM_H
#define AIECC_AIESIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace aiecc {

/// Configuration for aiesim generation.
struct AiesimConfig {
  bool enabled = false;
  bool verbose = false;
  bool dryRun = false;
};

/// Generate AIE simulation work folder for a device.
/// This creates the sim/ directory structure with:
/// - sim/reports/graph.xpe
/// - sim/arch/aieshim_solution.aiesol
/// - sim/config/scsim_config.json
/// - sim/flows_physical.mlir and sim/flows_physical.json
/// - sim/ps/ps.so (compiled from genwrapper_for_ps.cpp)
/// - aiesim.sh script
///
/// @param moduleOp The MLIR module containing the device
/// @param tmpDirName The temporary directory for output
/// @param devName The device name to generate simulation for
/// @param aieTarget The AIE target (e.g., "aie2")
/// @param aietoolsPath Path to aietools installation
/// @param installPath Path to mlir-aie installation
/// @param config Configuration options
/// @return success() if generation succeeded, failure() otherwise
mlir::LogicalResult generateAiesimWorkFolder(mlir::ModuleOp moduleOp,
                                             llvm::StringRef tmpDirName,
                                             llvm::StringRef devName,
                                             llvm::StringRef aieTarget,
                                             llvm::StringRef aietoolsPath,
                                             llvm::StringRef installPath,
                                             const AiesimConfig &config);

} // namespace aiecc

#endif // AIECC_AIESIM_H
