//===- aiecc_aiesim.h - AIE Simulation support for aiecc --------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIE simulation work folder generation for the C++ aiecc compiler driver.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_AIESIM_H
#define AIECC_AIESIM_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace xilinx {
namespace aiecc {

/// Configuration options passed from the main aiecc driver to aiesim
/// generation. Paths use std::string for lifetime management.
struct AiesimConfig {
  bool enabled = false;
  bool compileHost = false;
  bool verbose = false;
  bool dryRun = false;
  std::string hostTarget = "x86_64-linux-gnu";
  std::string aietoolsPath;
  std::string installPath;
  /// Host args for aiesim ps.so compilation: all host args except -o (source
  /// files, -I, -L, -l flags).
  std::vector<std::string> hostArgs;
};

/// Generate aie_inc.cpp file for host compilation or aiesim.
/// Uses aie-translate --aie-generate-xaie.
mlir::LogicalResult generateAieIncCpp(mlir::ModuleOp moduleOp,
                                      llvm::StringRef tmpDirName,
                                      llvm::StringRef devName,
                                      const AiesimConfig &config);

/// Generate AIE simulation work folder.
/// Creates sim directory structure, generates simulation files, and builds
/// ps.so.
mlir::LogicalResult generateAiesim(mlir::ModuleOp moduleOp,
                                   llvm::StringRef tmpDirName,
                                   llvm::StringRef devName,
                                   llvm::StringRef aieTarget,
                                   const AiesimConfig &config);

/// Get AIE target defines for host/sim compilation.
llvm::SmallVector<std::string> getAieTargetDefines(llvm::StringRef aieTarget);

} // namespace aiecc
} // namespace xilinx

#endif // AIECC_AIESIM_H
