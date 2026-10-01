//===- AIERoutingDiagnostics.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// How the router names ports and flows in its diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEROUTINGDIAGNOSTICS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEROUTINGDIAGNOSTICS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace xilinx::AIE {

/// Names a port the way diagnostics do, e.g. "DMA:1".
std::string describePort(Port port);

/// Names a tile port the way diagnostics do, e.g. "(0, 2) DMA:1".
std::string describeTilePort(TileID tile, Port port);

/// Lists `names` as "A, B and C", the first `shown` of them followed by
/// "N more" if there are more.
std::string joinNames(llvm::ArrayRef<std::string> names, size_t shown = 4);

/// Says that the packet flows from `sources` keep the route they take alone,
/// as priority_route asks.
std::string describePrioritized(llvm::ArrayRef<std::string> sources);

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEROUTINGDIAGNOSTICS_H
