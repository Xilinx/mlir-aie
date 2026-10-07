//===- AIERoutingDiagnostics.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIERoutingDiagnostics.h"

#include "llvm/Support/FormatVariadic.h"

using namespace xilinx;

std::string AIE::describePort(Port port) {
  return llvm::formatv("{0}:{1}", stringifyWireBundle(port.bundle),
                       port.channel);
}

std::string AIE::describeTilePort(TileID tile, Port port) {
  return llvm::formatv("({0}, {1}) {2}", tile.col, tile.row,
                       describePort(port));
}

std::string AIE::joinNames(llvm::ArrayRef<std::string> names, size_t shown) {
  size_t more = names.size() > shown ? names.size() - shown : 0;
  llvm::ArrayRef<std::string> listed = names.drop_back(more);
  std::string list;
  for (auto [i, name] : llvm::enumerate(listed))
    list += (i == 0                            ? ""
             : i + 1 == listed.size() && !more ? " and "
                                               : ", ") +
            name;
  if (more)
    list += " and " + std::to_string(more) + " more";
  return list;
}

std::string AIE::describePrioritized(llvm::ArrayRef<std::string> sources) {
  return "packet flows from " + joinNames(sources) +
         " are prioritized (priority_route) in a design a control-packet "
         "reload configures (has_ctrl_pkt_overlay), so they keep the route "
         "they take alone, as in @ctrl_pkt_overlay";
}
