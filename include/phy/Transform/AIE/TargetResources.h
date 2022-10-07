//===- TargetResources.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H
#define MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H

#include "phy/Transform/Base/TargetResources.h"

#include <set>

#include "mlir/Support/LLVM.h"

namespace xilinx {
namespace phy {
namespace transform {
namespace aie {

using namespace xilinx::phy::connectivity;

class TargetResources : xilinx::phy::transform::TargetResourcesBase {
  int array_height = 8;
  int array_width = 50;

  std::map<std::string, TargetSupport> physical_support = {
      {"lock", {{"states", 2}}},
      {"stream", {{"width_bytes", 32}}},
  };

  std::map<std::string, Capacity> physical_capacity = {
      {"buffer", {{"depth_bytes", 32 * 1024}}},
      {"core", {{"count", 1}, {"depth_bytes", 16 * 1024}}},
      {"lock", {{"count", 16}}},
      {"stream_dma", {{"count", 2}}}};

  std::map<std::string, Capacity> stream_port_capacity = {
      {"Core.I", {{"count", 2}}},
      {"Core.O", {{"count", 2}}},
      {"DMA.I", {{"count", 2}}},
      {"DMA.O", {{"count", 2}}},
      {"FIFO.I", {{"count", 2}}},
      {"FIFO.O", {{"count", 2}}},
      {"North.I", {{"count", 4}}},
      {"North.O", {{"count", 6}}},
      // South.O is the same endpoint of North.I of its south switch
      {"East.I", {{"count", 4}}},
      {"East.O", {{"count", 4}}},
      // West.O is the same endpoint of East.I of its south switch
  };

public:
  bool isShimTile(int col, int row);
  bool isLegalAffinity(int core_col, int core_row, int buf_col, int buf_row);
  std::set<std::pair<int, int>> getAffinity(int col, int row,
                                            std::string neigh_type);

  std::list<VirtualResource>
  getVirtualResourceVertices(std::string virt_key) override;
  std::list<VirtualResource>
  getVirtualResourceNeighbors(VirtualResource &slot) override;
  Capacity getVirtualResourceCapacity(VirtualResource &virt) override;

  TargetSupport getPhysicalResourceSupport(PhysicalResource &phy) override;
  Capacity getPhysicalResourceCapacity(PhysicalResource &phy) override;
  Utilization getPhysicalResourceUtilization(PhysicalResource &phy,
                                             mlir::Operation *vertex) override;
};

} // namespace aie
} // namespace transform
} // namespace phy
} // namespace xilinx

#endif // MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H
