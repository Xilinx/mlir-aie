//===- LaneSolverUtils.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_LIB_VISUALIZE_LANESOLVERUTILS_H
#define AIE_LIB_VISUALIZE_LANESOLVERUTILS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace xilinx::AIE::detail {

struct LaneRun {
  unsigned route = 0;
  unsigned segment = 0;
  bool horizontal = false;
  double cross = 0.0;
  double low = 0.0;
  double high = 0.0;
};

inline std::optional<std::array<unsigned, 2>>
crossingRoutePair(const LaneRun &first, const LaneRun &second, bool exempt,
                  double epsilon) {
  if (exempt || first.horizontal == second.horizontal)
    return std::nullopt;
  const LaneRun &horizontal = first.horizontal ? first : second;
  const LaneRun &vertical = first.horizontal ? second : first;
  if (horizontal.low + epsilon < vertical.cross &&
      vertical.cross < horizontal.high - epsilon &&
      vertical.low + epsilon < horizontal.cross &&
      horizontal.cross < vertical.high - epsilon)
    return std::array{std::min(first.route, second.route),
                      std::max(first.route, second.route)};
  return std::nullopt;
}

inline bool enterSearchState(uint64_t cap, uint64_t &visited, bool &capped) {
  if (visited >= cap) {
    capped = true;
    return false;
  }
  ++visited;
  return true;
}

} // namespace xilinx::AIE::detail

#endif