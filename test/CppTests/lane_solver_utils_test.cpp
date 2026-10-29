//===- lane_solver_utils_test.cpp -----------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaneSolverUtils.h"

#include <functional>
#include <set>

using xilinx::AIE::detail::LaneRun;

int main() {
  uint64_t visited = 0;
  bool capped = false;
  unsigned executed = 0;
  unsigned leaves = 0;
  std::function<void(unsigned)> visit = [&](unsigned depth) {
    if (!xilinx::AIE::detail::enterSearchState(5, visited, capped))
      return;
    ++executed;
    if (depth == 3) {
      ++leaves;
      return;
    }
    visit(depth + 1);
    visit(depth + 1);
  };
  visit(0);
  if (visited != 5 || executed != 5 || leaves != 2 || !capped)
    return 1;

  const LaneRun vertical{1, 0, false, 2.0, 0.0, 4.0};
  const LaneRun firstHorizontal{0, 0, true, 1.0, 0.0, 4.0};
  const LaneRun secondHorizontal{0, 2, true, 3.0, 0.0, 4.0};
  std::set<std::array<unsigned, 2>> crossingPairs;
  auto firstCrossing = xilinx::AIE::detail::crossingRoutePair(
      firstHorizontal, vertical, false, 1e-9);
  auto secondCrossing = xilinx::AIE::detail::crossingRoutePair(
      vertical, secondHorizontal, false, 1e-9);
  if (!firstCrossing || !secondCrossing)
    return 2;
  crossingPairs.insert(*firstCrossing);
  crossingPairs.insert(*secondCrossing);
  if (crossingPairs.size() != 1 ||
      *crossingPairs.begin() != (std::array<unsigned, 2>{0, 1}))
    return 3;
}