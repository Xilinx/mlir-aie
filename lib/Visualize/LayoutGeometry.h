//===- LayoutGeometry.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H
#define AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H

#include <map>
#include <utility>
#include <vector>

namespace xilinx::AIE {

/// Where the tiles sit, in inches. Columns are evenly spaced; rows are not,
/// because a row is as tall as its tallest tile.
struct LayoutGeometry {
  double tileWidth = 0.0;
  double tileGapX = 0.0;
  double tileGapY = 0.0;
  std::vector<double> rowHeights;
  std::vector<double> rowCenters;
  std::map<std::pair<int, int>, double> ownHeights;

  double tileCenterX(int col) const { return col * (tileWidth + tileGapX); }
  double tileCenterY(int row) const { return rowCenters.at(row); }
  double tileHeight(int row) const { return rowHeights.at(row); }

  /// A tile is only as tall as its own contents. Sizing every tile to the
  /// tallest one in its row shrinks the text in all the others to fit a
  /// neighbour's buffer list. Rows still reserve the tallest tile's height so
  /// that boxes never overlap.
  double tileHeight(int col, int row) const {
    auto found = ownHeights.find({col, row});
    return found == ownHeights.end() ? rowHeights.at(row) : found->second;
  }

  double tileLeft(int col) const { return tileCenterX(col) - tileWidth / 2; }
  double tileRight(int col) const { return tileCenterX(col) + tileWidth / 2; }
  double tileBottom(int col, int row) const {
    return tileCenterY(row) - tileHeight(col, row) / 2;
  }
  double tileTop(int col, int row) const {
    return tileCenterY(row) + tileHeight(col, row) / 2;
  }
};

} // namespace xilinx::AIE

#endif // AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H
