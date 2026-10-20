//===- LayoutGeometry.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H
#define AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xilinx::AIE {

/// Where the tiles sit, in inches. Columns are evenly spaced; rows are not,
/// because a row is as tall as its tallest tile.
struct LayoutGeometry {
  /// Graphviz reads a pinned `pos` in inches when it is laying the graph out
  /// and in points when it is not. A renderer that supplies its own edge
  /// geometry needs the latter, since only `nop2` leaves edges alone.
  bool preLaidOut = false;

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

  /// A position as graphviz should read it, in whichever units apply.
  std::string pos(double x, double y) const {
    double scale = preLaidOut ? 72.0 : 1.0;
    return std::to_string(x * scale) + "," + std::to_string(y * scale) +
           (preLaidOut ? "" : "!");
  }

  /// The band a row reserves, which is as tall as its tallest tile. Tiles
  /// hang from the top of their band and switchboxes sit below it, so both
  /// line up across a row however tall any one tile turns out to be.
  double rowTop(int row) const {
    return rowCenters.at(row) + rowHeights.at(row) / 2;
  }
  double rowBottom(int row) const {
    return rowCenters.at(row) - rowHeights.at(row) / 2;
  }

  double tileLeft(int col) const { return tileCenterX(col) - tileWidth / 2; }
  double tileRight(int col) const { return tileCenterX(col) + tileWidth / 2; }
  double tileTop(int col, int row) const { return rowTop(row); }
  double tileBottom(int col, int row) const {
    return rowTop(row) - tileHeight(col, row);
  }
  double tileCenterY(int col, int row) const {
    return rowTop(row) - tileHeight(col, row) / 2;
  }

  /// The free span between two columns, and between two rows, that arrows may
  /// run in without touching a tile.
  double channelX(int col) const { return tileLeft(col) - tileGapX / 2; }
  double channelY(int row) const { return rowBottom(row) - tileGapY / 2; }
};

} // namespace xilinx::AIE

#endif // AIE_LIB_VISUALIZE_LAYOUTGEOMETRY_H
