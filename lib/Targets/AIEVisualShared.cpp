//===- AIEVisualShared.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIEVisualShared.h"

using namespace xilinx::AIE;

namespace xilinx::AIE {

static const std::map<AIETileType, ColorScheme> kTileColorSchemes = {
    {AIETileType::CoreTile, {"#DAE8FC", "#6C8EBF"}},    // Blue
    {AIETileType::MemTile, {"#D5E8D4", "#82B366"}},     // Green
    {AIETileType::ShimNOCTile, {"#FFE6CC", "#D79B00"}}, // Orange
    {AIETileType::ShimPLTile, {"#E1D5E7", "#9673A6"}},  // Purple
};

// Color-blind friendly palette for routes
// From Wong's color-blind friendly palette
static const std::vector<std::string> kRouteColorPalette = {
    "steelblue3", // Steel blue
    "#D55E00",    // Vermillion
    "#009E73",    // Bluish green
    "#F0E442",    // Yellow
    "#0072B2",    // Blue
    "#E69F00",    // Orange
    "#56B4E9",    // Sky blue
    "#CC79A7"     // Reddish purple
};

// Special color for shared memory connections
static const std::string kSharedMemoryColor = "#CC0000"; // Red
static const std::string kSharedMemoryStyle = "dashed";

const ColorScheme &getTileColorScheme(AIETileType type) {
  auto it = kTileColorSchemes.find(type);
  if (it != kTileColorSchemes.end())
    return it->second;

  // Default color for unknown tile types
  static const ColorScheme defaultColor = {"#F5F5F5", "#666666"}; // Grey
  return defaultColor;
}

std::string getRouteColor(int routeIndex) {
  return kRouteColorPalette[routeIndex % kRouteColorPalette.size()];
}

std::string getSharedMemoryColor() { return kSharedMemoryColor; }

std::string getSharedMemoryStyle() { return kSharedMemoryStyle; }

void emitDOTHeader(llvm::raw_ostream &output, llvm::StringRef graphName,
                   llvm::StringRef layout) {
  output << "digraph " << graphName << " {\n";
  output << "  layout=" << layout << ";\n";
  output << "  graph[outputMode=nodesfirst; splines=false];\n";
  output << "  node[shape=square; style=filled; penwidth=2.5; fontsize=35];\n";
  output << "\n";
}

void emitDOTFooter(llvm::raw_ostream &output) { output << "}\n"; }

std::string getSwitchboxNodeName(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row) + "_sb";
}

std::string getCoreNodeName(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row) + "_core";
}

std::string getBufferNodeName(int col, int row) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row) + "_memory";
}

std::string getDMANodeName(int col, int row, bool isS2MM, int channel) {
  return "tile_" + std::to_string(col) + "_" + std::to_string(row) + "_dma_" +
         (isS2MM ? "s2mm_" : "mm2s_") + std::to_string(channel);
}

TileID getNextCoords(int col, int row, WireBundle bundle) {
  switch (bundle) {
  case WireBundle::North:
    return {col, row + 1};
  case WireBundle::South:
    return {col, row - 1};
  case WireBundle::East:
    return {col + 1, row};
  case WireBundle::West:
    return {col - 1, row};
  default:
    return {col, row};
  }
}

const double kSwitchboxWidth = 1.5;
const double kTileWidth = 5.0;
const double kTileHeight = 4.5;
const double kInternalGap = 0.2;
const double kGridWidth = kTileWidth + kSwitchboxWidth + 2 * kInternalGap;
const double kGridHeight = kTileHeight + kSwitchboxWidth + 2 * kInternalGap;
const double kPortSize = 0.25;
const double kDMAPortHeight = 1.0;
const double kChannelSpacing = 0.15;

} // namespace xilinx::AIE
