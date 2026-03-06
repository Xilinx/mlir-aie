//===- AIEVisualShared.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TARGETS_AIEVISUALSHARED_H
#define AIE_TARGETS_AIEVISUALSHARED_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
#include <vector>

namespace xilinx::AIE {

/// Color scheme for visualization (fill and stroke colors)
struct ColorScheme {
  std::string fill;
  std::string stroke;
};

/// Grid position for DOT layout
struct GridPosition {
  double x;
  double y;
};

/// Get color scheme for a tile type
const ColorScheme &getTileColorScheme(AIETileType type);

/// Get color for a route (cycles through color-blind friendly palette)
std::string getRouteColor(int routeIndex);

/// Get color for shared-memory connections (distinct purple)
std::string getSharedMemoryColor();

/// Get style for shared-memory connections (dashed)
std::string getSharedMemoryStyle();

/// Emit DOT file header with graph name and layout engine
void emitDOTHeader(llvm::raw_ostream &output, llvm::StringRef graphName,
                   llvm::StringRef layout = "neato");

/// Emit DOT file footer
void emitDOTFooter(llvm::raw_ostream &output);

/// Get node name for switchbox
std::string getSwitchboxNodeName(int col, int row);

/// Get node name for core component (or shim tile box)
std::string getCoreNodeName(int col, int row);

/// Get node name for memory/buffer component
std::string getBufferNodeName(int col, int row);

/// Get node name for DMA port
/// @param isS2MM true for S2MM (stream-to-memory, input), false for MM2S
/// (memory-to-stream, output)
/// @param channel DMA channel index
std::string getDMANodeName(int col, int row, bool isS2MM, int channel);

/// Get the next tile coordinates when moving in the specified direction
TileID getNextCoords(int col, int row, WireBundle bundle);

// Grid layout constants for routing and placement visualization
extern const double kSwitchboxWidth;
extern const double kTileWidth;
extern const double kTileHeight;
extern const double kInternalGap;
extern const double kGridWidth;
extern const double kGridHeight;
extern const double kPortSize;
extern const double kDMAPortHeight;
extern const double kChannelSpacing;

} // namespace xilinx::AIE

#endif // AIE_TARGETS_AIEVISUALSHARED_H
