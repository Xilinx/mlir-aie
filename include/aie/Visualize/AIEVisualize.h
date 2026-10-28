//===- AIEVisualize.h -----------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_VISUALIZE_AIEVISUALIZE_H
#define AIE_VISUALIZE_AIEVISUALIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace xilinx::AIE {

/// How an arrow gets from one tile to another.
enum class ArrowStyle {
  /// A straight line between the nodes it joins, crossing whatever lies
  /// between.
  Straight,
  /// Along the gaps between tiles, never over a tile's contents.
  Channel,
};

/// Which colors to draw in.
enum class ColorScheme {
  /// Dark ink on a white page.
  Light,
  /// Seaborn's mako ramp on black, which separates many flows without any of
  /// them shouting, and keeps the tiles quiet behind them.
  Mako,
};

/// What to draw and how large to draw it. Drawing a flow's switchbox hops
/// needs flows that carry `via` data; run `aie-find-flows=emit-vias=true`
/// first, or contract ports away.
struct VisualizeOptions {
  /// Select one `aie.device` by symbol when the input holds several.
  std::string deviceName;
  /// Select one `aie.runtime_sequence` by symbol when the device holds several.
  std::string runtimeSequenceName;
  /// Omit runtime annotations rather than reading a runtime sequence.
  bool skipRuntime = false;

  /// Draw only these grouped flows; empty draws all of them.
  std::vector<unsigned> onlyFlows;
  /// Node kinds to contract out of the graph: tile, port, dma, buffer,
  /// runtime-argument, call. Contracting nothing draws every hop.
  std::vector<std::string> contractedNodeKinds;

  /// Draw the control packet overlay that aie-generate-column-control-overlay
  /// adds. These rules are marked aie.is_ctrl_pkt_overlay and carry task
  /// completion tokens rather than the data the design moves.
  bool controlOverlay = false;

  /// Add the packet ID to the name of each switchbox port, so that two
  /// packet flows crossing one port become two nodes and contract
  /// separately. Without this they share a node, which shows that they share
  /// the channel.
  bool splitPacketPorts = false;

  /// How arrows are drawn between tiles.
  ArrowStyle arrows = ArrowStyle::Straight;

  /// Which colors to draw in.
  ColorScheme colors = ColorScheme::Light;

  /// Scale edge width by the bytes each flow group moves per iteration.
  bool weighEdges = false;
  double minEdgeWidth = 0.5;
  double maxEdgeWidth = 7.0;

  /// Headline each tile with the kernels it runs, and fill tiles that run the
  /// same kernels alike.
  bool tileRoles = false;
  unsigned maxHeadlineKernels = 4;

  double tileWidth = 2.15;
  double minimumTileHeight = 2.15;
  double tileGapX = 0.85;
  double tileGapY = 0.85;
  double localSpacing = 0.4;

  /// How much smaller than its measurements the picture is drawn. Text and
  /// pen widths keep their size, so lowering this leaves the labels reading
  /// larger against everything else.
  double scale = 0.5;
};

/// Writes a DOT route visualization of the device `options` selects. Reports
/// selection and analysis failures through the module's diagnostic handler.
mlir::LogicalResult emitRouteDot(mlir::ModuleOp module,
                                 llvm::raw_ostream &output,
                                 const VisualizeOptions &options);

} // namespace xilinx::AIE

#endif // AIE_VISUALIZE_AIEVISUALIZE_H
