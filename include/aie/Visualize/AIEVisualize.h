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

/// What to draw and how large to draw it. Physical views need flows that carry
/// `via` data; run `aie-find-flows=emit-vias=true` first.
struct VisualizeOptions {
  /// Select one `aie.device` by symbol when the input holds several.
  std::string deviceName;
  /// Select one `aie.runtime_sequence` by symbol when the device holds several.
  std::string runtimeSequenceName;
  /// Omit runtime annotations rather than reading a runtime sequence.
  bool skipRuntime = false;

  /// Draw only these grouped flows; empty draws all of them.
  std::vector<unsigned> onlyFlows;
  /// Semantic node kinds to contract: tile, port, buffer, runtime-argument,
  /// call. A non-empty list selects a semantic view over a physical one.
  std::vector<std::string> contractedNodeKinds;
  /// Keep flows that share a buffer or an endpoint in separate groups.
  bool noFollowBuffers = false;

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
};

/// Writes a DOT route visualization of the device `options` selects. Reports
/// selection and analysis failures through the module's diagnostic handler.
mlir::LogicalResult emitRouteDot(mlir::ModuleOp module,
                                 llvm::raw_ostream &output,
                                 const VisualizeOptions &options);

} // namespace xilinx::AIE

#endif // AIE_VISUALIZE_AIEVISUALIZE_H
