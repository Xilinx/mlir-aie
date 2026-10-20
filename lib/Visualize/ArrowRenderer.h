//===- ArrowRenderer.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_LIB_VISUALIZE_ARROWRENDERER_H
#define AIE_LIB_VISUALIZE_ARROWRENDERER_H

#include "aie/Visualize/AIEVisualize.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

#include <map>
#include <optional>
#include <string>
#include <utility>

namespace xilinx::AIE {

struct LayoutGeometry;

using Point = std::pair<double, double>;

/// One arrow to draw, named by the tiles it passes through. A physical hop
/// names both of its tiles (the same tile twice when a switchbox turns a flow
/// around inside itself); a contracted edge names only its endpoints and
/// leaves the way between them to the renderer.
struct ArrowRequest {
  unsigned id;
  /// The DOT nodes at either end, for a renderer that places them itself.
  std::string sourceNode, destNode;
  llvm::SmallVector<std::pair<int, int>> tiles;
  /// Where the arrow leaves and enters, when the caller has pinned those
  /// points itself. A renderer that places its own endpoints ignores these.
  std::optional<Point> pinnedSource;
  std::optional<Point> pinnedDest;
  unsigned flowGroup;
  double width;
};

/// A drawn arrow: a cubic Bezier chain, as graphviz's edge `pos` spells it.
/// `points` holds 3n+1 control points; straight runs repeat their endpoints.
struct ArrowPath {
  unsigned id;
  llvm::SmallVector<Point> points;
};

/// Decides where arrows go. Laying out every arrow in one call is what lets an
/// implementation separate the ones that share a channel and order them to
/// cross less; neither is decidable one arrow at a time.
class ArrowRenderer {
public:
  virtual ~ArrowRenderer();

  /// Extra space this renderer needs between tiles, in inches, or nothing to
  /// keep the caller's spacing.
  virtual std::optional<double> preferredTileGap() const {
    return std::nullopt;
  }

  /// Where ports sit, for renderers that move them off the tile. An absent
  /// entry leaves the port where the caller put it.
  virtual std::map<std::string, Point>
  portPositions(llvm::ArrayRef<ArrowRequest> requests,
                const LayoutGeometry &layout) {
    return {};
  }

  virtual llvm::SmallVector<ArrowPath>
  route(llvm::ArrayRef<ArrowRequest> requests,
        const LayoutGeometry &layout) = 0;
};

/// Draws nothing, leaving every arrow a straight line between the nodes it
/// joins. What the viewer has always drawn.
std::unique_ptr<ArrowRenderer> createStraightArrowRenderer();

/// Keeps arrows in the gaps between tiles: an arrow runs along a ring around
/// its tile and crosses the channel to the next one, so none is drawn over a
/// tile's contents. Arrows sharing a stretch are spread onto tracks, and
/// corners are rounded.
std::unique_ptr<ArrowRenderer> createChannelArrowRenderer();

} // namespace xilinx::AIE

#endif // AIE_LIB_VISUALIZE_ARROWRENDERER_H
