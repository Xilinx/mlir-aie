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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace xilinx::AIE {

struct LayoutGeometry;

using Point = std::pair<double, double>;

/// What an arrow ends on, which decides where a renderer may put it.
enum class EndKind {
  /// A switchbox port: the stream side of a tile, which only ever reaches a
  /// neighbouring switchbox or its own tile's DMA.
  Switch,
  /// A DMA channel, facing the switchbox on one side and the tile's buffers
  /// on the other.
  DMA,
  /// Something inside a tile: a buffer, or a kernel call.
  Content,
  /// Something outside the array, such as a runtime sequence argument.
  External,
};

/// Which way a switchbox port faces, so that it can be put on that side of
/// the switchbox and a flow passing straight through need not meet the ports
/// it passes.
enum class Facing { North, East, South, West, Local };

struct ArrowEnd {
  std::string node;
  std::pair<int, int> tile{0, 0};
  EndKind kind = EndKind::Content;
  /// Stacking order among a tile's ends of the same kind.
  int slot = 0;
  /// Where the caller would have put it, used when a renderer places nothing.
  Point pinned{0, 0};
  /// Half the box drawn there, so an arrow can stop at its edge rather than
  /// run to the middle and over the text.
  double halfWidth = 0.0;
  double halfHeight = 0.0;
  Facing facing = Facing::Local;
};

/// One arrow to draw.
struct ArrowRequest {
  unsigned id = 0;
  ArrowEnd from, to;
  unsigned flowGroup = 0;
  double width = 1.0;
};

/// A drawn arrow: a cubic Bezier chain, as graphviz's edge `pos` spells it.
/// `points` holds 3n+1 control points; straight runs repeat their endpoints.
struct ArrowPath {
  unsigned id;
  llvm::SmallVector<Point> points;
};

/// Decides where arrows go, and where the things they join sit. Laying out
/// every arrow in one call is what lets an implementation separate the ones
/// that share a channel and order them to cross less; neither is decidable one
/// arrow at a time.
class ArrowRenderer {
public:
  virtual ~ArrowRenderer();

  /// Extra space this renderer needs between tiles, in inches, or nothing to
  /// keep the caller's spacing.
  virtual std::optional<double> preferredTileGap() const {
    return std::nullopt;
  }

  /// How wide a tile has to be for the arrows drawn inside it to have room,
  /// given the most buffers and calls any one tile holds.
  virtual std::optional<double> preferredTileWidth(unsigned contents) const {
    return std::nullopt;
  }

  /// The widest a line may be drawn, in points, before it reaches the one
  /// beside it. A renderer that puts lines on tracks knows how far apart it
  /// put them; nothing else does.
  virtual std::optional<double> widestLine() const { return std::nullopt; }

  /// Where the nodes arrows end on should sit. An absent entry leaves a node
  /// where the caller put it.
  virtual std::map<std::string, Point>
  nodePositions(llvm::ArrayRef<ArrowRequest> requests,
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

/// Keeps arrows in the gaps between tiles. A tile's switchbox ports gather at
/// the channel junction off its corner, from which all four channels are
/// reachable, so a hop to a neighbour is one straight run and a hop the
/// switchbox makes within itself never crosses the tile. DMA channels stack
/// along the tile's edge, a short stub from the buffers they serve.
std::unique_ptr<ArrowRenderer> createChannelArrowRenderer();

} // namespace xilinx::AIE

#endif // AIE_LIB_VISUALIZE_ARROWRENDERER_H
