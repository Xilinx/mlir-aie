//===- ChannelArrowRenderer.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Draws every arrow in the gaps between tiles.
//
// Each tile is circled by a ring in the gap around it. An arrow leaves its
// source tile's ring, crosses the channel to the next tile's ring, and so on,
// so no arrow is ever drawn over a tile's contents. Arrows sharing a stretch
// of ring or channel are spread onto separate tracks, and the tracks are
// ordered to reduce crossings. Corners are drawn as quarter arcs.
//
//===----------------------------------------------------------------------===//

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>

using namespace llvm;

namespace xilinx::AIE {
namespace {

/// A circle's quarter arc is within half a percent of a cubic Bezier whose
/// control points lie this far along the tangents.
constexpr double kArcControl = 0.5523;

/// Distance from a tile's edge to the innermost ring track, and between
/// adjacent tracks, in inches. Tracks have to clear the widest arrow.
constexpr double kRingInset = 0.18;
constexpr double kTrackPitch = 0.115;
constexpr double kCornerRadius = 0.09;

/// Which side of its tile a point sits on.
enum class Side { West, East, South, North };

struct Ring {
  double left, right, bottom, top;

  Point corner(Side vertical, Side horizontal) const {
    return {horizontal == Side::West ? left : right,
            vertical == Side::South ? bottom : top};
  }
};

Side sideOf(const LayoutGeometry &layout, int col, int row, Point p) {
  double dx = p.first - layout.tileCenterX(col);
  double dy = p.second - layout.tileCenterY(row);
  double halfWidth = std::max(layout.tileWidth / 2, 1e-6);
  double halfHeight = std::max(layout.tileHeight(col, row) / 2, 1e-6);
  if (std::abs(dx) / halfWidth >= std::abs(dy) / halfHeight)
    return dx >= 0 ? Side::East : Side::West;
  return dy >= 0 ? Side::North : Side::South;
}

bool isVertical(Side side) { return side == Side::West || side == Side::East; }

/// Appends a quarter arc turning from the segment ending at `corner` onto the
/// one leaving it, rounded by `radius`, as three Bezier control points.
void appendCorner(SmallVectorImpl<Point> &points, Point from, Point corner,
                  Point to, double radius) {
  auto unit = [](Point a, Point b) -> Point {
    double dx = b.first - a.first, dy = b.second - a.second;
    double len = std::hypot(dx, dy);
    return len < 1e-9 ? Point{0, 0} : Point{dx / len, dy / len};
  };
  Point in = unit(from, corner);
  Point out = unit(corner, to);
  double reach = std::min(
      {radius,
       std::hypot(corner.first - from.first, corner.second - from.second) / 2,
       std::hypot(to.first - corner.first, to.second - corner.second) / 2});
  Point start{corner.first - in.first * reach,
              corner.second - in.second * reach};
  Point end{corner.first + out.first * reach,
            corner.second + out.second * reach};
  // Straight run up to where the turn starts, then the arc itself.
  points.push_back(start);
  points.push_back(start);
  points.push_back({start.first + in.first * reach * kArcControl,
                    start.second + in.second * reach * kArcControl});
  points.push_back({end.first - out.first * reach * kArcControl,
                    end.second - out.second * reach * kArcControl});
  points.push_back(end);
}

/// Turns a corner list into the Bezier chain graphviz's edge `pos` wants:
/// 3n+1 control points, with straight runs repeating their endpoints.
SmallVector<Point> toBezier(ArrayRef<Point> corners, double radius) {
  SmallVector<Point> out;
  if (corners.size() < 2)
    return out;
  out.push_back(corners.front());
  for (size_t i = 1; i + 1 < corners.size(); ++i) {
    SmallVector<Point> arc;
    appendCorner(arc, corners[i - 1], corners[i], corners[i + 1], radius);
    // appendCorner emits [start, start, c1, c2, end]; the first pair closes
    // the straight run that led here.
    out.push_back(out.back());
    out.push_back(arc[0]);
    out.push_back(arc[0]);
    out.push_back(arc[2]);
    out.push_back(arc[3]);
    out.push_back(arc[4]);
  }
  Point last = corners.back();
  out.push_back(out.back());
  out.push_back(last);
  out.push_back(last);
  return out;
}

/// Where an arrow meets a ring: a point on one side of one tile, on a track
/// that separates it from the other arrows using that side.
struct Anchor {
  int col = 0, row = 0;
  Side side = Side::West;
  Point at{0, 0};
  unsigned track = 0;
  bool valid = false;
};

/// A stretch of space arrows share: one side of one tile's ring. Arrows
/// running along the same stretch need separate tracks.
struct Lane {
  int col, row;
  Side side;
  bool operator<(const Lane &o) const {
    return std::tie(col, row, side) <
           std::tie(o.col, o.row, static_cast<const Side &>(o.side));
  }
};

/// Both ends of every arrow, each on its own track.
struct Plan {
  SmallVector<std::array<Anchor, 2>> ends;
};

Plan planAnchors(ArrayRef<ArrowRequest> requests,
                 const LayoutGeometry &layout) {
  Plan plan;
  plan.ends.resize(requests.size());
  std::map<Lane, SmallVector<std::pair<unsigned, unsigned>>> laneUsers;
  for (const ArrowRequest &request : requests) {
    if (request.tiles.size() < 2 || !request.pinnedSource ||
        !request.pinnedDest)
      continue;
    auto [sourceCol, sourceRow] = request.tiles.front();
    auto [destCol, destRow] = request.tiles.back();
    std::array<Anchor, 2> &ends = plan.ends[request.id];
    ends[0] = {sourceCol,
               sourceRow,
               sideOf(layout, sourceCol, sourceRow, *request.pinnedSource),
               *request.pinnedSource,
               0,
               true};
    ends[1] = {destCol,
               destRow,
               sideOf(layout, destCol, destRow, *request.pinnedDest),
               *request.pinnedDest,
               0,
               true};
    for (unsigned end = 0; end < 2; ++end)
      laneUsers[{ends[end].col, ends[end].row, ends[end].side}].push_back(
          {request.id, end});
  }

  // Order a lane's arrows by where their other end lies, so that a bundle
  // heading the same way keeps its order along the whole run instead of
  // swapping places and crossing.
  for (auto &[lane, users] : laneUsers) {
    llvm::stable_sort(users, [&](auto a, auto b) {
      auto reach = [&](std::pair<unsigned, unsigned> use) {
        const Anchor &other = plan.ends[use.first][1 - use.second];
        return isVertical(lane.side) ? other.at.second : other.at.first;
      };
      return reach(a) < reach(b);
    });
    for (auto [index, use] : llvm::enumerate(users))
      plan.ends[use.first][use.second].track = index;
  }
  return plan;
}

class ChannelArrowRenderer : public ArrowRenderer {
public:
  std::optional<double> preferredTileGap() const override {
    // Room for a ring either side of a channel and the tracks between them.
    return 2 * kRingInset + 12 * kTrackPitch;
  }

  std::map<std::string, Point>
  portPositions(ArrayRef<ArrowRequest> requests,
                const LayoutGeometry &layout) override {
    // Ports move out of the tile onto its ring, so that an arrow reaches one
    // without crossing the tile; what stays inside is the stub from the port
    // to the buffer or core it serves.
    Plan plan = planAnchors(requests, layout);
    std::map<std::string, Point> placed;
    for (const ArrowRequest &request : requests) {
      const std::array<Anchor, 2> &ends = plan.ends[request.id];
      if (!ends[0].valid)
        continue;
      for (auto [end, node] : {std::pair{0u, std::cref(request.sourceNode)},
                               std::pair{1u, std::cref(request.destNode)}}) {
        if (node.get().empty())
          continue;
        placed[node.get()] = onRing(
            ringOf(layout, ends[end].col, ends[end].row, ends[end].track),
            ends[end].side, ends[end].at);
      }
    }
    return placed;
  }

  SmallVector<ArrowPath> route(ArrayRef<ArrowRequest> requests,
                               const LayoutGeometry &layout) override {
    Plan plan = planAnchors(requests, layout);
    SmallVector<ArrowPath> paths;
    for (const ArrowRequest &request : requests) {
      const std::array<Anchor, 2> &ends = plan.ends[request.id];
      if (!ends[0].valid)
        continue;
      const Anchor &from = ends[0];
      const Anchor &to = ends[1];
      Ring fromRing = ringOf(layout, from.col, from.row, from.track);
      Ring toRing = ringOf(layout, to.col, to.row, to.track);

      SmallVector<Point> corners;
      Point leave = onRing(fromRing, from.side, from.at);
      corners.push_back(leave);
      if (from.col == to.col && from.row == to.row) {
        // A switchbox turning a flow around inside one tile: follow the ring
        // around rather than cutting back across the tile.
        if (isVertical(from.side) != isVertical(to.side)) {
          Side vertical = (from.side == Side::North || to.side == Side::North)
                              ? Side::North
                              : Side::South;
          Side horizontal = (from.side == Side::West || to.side == Side::West)
                                ? Side::West
                                : Side::East;
          corners.push_back(fromRing.corner(vertical, horizontal));
        }
        corners.push_back(onRing(fromRing, to.side, to.at));
      } else {
        Point enter = onRing(toRing, to.side, to.at);
        // Cross the channel halfway between the rings, turning once on each
        // side so the crossing is square to both tiles.
        if (isVertical(from.side)) {
          double mid = (leave.first + enter.first) / 2;
          corners.push_back({mid, leave.second});
          corners.push_back({mid, enter.second});
        } else {
          double mid = (leave.second + enter.second) / 2;
          corners.push_back({leave.first, mid});
          corners.push_back({enter.first, mid});
        }
        corners.push_back(enter);
      }

      SmallVector<Point> tidy;
      for (Point p : corners)
        if (tidy.empty() || std::hypot(p.first - tidy.back().first,
                                       p.second - tidy.back().second) > 1e-6)
          tidy.push_back(p);
      SmallVector<Point> bezier = toBezier(tidy, kCornerRadius);
      if (!bezier.empty())
        paths.push_back({request.id, std::move(bezier)});
    }
    return paths;
  }

private:
  static Ring ringOf(const LayoutGeometry &layout, int col, int row,
                     unsigned track) {
    double pad = kRingInset + track * kTrackPitch;
    return {layout.tileLeft(col) - pad, layout.tileRight(col) + pad,
            layout.tileBottom(col, row) - pad, layout.tileTop(col, row) + pad};
  }

  static Point onRing(const Ring &ring, Side side, Point along) {
    switch (side) {
    case Side::West:
      return {ring.left, along.second};
    case Side::East:
      return {ring.right, along.second};
    case Side::South:
      return {along.first, ring.bottom};
    case Side::North:
      return {along.first, ring.top};
    }
    return along;
  }
};

} // namespace

std::unique_ptr<ArrowRenderer> createChannelArrowRenderer() {
  return std::make_unique<ChannelArrowRenderer>();
}

} // namespace xilinx::AIE
