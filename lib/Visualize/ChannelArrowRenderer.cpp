//===- ChannelArrowRenderer.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Draws every arrow in the gaps between tiles.
//
// A tile's switchbox ports gather in a column at the channel junction off its
// south-west corner. All four channels meet there, so a hop to a neighbouring
// switchbox is a single run along one channel, and a hop the switchbox makes
// within itself stays in that column instead of crossing the tile. DMA
// channels stack down the tile's west edge, reached from the junction up the
// west channel, and leave a short stub east into the buffers they serve.
//
// Arrows sharing a channel are spread onto tracks and ordered to cross less.
//
//===----------------------------------------------------------------------===//

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

using namespace llvm;

namespace xilinx::AIE {
namespace {

/// A circle's quarter arc is within half a percent of a cubic Bezier whose
/// control points lie this far along the tangents.
constexpr double kArcControl = 0.5523;

constexpr double kCornerRadius = 0.06;
/// Space between parallel runs sharing a channel, and between stacked ports.
constexpr double kTrackPitch = 0.07;
constexpr double kPortPitch = 0.13;
/// How far a DMA port sits off the tile's west edge.
constexpr double kDMAOffset = 0.30;
/// Tracks past this share a line rather than widening the channel for ever.
constexpr unsigned kMaxTracks = 10;

/// Appends the Bezier triple for a straight run ending at `to`.
void lineTo(SmallVectorImpl<Point> &out, Point to) {
  Point from = out.back();
  out.push_back({from.first + (to.first - from.first) / 3,
                 from.second + (to.second - from.second) / 3});
  out.push_back({from.first + 2 * (to.first - from.first) / 3,
                 from.second + 2 * (to.second - from.second) / 3});
  out.push_back(to);
}

Point unitBetween(Point a, Point b) {
  double dx = b.first - a.first, dy = b.second - a.second;
  double len = std::hypot(dx, dy);
  return len < 1e-9 ? Point{0, 0} : Point{dx / len, dy / len};
}

/// Turns a corner list into the Bezier chain graphviz's edge `pos` wants,
/// rounding each corner into a quarter arc.
SmallVector<Point> toBezier(ArrayRef<Point> corners, double radius) {
  SmallVector<Point> out;
  if (corners.size() < 2)
    return out;
  out.push_back(corners.front());
  for (size_t i = 1; i + 1 < corners.size(); ++i) {
    Point in = unitBetween(corners[i - 1], corners[i]);
    Point away = unitBetween(corners[i], corners[i + 1]);
    double reach =
        std::min({radius,
                  std::hypot(corners[i].first - corners[i - 1].first,
                             corners[i].second - corners[i - 1].second) /
                      2,
                  std::hypot(corners[i + 1].first - corners[i].first,
                             corners[i + 1].second - corners[i].second) /
                      2});
    Point enter{corners[i].first - in.first * reach,
                corners[i].second - in.second * reach};
    Point leave{corners[i].first + away.first * reach,
                corners[i].second + away.second * reach};
    lineTo(out, enter);
    out.push_back({enter.first + in.first * reach * kArcControl,
                   enter.second + in.second * reach * kArcControl});
    out.push_back({leave.first - away.first * reach * kArcControl,
                   leave.second - away.second * reach * kArcControl});
    out.push_back(leave);
  }
  lineTo(out, corners.back());
  return out;
}

/// Where a tile's switchbox ports gather: the junction off its south-west
/// corner, where the channel west of the tile meets the channel below it.
Point junction(const LayoutGeometry &layout, int col, int row) {
  return {layout.tileLeft(col) - layout.tileGapX / 2,
          layout.tileBottom(col, row) - layout.tileGapY / 2};
}

/// A stretch arrows share, so the ones running together can be spread apart.
/// A vertical lane is the channel west of a column; a horizontal one is the
/// channel below a row.
struct Lane {
  bool vertical;
  int index;
  bool operator<(const Lane &o) const {
    return std::tie(vertical, index) < std::tie(o.vertical, o.index);
  }
};

class ChannelArrowRenderer : public ArrowRenderer {
public:
  std::optional<double> preferredTileGap() const override {
    // Room for a column of switchbox ports, the DMA stubs beside it, and the
    // tracks that run between them.
    return 1.3;
  }

  std::map<std::string, Point>
  nodePositions(ArrayRef<ArrowRequest> requests,
                const LayoutGeometry &layout) override {
    std::map<std::string, Point> placed;
    for (const ArrowRequest &request : requests)
      for (const ArrowEnd *end : {&request.from, &request.to})
        place(*end, layout, placed);
    return placed;
  }

  SmallVector<ArrowPath> route(ArrayRef<ArrowRequest> requests,
                               const LayoutGeometry &layout) override;

private:
  /// Switchbox ports sit in a column at the junction; DMA channels stack down
  /// the tile's west edge. Both read in slot order, rather than wherever the
  /// bundle each belongs to happened to face.
  static void place(const ArrowEnd &end, const LayoutGeometry &layout,
                    std::map<std::string, Point> &placed) {
    if (end.node.empty() || placed.count(end.node))
      return;
    auto [col, row] = end.tile;
    if (end.kind == EndKind::Switch) {
      Point at = junction(layout, col, row);
      placed[end.node] = {at.first, at.second + end.slot * kPortPitch};
    } else if (end.kind == EndKind::DMA) {
      placed[end.node] = {layout.tileLeft(col) - kDMAOffset,
                          layout.tileTop(col, row) - 0.55 -
                              end.slot * kPortPitch};
    }
  }

  static Point positionOf(const ArrowEnd &end, const LayoutGeometry &layout) {
    std::map<std::string, Point> one;
    place(end, layout, one);
    auto found = one.find(end.node);
    return found == one.end() ? end.pinned : found->second;
  }
};

SmallVector<ArrowPath>
ChannelArrowRenderer::route(ArrayRef<ArrowRequest> requests,
                            const LayoutGeometry &layout) {
  // Spread arrows sharing a channel onto tracks, ordered by where they are
  // heading so a bundle travelling together keeps its order for the whole run.
  std::map<Lane, SmallVector<unsigned>> laneUsers;
  auto laneOf = [&](const ArrowRequest &r) -> std::optional<Lane> {
    if (r.from.tile == r.to.tile)
      return std::nullopt;
    if (r.from.tile.second != r.to.tile.second)
      return Lane{true, std::min(r.from.tile.first, r.to.tile.first)};
    return Lane{false, std::min(r.from.tile.second, r.to.tile.second)};
  };
  for (const ArrowRequest &request : requests)
    if (laneOf(request))
      laneUsers[*laneOf(request)].push_back(request.id);

  DenseMap<unsigned, unsigned> trackOf;
  for (auto &[lane, users] : laneUsers) {
    llvm::stable_sort(users, [&](unsigned a, unsigned b) {
      Point reachA = positionOf(requests[a].to, layout);
      Point reachB = positionOf(requests[b].to, layout);
      return lane.vertical ? reachA.second < reachB.second
                           : reachA.first < reachB.first;
    });
    for (auto [index, id] : llvm::enumerate(users))
      trackOf[id] = std::min<unsigned>(index, kMaxTracks);
  }

  SmallVector<ArrowPath> paths;
  for (const ArrowRequest &request : requests) {
    Point from = positionOf(request.from, layout);
    Point to = positionOf(request.to, layout);
    double track = trackOf.lookup(request.id) * kTrackPitch;
    auto [col, row] = request.from.tile;
    bool sameTile = request.from.tile == request.to.tile;

    SmallVector<Point> corners{from};
    if (sameTile && request.from.kind == EndKind::Switch &&
        request.to.kind == EndKind::Switch) {
      // Both ends are in the same column of ports: a straight run that never
      // approaches the tile.
    } else if (sameTile &&
               (request.from.kind == EndKind::DMA ||
                request.to.kind == EndKind::DMA) &&
               (request.from.kind == EndKind::Switch ||
                request.to.kind == EndKind::Switch)) {
      // Switchbox to its own DMA: along the west channel, then in to the port.
      double lane = junction(layout, col, row).first;
      corners.push_back({lane, from.second});
      corners.push_back({lane, to.second});
    } else if (request.from.kind == EndKind::External ||
               request.to.kind == EndKind::External) {
      // Something outside the array reaching a shim DMA: up to the channel
      // below the tile's row, along to its column, then up to the port.
      const ArrowEnd &inside =
          request.from.kind == EndKind::External ? request.to : request.from;
      Point at = junction(layout, inside.tile.first, inside.tile.second);
      Point outer = request.from.kind == EndKind::External ? from : to;
      Point port = request.from.kind == EndKind::External ? to : from;
      SmallVector<Point> via{{outer.first, at.second - track},
                             {at.first - track, at.second - track},
                             {at.first - track, port.second}};
      if (request.from.kind != EndKind::External)
        std::reverse(via.begin(), via.end());
      corners.append(via.begin(), via.end());
    } else if (sameTile && request.from.kind == EndKind::Content &&
               request.to.kind == EndKind::Content) {
      // Two things stacked inside one tile: out to a lane beside the stack and
      // back, rather than straight through whatever sits between them.
      double lane = layout.tileRight(col) - 0.22 - track;
      corners.push_back({lane, from.second});
      corners.push_back({lane, to.second});
    } else if (sameTile) {
      // A DMA channel and the buffer it serves: straight in from the edge.
      corners.push_back({from.first, to.second});
    } else {
      // Neighbouring switchboxes share a channel, so one turn reaches it.
      bool vertical = request.from.tile.second != request.to.tile.second;
      if (vertical) {
        double lane = from.first - track;
        corners.push_back({lane, from.second});
        corners.push_back({lane, to.second});
      } else {
        double lane = from.second - track;
        corners.push_back({from.first, lane});
        corners.push_back({to.first, lane});
      }
    }
    corners.push_back(to);

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

} // namespace

std::unique_ptr<ArrowRenderer> createChannelArrowRenderer() {
  return std::make_unique<ChannelArrowRenderer>();
}

} // namespace xilinx::AIE
