//===- ChannelArrowRenderer.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Draws arrows so that no two of them merge.
//
// Two runs that are parallel and overlap read as one line: a reader cannot see
// where one ends and the other begins, and the flows appear to join. A
// perpendicular crossing stays readable. So the only thing this renderer has
// to guarantee is that parallel runs never share a line.
//
// It gets that from two rules rather than from tuning.
//
// The free space between tiles is a grid of corridors: one vertical channel
// per column, left of that column's switchboxes and clear for the whole
// height of the array, one horizontal channel per row, and short strips
// inside each tile and across each switchbox. Every arrow is a chain of
// straight legs, each lying in one corridor, with a stub at either end
// joining the chain to the box it meets. An arrow between two tiles in
// different rows and columns runs down the source column's channel, across a
// row channel, and up the destination column's channel. One between tiles in
// the same row or the same column needs a single leg.
//
// Within a corridor a leg is given a track, and tracks are handed out by
// interval coloring over the stretch each leg occupies: two legs share a
// track only when they do not overlap along it. The corridor is as wide as
// its busiest point needs and no wider. Corridors do not overlap, so legs in
// different corridors cross but never merge.
//
// Where an arrow is free to choose a row channel to cross in, it takes the
// one carrying least across the columns it spans.
//
//===----------------------------------------------------------------------===//

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

using namespace llvm;

namespace xilinx::AIE {
namespace {

/// A circle's quarter arc is within half a percent of a cubic Bezier whose
/// control points lie this far along the tangents.
constexpr double kArcControl = 0.5523;
constexpr double kCornerRadius = 0.05;

/// Spacing between tracks, and between the ports stacked on a switchbox.
constexpr double kTrack = 0.1;
constexpr double kPortPitch = 0.17;
/// How far each port steps out from its side, so no two share a line.
constexpr double kStagger = 0.12;
constexpr double kDMAPitch = 0.23;

/// The channel a column reserves, and how it is divided. Measured leftwards
/// from the tile's edge, so each piece has a place that cannot drift.
constexpr double kChannel = 3.4;
constexpr double kDMAHalfWidth = 0.42;
constexpr double kDMAInset = 0.06;
constexpr double kSwitchInset = 1.45;
constexpr double kSwitchHalfWidth = 0.30;
constexpr double kEdgeMargin = 0.09;
/// However busy a tile is, it stops widening here.
constexpr double kWidestTile = 7.5;

double dmaCenterX(const LayoutGeometry &layout, int col) {
  return layout.tileLeft(col) - kDMAInset - layout.text(kDMAHalfWidth);
}
double switchCenterX(const LayoutGeometry &layout, int col) {
  return layout.tileLeft(col) - layout.tileGapX + kSwitchInset;
}

/// How tall a switchbox is. It has to hold a track for every flow passing
/// through it, so it grows with what it carries rather than being guessed.
struct SwitchSize {
  double halfHeight = 0.2;
};

/// Which strip of empty space a run lies in. Corridors never overlap, so two
/// runs in different corridors can cross but never merge.
enum class CorridorKind {
  /// Across a switchbox, between the two sides a flow uses.
  Switchbox,
  /// Beside a row, where a flow steps between two switchboxes above and below.
  ColumnGap,
  /// Below a row, where a flow steps between two switchboxes left and right.
  RowGap,
  /// Between a switchbox and the DMA channels of its own tile.
  DMAApproach,
  /// Inside a tile, between its edge and its contents.
  TileIn,
  /// Inside a tile, beyond its contents.
  TileOut,
  /// Below the array, where the runtime arguments sit.
  Arguments,
};

struct CorridorKey {
  CorridorKind kind;
  int a = 0, b = 0;
  bool operator<(const CorridorKey &o) const {
    return std::tie(kind, a, b) < std::tie(o.kind, o.a, o.b);
  }
};

/// Where a corridor's tracks start and which way they step.
struct Corridor {
  double base = 0;
  double span = 0;
};

/// One straight piece of an arrow, before it knows which track it is on.
struct Leg {
  CorridorKey corridor;
  bool vertical = false;
  double lo = 0, hi = 0;
  double cross = 0;
};

/// One arrow's chain of legs. Consecutive legs are perpendicular, and a stub
/// at each end joins the chain to the box it meets.
struct Path {
  unsigned id = 0;
  Point from{0, 0}, to{0, 0};
  SmallVector<Leg, 4> legs;
  bool valid = false;
};

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

/// Pulls an arrow's end back from a node's middle to the edge it arrives at.
Point onEdge(const ArrowEnd &end, Point center, Point toward) {
  double dx = toward.first - center.first;
  double dy = toward.second - center.second;
  if (std::abs(dx) < 1e-9 && std::abs(dy) < 1e-9)
    return center;
  double infinity = std::numeric_limits<double>::infinity();
  double scaleX = std::abs(dx) < 1e-9 ? infinity : end.halfWidth / std::abs(dx);
  double scaleY =
      std::abs(dy) < 1e-9 ? infinity : end.halfHeight / std::abs(dy);
  return {center.first + dx * std::min({scaleX, scaleY, 1.0}),
          center.second + dy * std::min({scaleX, scaleY, 1.0})};
}

class ChannelArrowRenderer : public ArrowRenderer {
public:
  std::optional<double> preferredTileGap(double textRoom) const override {
    // The channel holds the DMA boxes, the switchbox and a corridor either
    // side of it. Only the boxes have to keep their drawn size, so the
    // channel grows by what they gain.
    return kChannel + 2 * kDMAHalfWidth * (textRoom - 1);
  }

  std::optional<double> widestLine() const override {
    // Leave better than half the gap between neighboring tracks clear, or
    // two lines running side by side read as one thick one.
    return kTrack * 72.0 * 0.42;
  }

  std::optional<double> preferredTileWidth(unsigned contents,
                                           double contentWidth,
                                           double textRoom) const override {
    // Contents down the middle, and a lane either side wide enough that the
    // arrows between them each get a track of their own. A tile with more
    // arrows than that stops growing and packs them closer instead: past a
    // point a wider tile costs the whole picture more than it buys this one.
    double wanted = contentWidth + 2 * (contents * kTrack + 2 * kEdgeMargin);
    return std::min(wanted, kWidestTile * textRoom);
  }

  std::map<std::string, Point>
  nodePositions(ArrayRef<ArrowRequest> requests,
                const LayoutGeometry &layout) override {
    measure(requests);
    std::map<std::string, Point> placed;
    for (const ArrowRequest &request : requests)
      for (const ArrowEnd *end : {&request.from, &request.to})
        place(*end, layout, placed);
    return placed;
  }

  SmallVector<ArrowPath> route(ArrayRef<ArrowRequest> requests,
                               const LayoutGeometry &layout) override;

private:
  /// A switchbox is as tall as the flows crossing it need, and as wide as the
  /// ports on its busiest side.
  void measure(ArrayRef<ArrowRequest> requests) {
    std::map<std::pair<int, int>, int> crossings;
    for (const ArrowRequest &request : requests) {
      if (request.from.kind == EndKind::Switch &&
          request.to.kind == EndKind::Switch &&
          request.from.tile == request.to.tile)
        ++crossings[request.from.tile];
      for (const ArrowEnd *end : {&request.from, &request.to})
        if (end->kind == EndKind::Switch)
          sides[{end->tile, end->facing}] =
              std::max(sides[{end->tile, end->facing}], end->slot + 1);
    }
    for (auto [tile, count] : crossings)
      switches[tile].halfHeight =
          std::max(0.2, (count + 1) * kTrack / 2 + kEdgeMargin);
    for (auto &[key, count] : sides)
      switches[key.first].halfHeight =
          std::max(switches[key.first].halfHeight, 0.2);
    rowSwitchHalf.clear();
    for (auto &[tile, size] : switches)
      rowSwitchHalf[tile.second] =
          std::max(rowSwitchHalf[tile.second], size.halfHeight);
  }

  void place(const ArrowEnd &end, const LayoutGeometry &layout,
             std::map<std::string, Point> &placed) const {
    if (end.node.empty() || placed.count(end.node))
      return;
    auto [col, row] = end.tile;
    if (end.kind == EndKind::Switch) {
      double cx = switchCenterX(layout, col);
      double cy = layout.channelY(row);
      auto found = switches.find(end.tile);
      double half = found == switches.end() ? 0.2 : found->second.halfHeight;
      // A port sits on the side it faces, so a flow passing straight through
      // never meets the ports facing the other way.
      auto spread = [&](Facing facing) {
        auto count = sides.find({end.tile, facing});
        int n = count == sides.end() ? 1 : count->second;
        return (end.slot - (n - 1) / 2.0) * layout.text(kPortPitch);
      };
      // Ports step outwards as well as along their side. Sharing a side
      // would otherwise mean sharing the line an arrow leaves it on, which
      // is the one thing a reader cannot see through.
      double out = end.slot * kStagger;
      switch (end.facing) {
      case Facing::North:
        placed[end.node] = {cx + spread(Facing::North), cy + half + out};
        break;
      case Facing::South:
        placed[end.node] = {cx + spread(Facing::South), cy - half - out};
        break;
      case Facing::East:
        placed[end.node] = {cx + kSwitchHalfWidth + out,
                            cy + spread(Facing::East)};
        break;
      case Facing::West:
        placed[end.node] = {cx - kSwitchHalfWidth - out,
                            cy + spread(Facing::West)};
        break;
      case Facing::Local:
        placed[end.node] = {cx + kSwitchHalfWidth + out,
                            cy + spread(Facing::Local)};
        break;
      }
    } else if (end.kind == EndKind::DMA) {
      placed[end.node] = {dmaCenterX(layout, col),
                          layout.tileTop(col, row) - layout.text(0.5) -
                              end.slot * layout.text(kDMAPitch)};
    }
  }

  Point positionOf(const ArrowEnd &end, const LayoutGeometry &layout) const {
    std::map<std::string, Point> one;
    place(end, layout, one);
    auto found = one.find(end.node);
    return found == one.end() ? end.pinned : found->second;
  }

  /// The chain of corridors an arrow follows. `rowLoad` records the column
  /// ranges already taken in each row channel, so that a path free to cross
  /// in more than one row takes the emptiest.
  Path
  plan(const ArrowRequest &request, const LayoutGeometry &layout,
       std::map<int, SmallVector<std::pair<double, double>>> &rowLoad) const;
  Corridor corridorOf(const CorridorKey &key,
                      const LayoutGeometry &layout) const;
  double corridorMiddle(const CorridorKey &key,
                        const LayoutGeometry &layout) const {
    Corridor corridor = corridorOf(key, layout);
    return corridor.base + corridor.span / 2;
  }

  std::map<std::pair<int, int>, SwitchSize> switches;
  /// The tallest switchbox in each row, which bounds the row's channel.
  std::map<int, double> rowSwitchHalf;
  std::map<std::pair<std::pair<int, int>, Facing>, int> sides;
  /// How wide a tile's contents are, which bounds the corridors either side
  /// of them, and where the runtime arguments start.
  double contentHalfWidth = 0;
  double argumentsTop = 0;
};

/// Which strip inside a tile an end leaves through, so that an arrow reaches
/// the column channel without crossing the tile's contents.
Path ChannelArrowRenderer::plan(
    const ArrowRequest &request, const LayoutGeometry &layout,
    std::map<int, SmallVector<std::pair<double, double>>> &rowLoad) const {
  Path path;
  path.id = request.id;
  path.from = positionOf(request.from, layout);
  path.to = positionOf(request.to, layout);
  path.valid = true;
  auto [col, row] = request.from.tile;
  auto [destCol, destRow] = request.to.tile;
  bool sameTile = request.from.tile == request.to.tile;
  EndKind a = request.from.kind, b = request.to.kind;

  auto add = [&](CorridorKind kind, int which, bool vertical) {
    path.legs.push_back({{kind, which, 0}, vertical, 0, 0, 0});
  };

  if (sameTile && a != EndKind::External && b != EndKind::External) {
    if (a == EndKind::Switch && b == EndKind::Switch)
      path.legs.push_back(
          {{CorridorKind::Switchbox, col, row}, false, 0, 0, 0});
    else if (a == EndKind::DMA || b == EndKind::DMA)
      add(a == EndKind::Switch || b == EndKind::Switch
              ? CorridorKind::DMAApproach
              : CorridorKind::TileIn,
          col, /*vertical=*/true);
    else
      add(CorridorKind::TileOut, col, /*vertical=*/true);
    return path;
  }

  // An end outside the array is reached along the strip below it, and from
  // there every column channel is open.
  if (a == EndKind::External || b == EndKind::External) {
    int inside = a == EndKind::External ? destCol : col;
    add(CorridorKind::Arguments, 0, /*vertical=*/false);
    add(CorridorKind::ColumnGap, inside, /*vertical=*/true);
    if (a == EndKind::External)
      return path;
    std::reverse(path.legs.begin(), path.legs.end());
    return path;
  }

  if (col == destCol) {
    add(CorridorKind::ColumnGap, col, /*vertical=*/true);
    return path;
  }
  if (row == destRow) {
    add(CorridorKind::RowGap, row, /*vertical=*/false);
    return path;
  }

  // Down one column channel, across a row channel, up the other. Any row
  // between the two ends will do, so take the one carrying least across the
  // columns this path spans.
  double left =
      corridorMiddle({CorridorKind::ColumnGap, std::min(col, destCol)}, layout);
  double right =
      corridorMiddle({CorridorKind::ColumnGap, std::max(col, destCol)}, layout);
  int bestRow = destRow;
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  for (int candidate = std::min(row, destRow);
       candidate <= std::max(row, destRow); ++candidate) {
    unsigned crowding = 0;
    for (auto [lo, hi] : rowLoad[candidate])
      if (lo < right && hi > left)
        ++crowding;
    unsigned detour = std::abs(candidate - destRow) + std::abs(candidate - row);
    unsigned cost = crowding * 4 + detour;
    if (cost < bestCost) {
      bestCost = cost;
      bestRow = candidate;
    }
  }
  rowLoad[bestRow].push_back({left, right});

  add(CorridorKind::ColumnGap, col, /*vertical=*/true);
  add(CorridorKind::RowGap, bestRow, /*vertical=*/false);
  add(CorridorKind::ColumnGap, destCol, /*vertical=*/true);
  return path;
}

Corridor ChannelArrowRenderer::corridorOf(const CorridorKey &key,
                                          const LayoutGeometry &layout) const {
  // Each kind names its own key fields; they are not all (column, row).
  Corridor corridor;
  switch (key.kind) {
  case CorridorKind::Switchbox: {
    int row = key.b;
    auto found = switches.find({key.a, key.b});
    double half = found == switches.end() ? 0.2 : found->second.halfHeight;
    corridor.base = layout.channelY(row) - half + kEdgeMargin;
    corridor.span = 2 * half - 2 * kEdgeMargin;
    break;
  }
  case CorridorKind::ColumnGap: {
    // The strip left of the switchboxes, which is clear the whole height of
    // the array, so a run passing a row has somewhere of its own to be.
    int col = key.a;
    double hi = switchCenterX(layout, col) - kSwitchHalfWidth - kEdgeMargin;
    double lo = layout.tileLeft(col) - layout.tileGapX + kEdgeMargin;
    corridor.base = hi;
    corridor.span = lo - hi;
    break;
  }
  case CorridorKind::RowGap: {
    int row = key.a;
    auto found = rowSwitchHalf.find(row);
    double half = found == rowSwitchHalf.end() ? 0.2 : found->second;
    double hi = layout.channelY(row) - half - kEdgeMargin;
    // The channel below the bottom row is open, so it has no far edge.
    double lo = row > 0 ? layout.rowTop(row - 1) + kEdgeMargin
                        : hi - layout.tileGapY / 2;
    corridor.base = hi;
    corridor.span = lo - hi;
    break;
  }
  case CorridorKind::DMAApproach: {
    int col = key.a;
    double lo = switchCenterX(layout, col) + kSwitchHalfWidth + kEdgeMargin;
    double hi =
        dmaCenterX(layout, col) - layout.text(kDMAHalfWidth) - kEdgeMargin;
    corridor.base = lo;
    corridor.span = hi - lo;
    break;
  }
  case CorridorKind::TileIn: {
    int col = key.a;
    double lo = layout.tileLeft(col) + kEdgeMargin;
    double hi = layout.tileCenterX(col) - contentHalfWidth - kEdgeMargin;
    corridor.base = lo;
    corridor.span = hi - lo;
    break;
  }
  case CorridorKind::TileOut: {
    int col = key.a;
    double lo = layout.tileCenterX(col) + contentHalfWidth + kEdgeMargin;
    double hi = layout.tileRight(col) - kEdgeMargin;
    corridor.base = lo;
    corridor.span = hi - lo;
    break;
  }
  case CorridorKind::Arguments: {
    // Above the arguments, never through them: they are the lowest thing
    // drawn, and the only way out of one is a short run up out of its top.
    corridor.base = argumentsTop + kEdgeMargin;
    corridor.span = layout.tileGapY / 2;
    break;
  }
  }
  return corridor;
}

SmallVector<ArrowPath>
ChannelArrowRenderer::route(ArrayRef<ArrowRequest> requests,
                            const LayoutGeometry &layout) {
  measure(requests);

  // Where a tile's contents sit, which bounds the two corridors inside it.
  contentHalfWidth = 0;
  argumentsTop = -std::numeric_limits<double>::infinity();
  for (const ArrowRequest &request : requests)
    for (const ArrowEnd *end : {&request.from, &request.to}) {
      if (end->kind == EndKind::Content)
        contentHalfWidth = std::max(contentHalfWidth, end->halfWidth);
      if (end->kind == EndKind::External)
        argumentsTop =
            std::max(argumentsTop, end->pinned.second + end->halfHeight);
    }

  std::map<int, SmallVector<std::pair<double, double>>> rowLoad;
  SmallVector<Path> plans;
  plans.reserve(requests.size());
  for (const ArrowRequest &request : requests)
    plans.push_back(plan(request, layout, rowLoad));

  // Arrows meeting the same box would otherwise all run along its center
  // line, which is the one place they cannot help but merge. Each takes its
  // own point on the edge it arrives at instead, so they stay apart until
  // they touch the box.
  auto meetsSideways = [&](const Path &path, bool isFrom) {
    if (path.legs.empty())
      return true;
    return isFrom ? path.legs.front().vertical : path.legs.back().vertical;
  };
  std::map<std::pair<std::string, bool>, SmallVector<std::pair<unsigned, bool>>>
      arrivals;
  for (const ArrowRequest &request : requests) {
    const Path &path = plans[request.id];
    if (!path.valid)
      continue;
    // An arrow leaving a box sideways may take any point up or down its edge;
    // one leaving it above or below may take any point along the top or
    // bottom.
    for (auto [end, isFrom] :
         {std::pair{&request.from, true}, std::pair{&request.to, false}})
      arrivals[{end->node, meetsSideways(path, isFrom)}].push_back(
          {request.id, isFrom});
  }
  DenseMap<uint64_t, double> offsetOf;
  for (auto &[key, members] : arrivals) {
    size_t count = members.size();
    if (count < 2)
      continue;
    for (auto [index, member] : llvm::enumerate(members)) {
      auto [id, isFrom] = member;
      const ArrowEnd &end = isFrom ? requests[id].from : requests[id].to;
      double reach = key.second ? end.halfHeight : end.halfWidth;
      double spread = 2 * reach * 0.72;
      double offset =
          count < 2 ? 0 : spread * (double(index) / (count - 1) - 0.5);
      offsetOf[(uint64_t(id) << 1) | unsigned(isFrom)] = offset;
    }
  }
  auto attach = [&](unsigned id, bool isFrom, Point at, bool sideways) {
    auto found = offsetOf.find((uint64_t(id) << 1) | unsigned(isFrom));
    if (found == offsetOf.end())
      return at;
    return sideways ? Point{at.first, at.second + found->second}
                    : Point{at.first + found->second, at.second};
  };

  // How far each leg reaches, measured between the middles of the corridors
  // on either side of it. A track sits within a corridor, so this is within
  // half a corridor's width of where the leg ends up.
  for (Path &path : plans) {
    if (!path.valid)
      continue;
    path.from = attach(path.id, true, path.from, meetsSideways(path, true));
    path.to = attach(path.id, false, path.to, meetsSideways(path, false));
    for (auto [index, leg] : llvm::enumerate(path.legs)) {
      double before =
          index == 0 ? (leg.vertical ? path.from.second : path.from.first)
                     : corridorMiddle(path.legs[index - 1].corridor, layout);
      double after =
          index + 1 == path.legs.size()
              ? (leg.vertical ? path.to.second : path.to.first)
              : corridorMiddle(path.legs[index + 1].corridor, layout);
      leg.lo = std::min(before, after);
      leg.hi = std::max(before, after);
    }
  }

  // Interval coloring: sorted by where each leg starts, a leg takes the
  // lowest track no other leg is still occupying. That is optimal here, and
  // it is what keeps two parallel legs off the same line.
  struct Slot {
    unsigned path;
    unsigned leg;
  };
  std::map<CorridorKey, SmallVector<Slot>> byCorridor;
  for (const Path &path : plans)
    if (path.valid)
      for (auto [index, leg] : llvm::enumerate(path.legs))
        byCorridor[leg.corridor].push_back({path.id, unsigned(index)});
  for (auto &[key, members] : byCorridor) {
    llvm::stable_sort(members, [&](const Slot &a, const Slot &b) {
      return plans[a.path].legs[a.leg].lo < plans[b.path].legs[b.leg].lo;
    });
    SmallVector<double> freeFrom;
    SmallVector<int> track(members.size(), 0);
    for (auto [index, member] : llvm::enumerate(members)) {
      Leg &leg = plans[member.path].legs[member.leg];
      int slot = -1;
      for (auto [candidate, end] : llvm::enumerate(freeFrom))
        if (end <= leg.lo + 1e-9) {
          slot = candidate;
          break;
        }
      if (slot < 0) {
        slot = freeFrom.size();
        freeFrom.push_back(0);
      }
      freeFrom[slot] = leg.hi;
      track[index] = slot;
    }
    // Tracks share out the corridor's width, so a busy corridor packs them
    // closer rather than spilling past its edge onto a tile.
    Corridor corridor = corridorOf(key, layout);
    int used = std::max<size_t>(1, freeFrom.size());
    double step = std::min(kTrack, std::abs(corridor.span) / used);
    for (auto [index, member] : llvm::enumerate(members))
      plans[member.path].legs[member.leg].cross =
          corridor.base + track[index] * std::copysign(step, corridor.span);
  }

  SmallVector<ArrowPath> paths;
  for (const ArrowRequest &request : requests) {
    const Path &path = plans[request.id];
    if (!path.valid || path.legs.empty())
      continue;

    SmallVector<Point> corners{path.from};
    Point cursor = path.from;
    for (const Leg &leg : path.legs) {
      cursor = leg.vertical ? Point{leg.cross, cursor.second}
                            : Point{cursor.first, leg.cross};
      corners.push_back(cursor);
    }
    corners.push_back(path.legs.back().vertical
                          ? Point{cursor.first, path.to.second}
                          : Point{path.to.first, cursor.second});
    corners.push_back(path.to);

    corners.front() = onEdge(request.from, corners.front(), corners[1]);
    corners.back() =
        onEdge(request.to, corners.back(), corners[corners.size() - 2]);

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
