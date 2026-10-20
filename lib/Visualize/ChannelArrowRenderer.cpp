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
// Every arrow is drawn the same way: a stub out of the box it leaves, one long
// run, and a stub into the box it enters. The stubs are perpendicular to the
// run and sit on their own box's line, so two stubs coincide only when they
// leave the same box, which is a real join.
//
// Each long run lies in a corridor -- a strip of empty space between tiles,
// inside a tile beside its contents, or across a switchbox. Corridors never
// overlap, so runs in different corridors cannot merge. Within one corridor a
// run is given a track, and tracks are handed out by interval colouring over
// the stretch each run occupies: two runs share a track only when they do not
// overlap along it. The corridor is as wide as its busiest point needs and no
// wider.
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
/// The widest a tile's buffer and call boxes are drawn.
constexpr double kContentWidth = 1.75;
/// However busy a tile is, it stops widening here.
constexpr double kWidestTile = 7.5;

double dmaCentreX(const LayoutGeometry &layout, int col) {
  return layout.tileLeft(col) - kDMAInset - kDMAHalfWidth;
}
double switchCentreX(const LayoutGeometry &layout, int col) {
  return layout.tileLeft(col) - kChannel + kSwitchInset;
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

/// One arrow's long run, before it knows which track it is on.
struct Run {
  unsigned id = 0;
  bool vertical = false;
  CorridorKey corridor;
  double lo = 0, hi = 0;
  Point from{0, 0}, to{0, 0};
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
Point onEdge(const ArrowEnd &end, Point centre, Point toward) {
  double dx = toward.first - centre.first;
  double dy = toward.second - centre.second;
  if (std::abs(dx) < 1e-9 && std::abs(dy) < 1e-9)
    return centre;
  double infinity = std::numeric_limits<double>::infinity();
  double scaleX = std::abs(dx) < 1e-9 ? infinity : end.halfWidth / std::abs(dx);
  double scaleY =
      std::abs(dy) < 1e-9 ? infinity : end.halfHeight / std::abs(dy);
  return {centre.first + dx * std::min({scaleX, scaleY, 1.0}),
          centre.second + dy * std::min({scaleX, scaleY, 1.0})};
}

class ChannelArrowRenderer : public ArrowRenderer {
public:
  std::optional<double> preferredTileGap() const override { return kChannel; }

  std::optional<double> widestLine() const override {
    // Leave better than half the gap between neighbouring tracks clear, or
    // two lines running side by side read as one thick one.
    return kTrack * 72.0 * 0.42;
  }

  std::optional<double> preferredTileWidth(unsigned contents) const override {
    // Contents down the middle, and a lane either side wide enough that the
    // arrows between them each get a track of their own. A tile with more
    // arrows than that stops growing and packs them closer instead: past a
    // point a wider tile costs the whole picture more than it buys this one.
    double wanted = kContentWidth + 2 * (contents * kTrack + 2 * kEdgeMargin);
    return std::min(wanted, kWidestTile);
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
  }

  void place(const ArrowEnd &end, const LayoutGeometry &layout,
             std::map<std::string, Point> &placed) const {
    if (end.node.empty() || placed.count(end.node))
      return;
    auto [col, row] = end.tile;
    if (end.kind == EndKind::Switch) {
      double cx = switchCentreX(layout, col);
      double cy = layout.channelY(row);
      auto found = switches.find(end.tile);
      double half = found == switches.end() ? 0.2 : found->second.halfHeight;
      // A port sits on the side it faces, so a flow passing straight through
      // never meets the ports facing the other way.
      auto spread = [&](Facing facing) {
        auto count = sides.find({end.tile, facing});
        int n = count == sides.end() ? 1 : count->second;
        return (end.slot - (n - 1) / 2.0) * kPortPitch;
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
      placed[end.node] = {dmaCentreX(layout, col), layout.tileTop(col, row) -
                                                       0.5 -
                                                       end.slot * kDMAPitch};
    }
  }

  Point positionOf(const ArrowEnd &end, const LayoutGeometry &layout) const {
    std::map<std::string, Point> one;
    place(end, layout, one);
    auto found = one.find(end.node);
    return found == one.end() ? end.pinned : found->second;
  }

  /// The strip a run lies in, and the stretch of it the run occupies.
  Run classify(const ArrowRequest &request, const LayoutGeometry &layout) const;
  Corridor corridorOf(const CorridorKey &key,
                      const LayoutGeometry &layout) const;

  std::map<std::pair<int, int>, SwitchSize> switches;
  std::map<std::pair<std::pair<int, int>, Facing>, int> sides;
  /// How wide a tile's contents are, which bounds the corridors either side
  /// of them, and where the runtime arguments start.
  double contentHalfWidth = 0;
  double argumentsTop = 0;
};

Run ChannelArrowRenderer::classify(const ArrowRequest &request,
                                   const LayoutGeometry &layout) const {
  Run run;
  run.id = request.id;
  run.from = positionOf(request.from, layout);
  run.to = positionOf(request.to, layout);
  run.valid = true;
  auto [col, row] = request.from.tile;
  bool sameTile = request.from.tile == request.to.tile;
  EndKind a = request.from.kind, b = request.to.kind;

  auto span = [&](bool vertical) {
    double lo = vertical ? std::min(run.from.second, run.to.second)
                         : std::min(run.from.first, run.to.first);
    double hi = vertical ? std::max(run.from.second, run.to.second)
                         : std::max(run.from.first, run.to.first);
    run.lo = lo;
    run.hi = hi;
    run.vertical = vertical;
  };

  if (a == EndKind::External || b == EndKind::External) {
    run.corridor = {CorridorKind::Arguments, 0, 0};
    span(/*vertical=*/false);
  } else if (sameTile && a == EndKind::Switch && b == EndKind::Switch) {
    run.corridor = {CorridorKind::Switchbox, col, row};
    span(/*vertical=*/false);
  } else if (sameTile && (a == EndKind::DMA || b == EndKind::DMA) &&
             (a == EndKind::Switch || b == EndKind::Switch)) {
    run.corridor = {CorridorKind::DMAApproach, col, row};
    span(/*vertical=*/true);
  } else if (sameTile && (a == EndKind::DMA || b == EndKind::DMA)) {
    run.corridor = {CorridorKind::TileIn, col, row};
    span(/*vertical=*/true);
  } else if (sameTile) {
    run.corridor = {CorridorKind::TileOut, col, row};
    span(/*vertical=*/true);
  } else if (request.from.tile.second != request.to.tile.second) {
    // Between switchboxes above and below: the step across happens beside the
    // row that lies between them.
    run.corridor = {CorridorKind::ColumnGap, col,
                    std::max(request.from.tile.second, request.to.tile.second)};
    span(/*vertical=*/true);
  } else {
    run.corridor = {CorridorKind::RowGap, row,
                    std::min(request.from.tile.first, request.to.tile.first)};
    span(/*vertical=*/false);
  }
  return run;
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
    double hi = switchCentreX(layout, col) - kSwitchHalfWidth - kEdgeMargin;
    double lo = layout.tileLeft(col) - kChannel + kEdgeMargin;
    corridor.base = hi;
    corridor.span = lo - hi;
    break;
  }
  case CorridorKind::RowGap: {
    int row = key.a;
    auto found = switches.find({key.b, key.a});
    double half = found == switches.end() ? 0.2 : found->second.halfHeight;
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
    double lo = switchCentreX(layout, col) + kSwitchHalfWidth + kEdgeMargin;
    double hi = dmaCentreX(layout, col) - kDMAHalfWidth - kEdgeMargin;
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

  SmallVector<Run> runs;
  runs.reserve(requests.size());
  for (const ArrowRequest &request : requests)
    runs.push_back(classify(request, layout));

  // Arrows meeting the same box would otherwise all run along its centre
  // line, which is the one place they cannot help but merge. Each takes its
  // own point on the edge it arrives at instead, so they stay apart until
  // they touch the box.
  std::map<std::pair<std::string, bool>, SmallVector<std::pair<unsigned, bool>>>
      arrivals;
  for (const ArrowRequest &request : requests) {
    const Run &run = runs[request.id];
    if (!run.valid)
      continue;
    // A run that is vertical meets its boxes on their left or right, so the
    // points it may take are spread up and down; a horizontal one the other
    // way round.
    for (auto [end, isFrom] :
         {std::pair{&request.from, true}, std::pair{&request.to, false}})
      arrivals[{end->node, run.vertical}].push_back({request.id, isFrom});
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
  auto attach = [&](unsigned id, bool isFrom, Point at, bool vertical) {
    auto found = offsetOf.find((uint64_t(id) << 1) | unsigned(isFrom));
    if (found == offsetOf.end())
      return at;
    return vertical ? Point{at.first, at.second + found->second}
                    : Point{at.first + found->second, at.second};
  };
  for (Run &run : runs) {
    if (!run.valid)
      continue;
    run.from = attach(run.id, true, run.from, run.vertical);
    run.to = attach(run.id, false, run.to, run.vertical);
    double lo = run.vertical ? std::min(run.from.second, run.to.second)
                             : std::min(run.from.first, run.to.first);
    double hi = run.vertical ? std::max(run.from.second, run.to.second)
                             : std::max(run.from.first, run.to.first);
    run.lo = lo;
    run.hi = hi;
  }

  // Interval colouring: sorted by where each run starts, a run takes the
  // lowest track no other run is still occupying. That is optimal here, and
  // it is what keeps two parallel runs off the same line.
  std::map<CorridorKey, SmallVector<unsigned>> byCorridor;
  for (const Run &run : runs)
    if (run.valid)
      byCorridor[run.corridor].push_back(run.id);
  DenseMap<unsigned, int> trackOf;
  DenseMap<unsigned, unsigned> keyIndex;
  DenseMap<unsigned, int> trackCount;
  unsigned nextKey = 0;
  for (auto &[key, members] : byCorridor) {
    unsigned index = nextKey++;
    llvm::stable_sort(members, [&](unsigned a, unsigned b) {
      return runs[a].lo < runs[b].lo;
    });
    SmallVector<double> freeFrom;
    for (unsigned id : members) {
      keyIndex[id] = index;
      int track = -1;
      for (auto [slot, end] : llvm::enumerate(freeFrom))
        if (end <= runs[id].lo + 1e-9) {
          track = slot;
          break;
        }
      if (track < 0) {
        track = freeFrom.size();
        freeFrom.push_back(0);
      }
      freeFrom[track] = runs[id].hi;
      trackOf[id] = track;
    }
    trackCount[index] = freeFrom.size();
  }

  SmallVector<ArrowPath> paths;
  for (const ArrowRequest &request : requests) {
    const Run &run = runs[request.id];
    if (!run.valid)
      continue;
    Corridor corridor = corridorOf(run.corridor, layout);
    int track = trackOf.lookup(request.id);
    // Tracks share out the corridor's width, so a busy corridor packs them
    // closer rather than spilling past its edge onto a tile.
    int used = std::max(1, trackCount.lookup(keyIndex.lookup(request.id)));
    double step = std::min(kTrack, std::abs(corridor.span) / used);
    double cross = corridor.base + track * std::copysign(step, corridor.span);

    SmallVector<Point> corners{run.from};
    if (run.vertical) {
      corners.push_back({cross, run.from.second});
      corners.push_back({cross, run.to.second});
    } else {
      corners.push_back({run.from.first, cross});
      corners.push_back({run.to.first, cross});
    }
    corners.push_back(run.to);

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
