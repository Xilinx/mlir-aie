//===- AvoidanceArrowRenderer.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArrowRenderer.h"
#include "LayoutGeometry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

using namespace llvm;

namespace xilinx::AIE {
namespace {

constexpr double kArcControl = 0.5523;
constexpr double kCornerRadius = 0.10;
constexpr double kEpsilon = 1e-9;
constexpr double kMinimumRenderedGap = 1.2;
constexpr uint64_t kNoParent = std::numeric_limits<uint64_t>::max();

enum class Direction : uint8_t { East, West, North, South };
enum class Side : uint8_t { East, West, North, South };

Direction directionOf(Side side) {
  switch (side) {
  case Side::East:
    return Direction::East;
  case Side::West:
    return Direction::West;
  case Side::North:
    return Direction::North;
  case Side::South:
    return Direction::South;
  }
  llvm_unreachable("unknown side");
}

Direction opposite(Direction direction) {
  switch (direction) {
  case Direction::East:
    return Direction::West;
  case Direction::West:
    return Direction::East;
  case Direction::North:
    return Direction::South;
  case Direction::South:
    return Direction::North;
  }
  llvm_unreachable("unknown direction");
}

Side opposite(Side side) {
  return static_cast<Side>(opposite(directionOf(side)));
}

bool horizontal(Direction direction) {
  return direction == Direction::East || direction == Direction::West;
}

Side sideOf(Facing facing) {
  switch (facing) {
  case Facing::North:
    return Side::North;
  case Facing::East:
    return Side::East;
  case Facing::South:
    return Side::South;
  case Facing::West:
    return Side::West;
  case Facing::Local:
    return Side::East;
  }
  llvm_unreachable("unknown facing");
}

void lineTo(SmallVectorImpl<Point> &out, Point to) {
  Point from = out.back();
  out.push_back({from.first + (to.first - from.first) / 3,
                 from.second + (to.second - from.second) / 3});
  out.push_back({from.first + 2 * (to.first - from.first) / 3,
                 from.second + 2 * (to.second - from.second) / 3});
  out.push_back(to);
}

Point unitBetween(Point a, Point b) {
  double dx = b.first - a.first;
  double dy = b.second - a.second;
  double length = std::hypot(dx, dy);
  return length < kEpsilon ? Point{0, 0} : Point{dx / length, dy / length};
}

SmallVector<Point> toBezier(ArrayRef<Point> corners, double radius) {
  SmallVector<Point> out;
  if (corners.size() < 2)
    return out;
  out.push_back(corners.front());
  for (size_t index = 1; index + 1 < corners.size(); ++index) {
    Point incoming = unitBetween(corners[index - 1], corners[index]);
    Point outgoing = unitBetween(corners[index], corners[index + 1]);
    double reach = std::min(
        {radius,
         std::hypot(corners[index].first - corners[index - 1].first,
                    corners[index].second - corners[index - 1].second) /
             2,
         std::hypot(corners[index + 1].first - corners[index].first,
                    corners[index + 1].second - corners[index].second) /
             2});
    Point enter{corners[index].first - incoming.first * reach,
                corners[index].second - incoming.second * reach};
    Point leave{corners[index].first + outgoing.first * reach,
                corners[index].second + outgoing.second * reach};
    lineTo(out, enter);
    out.push_back({enter.first + incoming.first * reach * kArcControl,
                   enter.second + incoming.second * reach * kArcControl});
    out.push_back({leave.first - outgoing.first * reach * kArcControl,
                   leave.second - outgoing.second * reach * kArcControl});
    out.push_back(leave);
  }
  lineTo(out, corners.back());
  return out;
}

struct InflatedObstacle {
  std::string node;
  std::optional<std::pair<int, int>> tile;
  Rectangle visible;
  Rectangle blocked;
  ObstacleKind kind;
};

struct Gateway {
  Point anchor;
  Point point;
  Direction outward;
  unsigned vertex = 0;
  double penalty = 0.0;
};

struct Blockage {
  uint16_t nodeCount = 0;
  int tile = -1;
};

struct RoutedPath {
  unsigned id = 0;
  SmallVector<unsigned> vertices;
  Point sourceAnchor{0, 0};
  Point destAnchor{0, 0};
  double cost = 0.0;
  bool valid = false;
};

struct ExplicitRoute {
  unsigned id = 0;
  unsigned flowGroup = 0;
  std::string from;
  std::string to;
  std::pair<int, int> fromTile;
  std::pair<int, int> toTile;
  SmallVector<Point> corners;
};

struct StubUsage {
  Point first;
  Point second;
  unsigned requestID = 0;
  unsigned count = 0;
};

struct QueueEntry {
  double estimate;
  double cost;
  uint64_t state;

  bool operator<(const QueueEntry &other) const {
    return std::tie(estimate, cost, state) >
           std::tie(other.estimate, other.cost, other.state);
  }
};

std::optional<unsigned> coordinateIndex(ArrayRef<double> values, double value) {
  const auto *found = llvm::lower_bound(values, value);
  if (found != values.end() && std::abs(*found - value) < kEpsilon)
    return found - values.begin();
  if (found != values.begin() && std::abs(*std::prev(found) - value) < kEpsilon)
    return std::prev(found) - values.begin();
  return std::nullopt;
}

unsigned requiredCoordinateIndex(ArrayRef<double> values, double value) {
  std::optional<unsigned> index = coordinateIndex(values, value);
  assert(index && "grid coordinate was not registered");
  return index.value_or(0);
}

class AvoidanceArrowRenderer : public ArrowRenderer {
public:
  bool suppliesLayout() const override { return true; }

  std::optional<double> widestLine() const override {
    return kTrackPitch * 72.0 * 0.42;
  }

  std::optional<double> preferredTileGap(double textRoom) const override {
    return 1.6 * textRoom;
  }

  std::optional<double> preferredTileWidth(unsigned contents,
                                           double contentWidth,
                                           double textRoom) const override {
    return std::max(contentWidth + 0.5 * textRoom, 2.15 * textRoom);
  }

  std::map<std::string, Point>
  nodePositions(ArrayRef<ArrowRequest> requests,
                const LayoutGeometry &layout) override;

  SmallVector<ArrowPath> route(ArrayRef<ArrowRequest> requests,
                               ArrayRef<RoutingObstacle> obstacles,
                               const LayoutGeometry &layout) override;

private:
  static constexpr double kTrackPitch = 0.45;

  void buildGrid(ArrayRef<ArrowRequest> requests,
                 ArrayRef<RoutingObstacle> obstacles,
                 const LayoutGeometry &layout);
  SmallVector<Side> sidesFor(const ArrowEnd &end, const ArrowEnd &other) const;
  SmallVector<Gateway> gatewaysFor(const ArrowEnd &end, const ArrowEnd &other,
                                  unsigned requestID,
                                  bool allowEndpointObstacles) const;
  double endpointSpread(StringRef node, unsigned requestID, double span) const;
  RoutedPath routeOne(const ArrowRequest &request, bool allowOccupied = false,
                      bool allowEndpointObstacles = false);
  void addUsage(const RoutedPath &path, int amount);
  double conflictScore(const RoutedPath &path) const;
  SmallVector<Point> cornersOf(const RoutedPath &path) const;
  void deconflict(SmallVectorImpl<ExplicitRoute> &routes) const;
  bool hasNearbyParallelUsage(unsigned x, unsigned y,
                              Direction direction, unsigned requestID) const;
  bool hasGatewayConflict(const Gateway &gateway, unsigned requestID) const;
  bool sharesEndpoint(unsigned first, unsigned second) const;
  void addStubUsage(Point first, Point second, unsigned requestID, int amount);

  unsigned vertex(unsigned x, unsigned y) const { return y * xs.size() + x; }
  std::pair<unsigned, unsigned> coordinates(unsigned vertex) const {
    return {vertex % xs.size(), vertex / xs.size()};
  }
  unsigned horizontalEdge(unsigned x, unsigned y) const {
    return y * (xs.size() - 1) + x;
  }
  unsigned verticalEdge(unsigned x, unsigned y) const {
    return y * xs.size() + x;
  }
  bool exemptTile(int tile, const ArrowRequest &request) const;
  bool open(const Blockage &blockage, const ArrowRequest &request) const;

  double clearance = 0.0;
  double bendPenalty = 0.0;
  double crossingPenalty = 0.0;
  double overlapPenalty = 0.0;
  double proximityPenalty = 0.0;
  double trackPitch = 0.0;
  double minimumGap = 0.0;
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<std::pair<unsigned, unsigned>> nearbyXs;
  std::vector<std::pair<unsigned, unsigned>> nearbyYs;
  std::vector<InflatedObstacle> scene;
  std::map<std::string, unsigned> obstacleByNode;
  std::map<std::pair<std::string, unsigned>, std::pair<unsigned, unsigned>>
      endpointRanks;
  std::vector<Blockage> vertexBlockage;
  std::vector<Blockage> horizontalBlockage;
  std::vector<Blockage> verticalBlockage;
  std::vector<uint16_t> horizontalUsage;
  std::vector<uint16_t> verticalUsage;
  std::vector<int32_t> horizontalOwner;
  std::vector<int32_t> verticalOwner;
  std::vector<uint16_t> vertexHorizontalUsage;
  std::vector<uint16_t> vertexVerticalUsage;
  std::vector<StubUsage> stubUsage;
  std::vector<std::pair<unsigned, unsigned>> requestEndpoints;
  std::vector<double> routeDistance;
  std::vector<uint64_t> routeParent;
  std::vector<uint32_t> routeGeneration;
  uint32_t generation = 0;
  uint64_t routeCalls = 0;
  uint64_t expandedStates = 0;
};

std::map<std::string, Point>
AvoidanceArrowRenderer::nodePositions(ArrayRef<ArrowRequest> requests,
                                      const LayoutGeometry &layout) {
  std::map<std::pair<std::pair<int, int>, Facing>, unsigned> counts;
  for (const ArrowRequest &request : requests)
    for (const ArrowEnd *end : {&request.from, &request.to})
      if (end->kind == EndKind::Switch)
        counts[{end->tile, end->facing}] =
            std::max(counts[{end->tile, end->facing}], unsigned(end->slot + 1));

  std::map<std::string, Point> placed;
  double pitch = layout.text(0.22);
  double offset = layout.text(0.08);
  double dmaPitch = layout.text(0.24);
  for (const ArrowRequest &request : requests) {
    for (const ArrowEnd *end : {&request.from, &request.to}) {
      if (placed.count(end->node))
        continue;
      auto [col, row] = end->tile;
      if (end->kind == EndKind::DMA) {
        placed[end->node] = {layout.tileLeft(col) - layout.text(0.7),
                             layout.tileTop(col, row) - layout.text(0.5) -
                                 end->slot * dmaPitch};
        continue;
      }
      if (end->kind != EndKind::Switch)
        continue;
      unsigned count = counts[{end->tile, end->facing}];
      double spread = (end->slot - (count - 1) / 2.0) * pitch;
      switch (end->facing) {
      case Facing::North:
        placed[end->node] = {layout.tileCenterX(col) + spread,
                             layout.tileTop(col, row) + offset};
        break;
      case Facing::South:
        placed[end->node] = {layout.tileCenterX(col) + spread,
                             layout.tileBottom(col, row) - offset};
        break;
      case Facing::East:
      case Facing::Local:
        placed[end->node] = {layout.tileRight(col) + offset,
                             layout.tileCenterY(col, row) + spread};
        break;
      case Facing::West:
        placed[end->node] = {layout.tileLeft(col) - offset,
                             layout.tileCenterY(col, row) + spread};
        break;
      }
    }
  }
  return placed;
}

SmallVector<Side>
AvoidanceArrowRenderer::sidesFor(const ArrowEnd &end,
                                 const ArrowEnd &other) const {
  if (end.kind == EndKind::Switch) {
    Side preferred;
    if (end.facing == Facing::Local)
      preferred = other.kind == EndKind::DMA ? Side::East : Side::West;
    else {
      Side facing = sideOf(end.facing);
      bool wire = other.kind == EndKind::Switch && other.tile != end.tile;
      preferred = wire ? facing : opposite(facing);
    }
    SmallVector<Side> sides{preferred};
    for (Side side : {Side::North, Side::East, Side::South, Side::West})
      if (side != preferred)
        sides.push_back(side);
    return sides;
  }
  if (end.kind == EndKind::DMA)
    return other.kind == EndKind::Switch
               ? SmallVector<Side>{Side::West, Side::North, Side::South,
                                   Side::East}
               : SmallVector<Side>{Side::East, Side::North, Side::South,
                                   Side::West};
  if (end.kind == EndKind::External)
    return {Side::North, Side::East, Side::West, Side::South};
  return {Side::North, Side::East, Side::South, Side::West};
}

SmallVector<Gateway>
AvoidanceArrowRenderer::gatewaysFor(const ArrowEnd &end, const ArrowEnd &other,
                                    unsigned requestID,
                                    bool allowEndpointObstacles) const {
  auto found = obstacleByNode.find(end.node);
  if (found == obstacleByNode.end())
    return {};
  const InflatedObstacle &obstacle = scene[found->second];
  SmallVector<Gateway> gateways;
  if (obstacle.kind == ObstacleKind::Tile) {
    Point center{(obstacle.visible.left + obstacle.visible.right) / 2,
                 (obstacle.visible.bottom + obstacle.visible.top) / 2};
    std::optional<unsigned> x = coordinateIndex(xs, center.first);
    std::optional<unsigned> y = coordinateIndex(ys, center.second);
    if (x && y) {
      unsigned centerVertex = vertex(*x, *y);
      for (Direction direction : {Direction::North, Direction::East,
                                  Direction::South, Direction::West})
        gateways.push_back({center, center, direction, centerVertex, 0.0});
    }
  }
  for (auto [rank, side] : llvm::enumerate(sidesFor(end, other))) {
    Gateway gateway;
    gateway.outward = directionOf(side);
    gateway.penalty = rank == 0 ? 0.0 : bendPenalty;
    double horizontalSpread = endpointSpread(
        end.node, requestID, obstacle.visible.right - obstacle.visible.left);
    double verticalSpread = endpointSpread(
        end.node, requestID, obstacle.visible.top - obstacle.visible.bottom);
    switch (side) {
    case Side::East:
      gateway.anchor = {obstacle.visible.right,
                        (obstacle.visible.bottom + obstacle.visible.top) / 2 +
                            verticalSpread};
      gateway.point = {obstacle.blocked.right, gateway.anchor.second};
      break;
    case Side::West:
      gateway.anchor = {obstacle.visible.left,
                        (obstacle.visible.bottom + obstacle.visible.top) / 2 +
                            verticalSpread};
      gateway.point = {obstacle.blocked.left, gateway.anchor.second};
      break;
    case Side::North:
      gateway.anchor = {(obstacle.visible.left + obstacle.visible.right) / 2,
                        obstacle.visible.top};
      gateway.anchor.first += horizontalSpread;
      gateway.point = {gateway.anchor.first, obstacle.blocked.top};
      break;
    case Side::South:
      gateway.anchor = {(obstacle.visible.left + obstacle.visible.right) / 2,
                        obstacle.visible.bottom};
      gateway.anchor.first += horizontalSpread;
      gateway.point = {gateway.anchor.first, obstacle.blocked.bottom};
      break;
    }
    std::optional<unsigned> x = coordinateIndex(xs, gateway.point.first);
    std::optional<unsigned> y = coordinateIndex(ys, gateway.point.second);
    if (!x || !y)
      continue;
    bool blocked = llvm::any_of(scene, [&](const InflatedObstacle &candidate) {
      if (candidate.node == end.node || candidate.kind == ObstacleKind::Tile)
        return false;
      bool useVisibleBounds = allowEndpointObstacles &&
                              end.kind != EndKind::External &&
                              candidate.tile == end.tile;
      double left = std::min(gateway.anchor.first, gateway.point.first);
      double right = std::max(gateway.anchor.first, gateway.point.first);
      double bottom = std::min(gateway.anchor.second, gateway.point.second);
      double top = std::max(gateway.anchor.second, gateway.point.second);
      return left < (useVisibleBounds ? candidate.visible.right
                                      : candidate.blocked.right) -
                        kEpsilon &&
             right > (useVisibleBounds ? candidate.visible.left
                                       : candidate.blocked.left) +
                         kEpsilon &&
             bottom < (useVisibleBounds ? candidate.visible.top
                                        : candidate.blocked.top) -
                          kEpsilon &&
             top > (useVisibleBounds ? candidate.visible.bottom
                                     : candidate.blocked.bottom) +
                       kEpsilon;
    });
    if (blocked)
      continue;
    gateway.vertex = vertex(*x, *y);
    gateways.push_back(gateway);
  }
  return gateways;
}

double AvoidanceArrowRenderer::endpointSpread(StringRef node,
                                              unsigned requestID,
                                              double span) const {
  auto found = endpointRanks.find({node.str(), requestID});
  if (found == endpointRanks.end() || found->second.second < 2)
    return 0.0;
  auto [rank, count] = found->second;
  return (rank - (count - 1) / 2.0) * trackPitch;
}

bool AvoidanceArrowRenderer::exemptTile(int tile,
                                        const ArrowRequest &request) const {
  if (tile < 0)
    return false;
  const InflatedObstacle &obstacle = scene[tile];
  return (request.from.kind != EndKind::External &&
          obstacle.tile == request.from.tile) ||
         (request.to.kind != EndKind::External &&
          obstacle.tile == request.to.tile);
}

bool AvoidanceArrowRenderer::open(const Blockage &blockage,
                                  const ArrowRequest &request) const {
  return blockage.nodeCount == 0 &&
         (blockage.tile < 0 || exemptTile(blockage.tile, request));
}

void AvoidanceArrowRenderer::buildGrid(ArrayRef<ArrowRequest> requests,
                                       ArrayRef<RoutingObstacle> obstacles,
                                       const LayoutGeometry &layout) {
  double maxWidth = 0.0;
  for (const ArrowRequest &request : requests)
    maxWidth = std::max(maxWidth, request.width);
  clearance = layout.text(0.07) + maxWidth / (144.0 * layout.scale);
  bendPenalty = layout.text(1.0);
  crossingPenalty = layout.text(3.0);
  overlapPenalty = 12.0;
  proximityPenalty = 0.3;
  trackPitch = layout.text(kTrackPitch);
  minimumGap = kMinimumRenderedGap / (72.0 * layout.scale);

  scene.clear();
  obstacleByNode.clear();
  endpointRanks.clear();
  requestEndpoints.clear();
  xs.clear();
  ys.clear();
  std::set<std::string> endpointNodes;
  std::map<std::string, SmallVector<unsigned>> endpointRequests;
  std::map<std::string, unsigned> endpointIDs;
  requestEndpoints.resize(requests.size());
  for (const ArrowRequest &request : requests) {
    auto endpointID = [&](const std::string &node) {
      auto [found, inserted] = endpointIDs.try_emplace(node, endpointIDs.size());
      return found->second;
    };
    requestEndpoints[request.id] = {endpointID(request.from.node),
                                    endpointID(request.to.node)};
    endpointNodes.insert(request.from.node);
    endpointNodes.insert(request.to.node);
    endpointRequests[request.from.node].push_back(request.id);
    endpointRequests[request.to.node].push_back(request.id);
  }
  for (auto &[node, ids] : endpointRequests) {
    llvm::sort(ids);
    ids.erase(llvm::unique(ids), ids.end());
    for (auto [rank, id] : llvm::enumerate(ids))
      endpointRanks[{node, id}] = {rank, ids.size()};
  }
  for (const RoutingObstacle &obstacle : obstacles) {
    Rectangle blocked{
        obstacle.bounds.left - clearance, obstacle.bounds.right + clearance,
        obstacle.bounds.bottom - clearance, obstacle.bounds.top + clearance};
    unsigned id = scene.size();
    scene.push_back({obstacle.node, obstacle.tile, obstacle.bounds, blocked,
                     obstacle.kind});
    obstacleByNode[obstacle.node] = id;
    xs.push_back(blocked.left);
    xs.push_back(blocked.right);
    ys.push_back(blocked.bottom);
    ys.push_back(blocked.top);
    xs.push_back(blocked.left - trackPitch);
    xs.push_back(blocked.right + trackPitch);
    ys.push_back(blocked.bottom - trackPitch);
    ys.push_back(blocked.top + trackPitch);
    if (endpointNodes.count(obstacle.node)) {
      double centerX = (obstacle.bounds.left + obstacle.bounds.right) / 2;
      double centerY = (obstacle.bounds.bottom + obstacle.bounds.top) / 2;
      for (unsigned id : endpointRequests[obstacle.node]) {
        xs.push_back(centerX + endpointSpread(obstacle.node, id,
                                              obstacle.bounds.right -
                                                  obstacle.bounds.left));
        ys.push_back(centerY + endpointSpread(obstacle.node, id,
                                              obstacle.bounds.top -
                                                  obstacle.bounds.bottom));
      }
    }
    if (obstacle.kind == ObstacleKind::Tile) {
      for (unsigned track = 2; track <= 5; ++track) {
        double offset = track * trackPitch;
        xs.push_back(blocked.left - offset);
        xs.push_back(blocked.right + offset);
        ys.push_back(blocked.bottom - offset);
        ys.push_back(blocked.top + offset);
      }
    }
  }

  auto deduplicate = [](std::vector<double> &values) {
    llvm::sort(values);
    values.erase(llvm::unique(values,
                              [](double first, double second) {
                                return std::abs(first - second) < kEpsilon;
                              }),
                 values.end());
  };
  deduplicate(xs);
  deduplicate(ys);
  double margin = layout.text(0.8);
  xs.insert(xs.begin(), xs.front() - margin);
  xs.push_back(xs.back() + margin);
  ys.insert(ys.begin(), ys.front() - margin);
  ys.push_back(ys.back() + margin);

  auto nearbyRanges = [&](ArrayRef<double> coordinates) {
    std::vector<std::pair<unsigned, unsigned>> ranges;
    ranges.reserve(coordinates.size());
    for (double coordinate : coordinates) {
      unsigned first = llvm::lower_bound(coordinates,
                                         coordinate - minimumGap) -
                       coordinates.begin();
      unsigned last = llvm::upper_bound(coordinates,
                                        coordinate + minimumGap) -
                      coordinates.begin();
      ranges.emplace_back(first, last);
    }
    return ranges;
  };
  nearbyXs = nearbyRanges(xs);
  nearbyYs = nearbyRanges(ys);

  vertexBlockage.assign(xs.size() * ys.size(), {});
  horizontalBlockage.assign((xs.size() - 1) * ys.size(), {});
  verticalBlockage.assign(xs.size() * (ys.size() - 1), {});
  auto mark = [](Blockage &blockage, unsigned obstacle, bool tile) {
    if (tile) {
      blockage.tile = obstacle;
    } else {
      ++blockage.nodeCount;
    }
  };
  for (auto [obstacleID, obstacle] : llvm::enumerate(scene)) {
    unsigned left = requiredCoordinateIndex(xs, obstacle.blocked.left);
    unsigned right = requiredCoordinateIndex(xs, obstacle.blocked.right);
    unsigned bottom = requiredCoordinateIndex(ys, obstacle.blocked.bottom);
    unsigned top = requiredCoordinateIndex(ys, obstacle.blocked.top);
    bool tile = obstacle.kind == ObstacleKind::Tile;
    for (unsigned y = bottom + 1; y < top; ++y) {
      for (unsigned x = left + 1; x < right; ++x)
        mark(vertexBlockage[vertex(x, y)], obstacleID, tile);
      for (unsigned x = left; x < right; ++x)
        mark(horizontalBlockage[horizontalEdge(x, y)], obstacleID, tile);
    }
    for (unsigned y = bottom; y < top; ++y)
      for (unsigned x = left + 1; x < right; ++x)
        mark(verticalBlockage[verticalEdge(x, y)], obstacleID, tile);
  }

  horizontalUsage.assign(horizontalBlockage.size(), 0);
  verticalUsage.assign(verticalBlockage.size(), 0);
  horizontalOwner.assign(horizontalBlockage.size(), -1);
  verticalOwner.assign(verticalBlockage.size(), -1);
  vertexHorizontalUsage.assign(vertexBlockage.size(), 0);
  vertexVerticalUsage.assign(vertexBlockage.size(), 0);
  stubUsage.clear();
  routeDistance.resize(vertexBlockage.size() * 4);
  routeParent.resize(vertexBlockage.size() * 4);
  routeGeneration.assign(vertexBlockage.size() * 4, 0);
  generation = 0;
}

RoutedPath AvoidanceArrowRenderer::routeOne(const ArrowRequest &request,
                      bool allowOccupied,
                      bool allowEndpointObstacles) {
  SmallVector<Gateway> sources =
    gatewaysFor(request.from, request.to, request.id,
          allowEndpointObstacles);
  SmallVector<Gateway> targets =
    gatewaysFor(request.to, request.from, request.id,
          allowEndpointObstacles);
  if (!allowOccupied) {
    llvm::erase_if(sources, [&](const Gateway &gateway) {
      return hasGatewayConflict(gateway, request.id);
    });
    llvm::erase_if(targets, [&](const Gateway &gateway) {
      return hasGatewayConflict(gateway, request.id);
    });
  }
  RoutedPath result;
  result.id = request.id;
  if (sources.empty() || targets.empty())
    return result;

  double minX = std::numeric_limits<double>::infinity();
  double maxX = -std::numeric_limits<double>::infinity();
  double minY = std::numeric_limits<double>::infinity();
  double maxY = -std::numeric_limits<double>::infinity();
  for (const Gateway &gateway : llvm::concat<Gateway>(sources, targets)) {
    minX = std::min(minX, gateway.point.first);
    maxX = std::max(maxX, gateway.point.first);
    minY = std::min(minY, gateway.point.second);
    maxY = std::max(maxY, gateway.point.second);
  }
  double corridorMargin = 2.0 * bendPenalty;
  minX -= corridorMargin;
  maxX += corridorMargin;
  minY -= corridorMargin;
  maxY += corridorMargin;

  if (++generation == 0) {
    llvm::fill(routeGeneration, 0);
    ++generation;
  }
  std::priority_queue<QueueEntry> queue;
  auto encode = [](unsigned vertex, Direction direction) {
    return uint64_t(vertex) * 4 + static_cast<unsigned>(direction);
  };
  auto heuristic = [&](unsigned candidate) {
    auto [x, y] = coordinates(candidate);
    double best = std::numeric_limits<double>::infinity();
    for (const Gateway &target : targets) {
      auto [targetX, targetY] = coordinates(target.vertex);
      best = std::min(best, std::abs(xs[x] - xs[targetX]) +
                                std::abs(ys[y] - ys[targetY]) + target.penalty);
    }
    return best;
  };
  for (const Gateway &source : sources) {
    if (!open(vertexBlockage[source.vertex], request))
      continue;
    uint64_t state = encode(source.vertex, source.outward);
    double initial = std::hypot(source.point.first - source.anchor.first,
                                source.point.second - source.anchor.second) +
                     source.penalty;
    if (routeGeneration[state] == generation && routeDistance[state] <= initial)
      continue;
    routeGeneration[state] = generation;
    routeDistance[state] = initial;
    routeParent[state] = kNoParent;
    queue.push({initial + heuristic(source.vertex), initial, state});
  }

  uint64_t goal = kNoParent;
  const Gateway *goalGateway = nullptr;
  while (!queue.empty()) {
    QueueEntry current = queue.top();
    queue.pop();
    if (routeGeneration[current.state] != generation ||
        current.cost > routeDistance[current.state] + kEpsilon)
      continue;
    ++expandedStates;
    unsigned currentVertex = current.state / 4;
    Direction incoming = static_cast<Direction>(current.state % 4);
    for (const Gateway &target : targets)
      if (target.vertex == currentVertex &&
          incoming == opposite(target.outward)) {
        goal = current.state;
        goalGateway = &target;
        break;
      }
    if (goal != kNoParent)
      break;

    auto [x, y] = coordinates(currentVertex);
    struct Step {
      Direction direction;
      int dx;
      int dy;
    };
    constexpr std::array<Step, 4> steps{{
        {Direction::East, 1, 0},
        {Direction::West, -1, 0},
        {Direction::North, 0, 1},
        {Direction::South, 0, -1},
    }};
    for (const Step &step : steps) {
      if (step.direction == opposite(incoming))
        continue;
      int nextX = int(x) + step.dx;
      int nextY = int(y) + step.dy;
      if (nextX < 0 || nextY < 0 || nextX >= int(xs.size()) ||
          nextY >= int(ys.size()))
        continue;
      if (xs[nextX] < minX || xs[nextX] > maxX || ys[nextY] < minY ||
          ys[nextY] > maxY)
        continue;
      unsigned nextVertex = vertex(nextX, nextY);
      if (!open(vertexBlockage[nextVertex], request))
        continue;

      unsigned edge;
      const Blockage *blockage;
      uint16_t usage;
      if (horizontal(step.direction)) {
        edge = horizontalEdge(std::min<unsigned>(x, nextX), y);
        blockage = &horizontalBlockage[edge];
        usage = horizontalUsage[edge];
      } else {
        edge = verticalEdge(x, std::min<unsigned>(y, nextY));
        blockage = &verticalBlockage[edge];
        usage = verticalUsage[edge];
      }
      if (!open(*blockage, request))
        continue;
      if (usage != 0 && !allowOccupied)
        continue;
        if (!allowOccupied &&
          hasNearbyParallelUsage(x, y, step.direction, request.id))
        continue;

      double length = std::abs(xs[x] - xs[nextX]) + std::abs(ys[y] - ys[nextY]);
      double cost = current.cost + length;
      if (incoming != step.direction)
        cost += bendPenalty;
      cost += usage * overlapPenalty * length;
      cost += (horizontal(step.direction) ? vertexVerticalUsage[nextVertex]
                                          : vertexHorizontalUsage[nextVertex]) *
              crossingPenalty;
      if (horizontal(step.direction)) {
        if (y > 0)
          cost += horizontalUsage[horizontalEdge(std::min<unsigned>(x, nextX),
                                                 y - 1)] *
                  proximityPenalty * length;
        if (y + 1 < ys.size())
          cost += horizontalUsage[horizontalEdge(std::min<unsigned>(x, nextX),
                                                 y + 1)] *
                  proximityPenalty * length;
      } else {
        if (x > 0)
          cost +=
              verticalUsage[verticalEdge(x - 1, std::min<unsigned>(y, nextY))] *
              proximityPenalty * length;
        if (x + 1 < xs.size())
          cost +=
              verticalUsage[verticalEdge(x + 1, std::min<unsigned>(y, nextY))] *
              proximityPenalty * length;
      }

      uint64_t nextState = encode(nextVertex, step.direction);
      if (routeGeneration[nextState] == generation &&
          routeDistance[nextState] <= cost + kEpsilon)
        continue;
      routeGeneration[nextState] = generation;
      routeDistance[nextState] = cost;
      routeParent[nextState] = current.state;
      queue.push({cost + heuristic(nextVertex), cost, nextState});
    }
  }
  if (goal == kNoParent)
    return result;

  SmallVector<uint64_t> reversedStates;
  for (uint64_t state = goal; state != kNoParent; state = routeParent[state])
    reversedStates.push_back(state);
  SmallVector<uint64_t> states(reversedStates.rbegin(), reversedStates.rend());
  for (auto pair : llvm::zip_equal(ArrayRef(states).drop_back(),
                                   ArrayRef(states).drop_front()))
    assert(static_cast<Direction>(std::get<1>(pair) % 4) !=
           opposite(static_cast<Direction>(std::get<0>(pair) % 4)));
  for (uint64_t state : states)
    result.vertices.push_back(state / 4);
  const Gateway *sourceGateway = nullptr;
  for (const Gateway &source : sources)
    if (source.vertex == result.vertices.front() &&
        source.outward == static_cast<Direction>(states.front() % 4)) {
      sourceGateway = &source;
      break;
    }
  if (!sourceGateway || !goalGateway)
    return {};
  result.sourceAnchor = sourceGateway->anchor;
  result.destAnchor = goalGateway->anchor;
  result.cost =
      routeDistance[goal] +
      std::hypot(goalGateway->point.first - goalGateway->anchor.first,
                 goalGateway->point.second - goalGateway->anchor.second) +
      goalGateway->penalty;
  result.valid = true;
  return result;
}

bool AvoidanceArrowRenderer::hasNearbyParallelUsage(
  unsigned x, unsigned y, Direction direction, unsigned requestID) const {
  if (horizontal(direction)) {
    unsigned edgeX = direction == Direction::East ? x : x - 1;
    auto [first, last] = nearbyYs[y];
    for (unsigned otherY = first; otherY < last; ++otherY) {
      unsigned edge = horizontalEdge(edgeX, otherY);
      if (otherY != y && std::abs(ys[otherY] - ys[y]) < minimumGap &&
          horizontalUsage[edge] != 0 &&
          (horizontalUsage[edge] != 1 ||
           !sharesEndpoint(requestID, horizontalOwner[edge])))
          return true;
    }
    return false;
  }

  unsigned edgeY = direction == Direction::North ? y : y - 1;
  auto [first, last] = nearbyXs[x];
  for (unsigned otherX = first; otherX < last; ++otherX) {
    unsigned edge = verticalEdge(otherX, edgeY);
    if (otherX != x && std::abs(xs[otherX] - xs[x]) < minimumGap &&
      verticalUsage[edge] != 0 &&
        (verticalUsage[edge] != 1 ||
         !sharesEndpoint(requestID, verticalOwner[edge])))
      return true;
  }
  return false;
}

bool AvoidanceArrowRenderer::hasGatewayConflict(const Gateway &gateway,
                                                unsigned requestID) const {
  bool horizontalStub =
      std::abs(gateway.anchor.second - gateway.point.second) < kEpsilon;
  double cross = horizontalStub ? gateway.anchor.second : gateway.anchor.first;
  double low = horizontalStub
                   ? std::min(gateway.anchor.first, gateway.point.first)
                   : std::min(gateway.anchor.second, gateway.point.second);
  double high = horizontalStub
                    ? std::max(gateway.anchor.first, gateway.point.first)
                    : std::max(gateway.anchor.second, gateway.point.second);
  const std::vector<double> &parallel = horizontalStub ? ys : xs;
  const std::vector<double> &along = horizontalStub ? xs : ys;
  auto first = llvm::lower_bound(parallel, cross - minimumGap);
  auto last = llvm::upper_bound(parallel, cross + minimumGap);
  unsigned alongFirst = llvm::upper_bound(along, low) - along.begin();
  unsigned alongLast = std::min<unsigned>(
      llvm::lower_bound(along, high) - along.begin(), along.size() - 1);
  for (auto other = first; other != last; ++other) {
    if (std::abs(*other - cross) >= minimumGap)
      continue;
    unsigned crossIndex = other - parallel.begin();
    for (unsigned index = alongFirst; index <= alongLast && index > 0;
         ++index) {
      unsigned edge = horizontalStub
                          ? horizontalEdge(index - 1, crossIndex)
                          : verticalEdge(crossIndex, index - 1);
        unsigned usage =
          horizontalStub ? horizontalUsage[edge] : verticalUsage[edge];
        int32_t owner =
          horizontalStub ? horizontalOwner[edge] : verticalOwner[edge];
        if (usage != 0 &&
            (usage != 1 || !sharesEndpoint(requestID, owner)))
        return true;
    }
  }
  for (const StubUsage &stub : stubUsage) {
    if (stub.count == 0 || sharesEndpoint(requestID, stub.requestID))
      continue;
    bool stubHorizontal =
        std::abs(stub.first.second - stub.second.second) < kEpsilon;
    if (stubHorizontal != horizontalStub)
      continue;
    double stubCross = stubHorizontal ? stub.first.second : stub.first.first;
    double stubLow = stubHorizontal
                         ? std::min(stub.first.first, stub.second.first)
                         : std::min(stub.first.second, stub.second.second);
    double stubHigh = stubHorizontal
                          ? std::max(stub.first.first, stub.second.first)
                          : std::max(stub.first.second, stub.second.second);
    if (std::abs(stubCross - cross) < minimumGap &&
        std::min(high, stubHigh) > std::max(low, stubLow) + kEpsilon)
      return true;
  }
  return false;
}

bool AvoidanceArrowRenderer::sharesEndpoint(unsigned first,
                                            unsigned second) const {
  if (first >= requestEndpoints.size() || second >= requestEndpoints.size())
    return false;
  auto [firstSource, firstDest] = requestEndpoints[first];
  auto [secondSource, secondDest] = requestEndpoints[second];
  return firstSource == secondSource || firstSource == secondDest ||
         firstDest == secondSource || firstDest == secondDest;
}

void AvoidanceArrowRenderer::addStubUsage(Point first, Point second,
                                          unsigned requestID, int amount) {
  if (std::hypot(first.first - second.first, first.second - second.second) <
      kEpsilon)
    return;
  auto found = llvm::find_if(stubUsage, [&](const StubUsage &stub) {
    return stub.first == first && stub.second == second &&
           stub.requestID == requestID;
  });
  if (found == stubUsage.end()) {
    assert(amount > 0 && "cannot remove an unknown gateway stub");
    stubUsage.push_back({first, second, requestID, unsigned(amount)});
  } else {
    found->count += amount;
  }
}

void AvoidanceArrowRenderer::addUsage(const RoutedPath &path, int amount) {
  if (!path.valid || path.vertices.size() < 2)
    return;
  auto pointOf = [&](unsigned vertexID) {
    auto [x, y] = coordinates(vertexID);
    return Point{xs[x], ys[y]};
  };
  addStubUsage(path.sourceAnchor, pointOf(path.vertices.front()), path.id,
               amount);
  addStubUsage(pointOf(path.vertices.back()), path.destAnchor, path.id, amount);
  for (auto pair : llvm::zip_equal(ArrayRef(path.vertices).drop_back(),
                                   ArrayRef(path.vertices).drop_front())) {
    auto [from, to] = pair;
    auto [fromX, fromY] = coordinates(from);
    auto [toX, toY] = coordinates(to);
    if (fromY == toY) {
      unsigned edge = horizontalEdge(std::min(fromX, toX), fromY);
      uint16_t before = horizontalUsage[edge];
      horizontalUsage[edge] += amount;
      horizontalOwner[edge] =
          amount > 0 && before == 0 ? int32_t(path.id) : -1;
      vertexHorizontalUsage[from] += amount;
      vertexHorizontalUsage[to] += amount;
    } else if (fromX == toX) {
      unsigned edge = verticalEdge(fromX, std::min(fromY, toY));
      uint16_t before = verticalUsage[edge];
      verticalUsage[edge] += amount;
      verticalOwner[edge] =
          amount > 0 && before == 0 ? int32_t(path.id) : -1;
      vertexVerticalUsage[from] += amount;
      vertexVerticalUsage[to] += amount;
    }
  }
}

double AvoidanceArrowRenderer::conflictScore(const RoutedPath &path) const {
  if (!path.valid || path.vertices.size() < 2)
    return 0.0;
  double score = 0.0;
  auto pointOf = [&](unsigned vertexID) {
    auto [x, y] = coordinates(vertexID);
    return Point{xs[x], ys[y]};
  };
  for (auto [anchor, point] :
       {std::pair{path.sourceAnchor, pointOf(path.vertices.front())},
        std::pair{path.destAnchor, pointOf(path.vertices.back())}}) {
    Direction outward = std::abs(anchor.first - point.first) < kEpsilon
                            ? Direction::North
                            : Direction::East;
    if (hasGatewayConflict({anchor, point, outward}, path.id))
      score += 1024.0;
  }
  for (auto pair : llvm::zip_equal(ArrayRef(path.vertices).drop_back(),
                                   ArrayRef(path.vertices).drop_front())) {
    auto [from, to] = pair;
    auto [fromX, fromY] = coordinates(from);
    auto [toX, toY] = coordinates(to);
    if (fromY == toY) {
      score += 1024.0 *
               (horizontalUsage[horizontalEdge(std::min(fromX, toX), fromY)] -
                1);
      Direction direction = fromX < toX ? Direction::East : Direction::West;
      if (hasNearbyParallelUsage(fromX, fromY, direction, path.id))
        score += 1024.0;
      score += vertexVerticalUsage[to];
    } else {
      score += 1024.0 *
               (verticalUsage[verticalEdge(fromX, std::min(fromY, toY))] - 1);
      Direction direction = fromY < toY ? Direction::North : Direction::South;
      if (hasNearbyParallelUsage(fromX, fromY, direction, path.id))
        score += 1024.0;
      score += vertexHorizontalUsage[to];
    }
  }
  return score;
}

SmallVector<Point>
AvoidanceArrowRenderer::cornersOf(const RoutedPath &path) const {
  SmallVector<Point> points{path.sourceAnchor};
  for (unsigned vertexID : path.vertices) {
    auto [x, y] = coordinates(vertexID);
    Point point{xs[x], ys[y]};
    if (std::hypot(point.first - points.back().first,
                   point.second - points.back().second) > kEpsilon)
      points.push_back(point);
  }
  if (std::hypot(path.destAnchor.first - points.back().first,
                 path.destAnchor.second - points.back().second) > kEpsilon)
    points.push_back(path.destAnchor);

  SmallVector<Point> corners;
  for (Point point : points) {
    while (corners.size() >= 2) {
      Point first = corners[corners.size() - 2];
      Point second = corners.back();
      bool sameX = std::abs(first.first - second.first) < kEpsilon &&
                   std::abs(second.first - point.first) < kEpsilon;
      bool sameY = std::abs(first.second - second.second) < kEpsilon &&
                   std::abs(second.second - point.second) < kEpsilon;
      if (!sameX && !sameY)
        break;
      corners.pop_back();
    }
    corners.push_back(point);
  }
  return corners;
}

void AvoidanceArrowRenderer::deconflict(
    SmallVectorImpl<ExplicitRoute> &routes) const {
  struct Segment {
    unsigned route = 0;
    unsigned index = 0;
    bool horizontal = false;
    double cross = 0.0;
    double low = 0.0;
    double high = 0.0;
  };

  auto segments = [&]() {
    SmallVector<Segment> result;
    for (unsigned route = 0; route < routes.size(); ++route) {
      ArrayRef<Point> corners = routes[route].corners;
      for (unsigned index = 0; index + 1 < corners.size(); ++index) {
        Point first = corners[index];
        Point second = corners[index + 1];
        bool isHorizontal = std::abs(first.second - second.second) < kEpsilon;
        result.push_back({route, index, isHorizontal,
                          isHorizontal ? first.second : first.first,
                          isHorizontal ? std::min(first.first, second.first)
                                       : std::min(first.second, second.second),
                          isHorizontal
                              ? std::max(first.first, second.first)
                              : std::max(first.second, second.second)});
      }
    }
    return result;
  };
  auto segmentsOf = [&](unsigned route) {
    SmallVector<Segment> result;
    ArrayRef<Point> corners = routes[route].corners;
    for (unsigned index = 0; index + 1 < corners.size(); ++index) {
      Point first = corners[index];
      Point second = corners[index + 1];
      bool isHorizontal = std::abs(first.second - second.second) < kEpsilon;
      result.push_back({route, index, isHorizontal,
                        isHorizontal ? first.second : first.first,
                        isHorizontal ? std::min(first.first, second.first)
                                     : std::min(first.second, second.second),
                        isHorizontal ? std::max(first.first, second.first)
                                     : std::max(first.second, second.second)});
    }
    return result;
  };
  auto related = [&](unsigned first, unsigned second) {
    const ExplicitRoute &a = routes[first];
    const ExplicitRoute &b = routes[second];
    return a.flowGroup == b.flowGroup || a.from == b.from || a.from == b.to ||
           a.to == b.from || a.to == b.to;
  };
  auto conflicts = [&](const Segment &first, const Segment &second) {
    return first.horizontal == second.horizontal &&
           !related(first.route, second.route) &&
           std::min(first.high, second.high) - std::max(first.low, second.low) >
               kEpsilon &&
           std::abs(first.cross - second.cross) < minimumGap - kEpsilon;
  };
  auto crosses = [](const Segment &first, const Segment &second) {
    if (first.horizontal == second.horizontal)
      return false;
    const Segment &horizontal = first.horizontal ? first : second;
    const Segment &vertical = first.horizontal ? second : first;
    return horizontal.low + kEpsilon < vertical.cross &&
           vertical.cross < horizontal.high - kEpsilon &&
           vertical.low + kEpsilon < horizontal.cross &&
           horizontal.cross < vertical.high - kEpsilon;
  };
  auto blocked = [&](Point first, Point second, const ExplicitRoute &route) {
    for (const InflatedObstacle &obstacle : scene) {
      if (obstacle.node == route.from || obstacle.node == route.to)
        continue;
      if (obstacle.kind == ObstacleKind::Tile && obstacle.tile &&
          (*obstacle.tile == route.fromTile || *obstacle.tile == route.toTile))
        continue;
      const Rectangle &box = obstacle.blocked;
      if (std::abs(first.second - second.second) < kEpsilon) {
        double low = std::min(first.first, second.first);
        double high = std::max(first.first, second.first);
        if (box.bottom + kEpsilon < first.second &&
            first.second < box.top - kEpsilon && low < box.right - kEpsilon &&
            high > box.left + kEpsilon)
          return true;
      } else {
        double low = std::min(first.second, second.second);
        double high = std::max(first.second, second.second);
        if (box.left + kEpsilon < first.first &&
            first.first < box.right - kEpsilon && low < box.top - kEpsilon &&
            high > box.bottom + kEpsilon)
          return true;
      }
    }
    return false;
  };
  auto countConflicts = [&](ArrayRef<Segment> routeSegments,
                            ArrayRef<Segment> all) {
    unsigned count = 0;
    for (const Segment &first : routeSegments)
      for (const Segment &second : all)
        if (second.route != first.route && conflicts(first, second))
          ++count;
    return count;
  };
  auto crossingCount = [&](ArrayRef<Segment> routeSegments,
                           ArrayRef<Segment> all) {
    unsigned count = 0;
    for (const Segment &first : routeSegments)
      for (const Segment &second : all)
        if (second.route != first.route &&
            !related(first.route, second.route) && crosses(first, second))
          ++count;
    return count;
  };

  std::set<std::tuple<unsigned, unsigned, unsigned, unsigned>> rejected;
  unsigned movedCount = 0;
  while (movedCount < 64 && rejected.size() < 64) {
    SmallVector<Segment> before = segments();
    std::optional<std::pair<unsigned, unsigned>> conflictIndices;
    for (bool horizontal : {false, true}) {
      SmallVector<unsigned> ordered;
      for (unsigned index = 0; index < before.size(); ++index)
        if (before[index].horizontal == horizontal)
          ordered.push_back(index);
      llvm::sort(ordered, [&](unsigned first, unsigned second) {
        return std::pair{before[first].cross, first} <
               std::pair{before[second].cross, second};
      });
      for (unsigned first = 0; first < ordered.size(); ++first) {
        for (unsigned second = first + 1; second < ordered.size(); ++second) {
          unsigned firstIndex = ordered[first];
          unsigned secondIndex = ordered[second];
          if (before[secondIndex].cross - before[firstIndex].cross >=
              minimumGap - kEpsilon)
            break;
          auto [lowIndex, highIndex] = std::minmax(firstIndex, secondIndex);
          if (conflicts(before[lowIndex], before[highIndex]) &&
              !rejected.count({before[lowIndex].route, before[lowIndex].index,
                               before[highIndex].route,
                               before[highIndex].index}) &&
              (!conflictIndices ||
               std::pair{lowIndex, highIndex} < *conflictIndices))
            conflictIndices = {lowIndex, highIndex};
        }
      }
    }
    if (!conflictIndices)
      break;
    std::pair<Segment, Segment> conflict{before[conflictIndices->first],
                                         before[conflictIndices->second]};

    bool moved = false;
    for (auto [selected, fixed] :
         {std::pair{conflict.second, conflict.first},
          std::pair{conflict.first, conflict.second}}) {
      ExplicitRoute &route = routes[selected.route];
      SmallVector<Point> original = route.corners;
      SmallVector<Segment> originalSegments = segmentsOf(selected.route);
      unsigned originalConflicts = countConflicts(originalSegments, before);
      unsigned originalCrossings = crossingCount(originalSegments, before);
      unsigned originalBlocked = 0;
      for (auto pair : llvm::zip_equal(ArrayRef(original).drop_back(),
                                       ArrayRef(original).drop_front()))
        originalBlocked += blocked(std::get<0>(pair), std::get<1>(pair), route);
      double laneGap = minimumGap * 1.01;
      double overlapLow = selected.low;
      double overlapHigh = selected.high;
      SmallVector<double> lanes;
      for (unsigned distance = 1; distance <= 2; ++distance)
        for (double center : {selected.cross, fixed.cross})
          for (double direction : {-1.0, 1.0})
            lanes.push_back(center + direction * distance * laneGap);
      for (const Segment &nearby : before) {
        if (nearby.horizontal != selected.horizontal ||
            std::min(nearby.high, selected.high) <=
                std::max(nearby.low, selected.low) + kEpsilon)
          continue;
        lanes.push_back(nearby.cross - laneGap);
        lanes.push_back(nearby.cross + laneGap);
      }
      llvm::sort(lanes, [&](double first, double second) {
        return std::pair{std::abs(first - selected.cross), first} <
               std::pair{std::abs(second - selected.cross), second};
      });
      lanes.erase(llvm::unique(lanes,
                               [](double first, double second) {
                                 return std::abs(first - second) < kEpsilon;
                               }),
                  lanes.end());
      unsigned evaluatedCandidates = 0;
      for (double lane : lanes) {
        if (selected.index > 0 && selected.index + 2 < original.size()) {
          SmallVector<Point> candidate = original;
          Point &first = candidate[selected.index];
          Point &second = candidate[selected.index + 1];
          if (selected.horizontal)
            first.second = second.second = lane;
          else
            first.first = second.first = lane;
          unsigned blockedCount = 0;
          for (auto pair : llvm::zip_equal(ArrayRef(candidate).drop_back(),
                                           ArrayRef(candidate).drop_front()))
            blockedCount +=
                blocked(std::get<0>(pair), std::get<1>(pair), route);
          if (blockedCount <= originalBlocked) {
            route.corners = std::move(candidate);
            SmallVector<Segment> candidateSegments = segmentsOf(selected.route);
            if (countConflicts(candidateSegments, before) < originalConflicts &&
                crossingCount(candidateSegments, before) <=
                    originalCrossings + 1) {
              moved = true;
              ++movedCount;
              break;
            }
            route.corners = original;
          }
        }
        for (auto [spanLow, spanHigh] : {std::pair{overlapLow, overlapHigh}}) {
          bool occupied = llvm::any_of(before, [&](const Segment &nearby) {
            return nearby.route != selected.route &&
                   nearby.horizontal == selected.horizontal &&
                   !related(nearby.route, selected.route) &&
                   std::min(nearby.high, spanHigh) >
                       std::max(nearby.low, spanLow) + kEpsilon &&
                   std::abs(nearby.cross - lane) < minimumGap - kEpsilon;
          });
          if (occupied)
            continue;
          if (evaluatedCandidates++ >= 64)
            break;
          Point first = original[selected.index];
          Point second = original[selected.index + 1];
          bool forward = selected.horizontal ? first.first < second.first
                                             : first.second < second.second;
          double start = forward ? spanLow : spanHigh;
          double end = forward ? spanHigh : spanLow;
          Point doglegStart = selected.horizontal ? Point{start, first.second}
                                                  : Point{first.first, start};
          Point shiftedFirst =
              selected.horizontal ? Point{start, lane} : Point{lane, start};
          Point shiftedSecond =
              selected.horizontal ? Point{end, lane} : Point{lane, end};
          Point doglegEnd = selected.horizontal ? Point{end, second.second}
                                                : Point{second.first, end};
          SmallVector<Point> candidate;
          candidate.append(original.begin(), original.begin() + selected.index);
          candidate.push_back(first);
          candidate.push_back(doglegStart);
          candidate.push_back(shiftedFirst);
          candidate.push_back(shiftedSecond);
          candidate.push_back(doglegEnd);
          candidate.push_back(second);
          candidate.append(original.begin() + selected.index + 2,
                           original.end());
          unsigned blockedCount = 0;
          for (auto pair : llvm::zip_equal(ArrayRef(candidate).drop_back(),
                                           ArrayRef(candidate).drop_front()))
            blockedCount +=
                blocked(std::get<0>(pair), std::get<1>(pair), route);
          if (blockedCount > originalBlocked)
            continue;
          route.corners = std::move(candidate);
          SmallVector<Segment> candidateSegments = segmentsOf(selected.route);
          unsigned candidateConflicts =
              countConflicts(candidateSegments, before);
          unsigned candidateCrossings =
              crossingCount(candidateSegments, before);
          if (candidateConflicts < originalConflicts &&
              candidateCrossings <= originalCrossings + 1) {
            moved = true;
            ++movedCount;
            break;
          }
          route.corners = original;
        }
        if (moved)
          break;
      }
      if (moved)
        break;
    }
    if (!moved)
      rejected.insert({conflict.first.route, conflict.first.index,
                       conflict.second.route, conflict.second.index});
  }
}

SmallVector<ArrowPath>
AvoidanceArrowRenderer::route(ArrayRef<ArrowRequest> requests,
                              ArrayRef<RoutingObstacle> obstacles,
                              const LayoutGeometry &layout) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  routeCalls = 0;
  expandedStates = 0;
  buildGrid(requests, obstacles, layout);
  auto gridBuilt = Clock::now();
  std::vector<unsigned> order(requests.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](unsigned first, unsigned second) {
    const ArrowRequest &a = requests[first];
    const ArrowRequest &b = requests[second];
    double aLength = std::abs(a.from.pinned.first - a.to.pinned.first) +
                     std::abs(a.from.pinned.second - a.to.pinned.second);
    double bLength = std::abs(b.from.pinned.first - b.to.pinned.first) +
                     std::abs(b.from.pinned.second - b.to.pinned.second);
    return std::tie(aLength, a.id) < std::tie(bLength, b.id);
  });

  std::vector<RoutedPath> routed(requests.size());
  for (unsigned requestID : order) {
    ++routeCalls;
    routed[requestID] = routeOne(requests[requestID]);
    addUsage(routed[requestID], 1);
  }
  for (unsigned requestID : order) {
    if (routed[requestID].valid)
      continue;
    ++routeCalls;
    routed[requestID] = routeOne(requests[requestID], false, true);
    addUsage(routed[requestID], 1);
  }
  for (unsigned requestID : order) {
    if (routed[requestID].valid)
      continue;
    ++routeCalls;
    routed[requestID] = routeOne(requests[requestID], true, true);
    addUsage(routed[requestID], 1);
  }
  auto initiallyRouted = Clock::now();

  for (unsigned pass = 0; pass < 2; ++pass) {
    llvm::stable_sort(order, [&](unsigned first, unsigned second) {
      return std::pair{-conflictScore(routed[first]), first} <
             std::pair{-conflictScore(routed[second]), second};
    });
    size_t rerouteLimit = std::min<size_t>(384, order.size());
    for (unsigned requestID : ArrayRef(order).take_front(rerouteLimit)) {
      if (conflictScore(routed[requestID]) <= 0)
        break;
      RoutedPath before = std::move(routed[requestID]);
      addUsage(before, -1);
      ++routeCalls;
      RoutedPath after = routeOne(requests[requestID]);
      routed[requestID] = after.valid ? std::move(after) : std::move(before);
      addUsage(routed[requestID], 1);
    }
  }
  auto conflictsRouted = Clock::now();

  if (::getenv("AIE_VISUALIZE_GRAPH_STATS")) {
    auto milliseconds = [](Clock::duration duration) {
      return std::chrono::duration<double, std::milli>(duration).count();
    };
    llvm::errs() << "avoidance requests=" << requests.size()
                 << " obstacles=" << obstacles.size() << " grid=" << xs.size()
                 << 'x' << ys.size() << " states=" << routeDistance.size()
                 << " route_calls=" << routeCalls
                 << " expanded=" << expandedStates
                 << " grid_ms=" << milliseconds(gridBuilt - start)
                 << " initial_ms=" << milliseconds(initiallyRouted - gridBuilt)
                 << " conflict_ms="
                 << milliseconds(conflictsRouted - initiallyRouted) << '\n';
  }

  SmallVector<ExplicitRoute> explicitRoutes;
  for (unsigned requestID = 0; requestID < routed.size(); ++requestID) {
    const RoutedPath &path = routed[requestID];
    if (!path.valid)
      continue;
    const ArrowRequest &request = requests[requestID];
    explicitRoutes.push_back({path.id, request.flowGroup, request.from.node,
                              request.to.node, request.from.tile,
                              request.to.tile, cornersOf(path)});
  }
  deconflict(explicitRoutes);

  SmallVector<ArrowPath> paths;
  for (ExplicitRoute &route : explicitRoutes) {
    SmallVector<Point> bezier = toBezier(route.corners, kCornerRadius);
    if (!bezier.empty())
      paths.push_back({route.id, std::move(bezier)});
  }
  return paths;
}

} // namespace

std::unique_ptr<ArrowRenderer> createAvoidanceArrowRenderer() {
  return std::make_unique<AvoidanceArrowRenderer>();
}

} // namespace xilinx::AIE
