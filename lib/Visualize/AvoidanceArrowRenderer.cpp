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

#include <algorithm>
#include <array>
#include <cmath>
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
constexpr double kCornerRadius = 0.05;
constexpr double kEpsilon = 1e-9;
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
  void buildGrid(ArrayRef<ArrowRequest> requests,
                 ArrayRef<RoutingObstacle> obstacles,
                 const LayoutGeometry &layout);
  SmallVector<Side> sidesFor(const ArrowEnd &end, const ArrowEnd &other) const;
  SmallVector<Gateway> gatewaysFor(const ArrowEnd &end,
                                   const ArrowEnd &other) const;
  RoutedPath routeOne(const ArrowRequest &request);
  void addUsage(const RoutedPath &path, int amount);
  double conflictScore(const RoutedPath &path) const;
  SmallVector<Point> cornersOf(const RoutedPath &path) const;

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
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<InflatedObstacle> scene;
  std::map<std::string, unsigned> obstacleByNode;
  std::vector<Blockage> vertexBlockage;
  std::vector<Blockage> horizontalBlockage;
  std::vector<Blockage> verticalBlockage;
  std::vector<uint16_t> horizontalUsage;
  std::vector<uint16_t> verticalUsage;
  std::vector<uint16_t> vertexHorizontalUsage;
  std::vector<uint16_t> vertexVerticalUsage;
  std::vector<double> routeDistance;
  std::vector<uint64_t> routeParent;
  std::vector<uint32_t> routeGeneration;
  uint32_t generation = 0;
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
AvoidanceArrowRenderer::gatewaysFor(const ArrowEnd &end,
                                    const ArrowEnd &other) const {
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
    switch (side) {
    case Side::East:
      gateway.anchor = {obstacle.visible.right,
                        (obstacle.visible.bottom + obstacle.visible.top) / 2};
      gateway.point = {obstacle.blocked.right, gateway.anchor.second};
      break;
    case Side::West:
      gateway.anchor = {obstacle.visible.left,
                        (obstacle.visible.bottom + obstacle.visible.top) / 2};
      gateway.point = {obstacle.blocked.left, gateway.anchor.second};
      break;
    case Side::North:
      gateway.anchor = {(obstacle.visible.left + obstacle.visible.right) / 2,
                        obstacle.visible.top};
      gateway.point = {gateway.anchor.first, obstacle.blocked.top};
      break;
    case Side::South:
      gateway.anchor = {(obstacle.visible.left + obstacle.visible.right) / 2,
                        obstacle.visible.bottom};
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
      double left = std::min(gateway.anchor.first, gateway.point.first);
      double right = std::max(gateway.anchor.first, gateway.point.first);
      double bottom = std::min(gateway.anchor.second, gateway.point.second);
      double top = std::max(gateway.anchor.second, gateway.point.second);
      return left < candidate.blocked.right - kEpsilon &&
             right > candidate.blocked.left + kEpsilon &&
             bottom < candidate.blocked.top - kEpsilon &&
             top > candidate.blocked.bottom + kEpsilon;
    });
    if (blocked)
      continue;
    gateway.vertex = vertex(*x, *y);
    gateways.push_back(gateway);
  }
  return gateways;
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
  overlapPenalty = 4.0;
  proximityPenalty = 0.3;

  scene.clear();
  obstacleByNode.clear();
  xs.clear();
  ys.clear();
  std::set<std::string> endpointNodes;
  for (const ArrowRequest &request : requests) {
    endpointNodes.insert(request.from.node);
    endpointNodes.insert(request.to.node);
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
    if (endpointNodes.count(obstacle.node)) {
      xs.push_back((obstacle.bounds.left + obstacle.bounds.right) / 2);
      ys.push_back((obstacle.bounds.bottom + obstacle.bounds.top) / 2);
    }
    if (obstacle.kind == ObstacleKind::Tile) {
      for (double offset : {layout.text(0.12), layout.text(0.24)}) {
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
  vertexHorizontalUsage.assign(vertexBlockage.size(), 0);
  vertexVerticalUsage.assign(vertexBlockage.size(), 0);
  routeDistance.resize(vertexBlockage.size() * 4);
  routeParent.resize(vertexBlockage.size() * 4);
  routeGeneration.assign(vertexBlockage.size() * 4, 0);
  generation = 0;
}

RoutedPath AvoidanceArrowRenderer::routeOne(const ArrowRequest &request) {
  SmallVector<Gateway> sources = gatewaysFor(request.from, request.to);
  SmallVector<Gateway> targets = gatewaysFor(request.to, request.from);
  RoutedPath result;
  result.id = request.id;
  if (sources.empty() || targets.empty())
    return result;

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

void AvoidanceArrowRenderer::addUsage(const RoutedPath &path, int amount) {
  if (!path.valid || path.vertices.size() < 2)
    return;
  for (auto pair : llvm::zip_equal(ArrayRef(path.vertices).drop_back(),
                                   ArrayRef(path.vertices).drop_front())) {
    auto [from, to] = pair;
    auto [fromX, fromY] = coordinates(from);
    auto [toX, toY] = coordinates(to);
    if (fromY == toY) {
      unsigned edge = horizontalEdge(std::min(fromX, toX), fromY);
      horizontalUsage[edge] += amount;
      vertexHorizontalUsage[from] += amount;
      vertexHorizontalUsage[to] += amount;
    } else {
      unsigned edge = verticalEdge(fromX, std::min(fromY, toY));
      verticalUsage[edge] += amount;
      vertexVerticalUsage[from] += amount;
      vertexVerticalUsage[to] += amount;
    }
  }
}

double AvoidanceArrowRenderer::conflictScore(const RoutedPath &path) const {
  if (!path.valid || path.vertices.size() < 2)
    return 0.0;
  double score = 0.0;
  for (auto pair : llvm::zip_equal(ArrayRef(path.vertices).drop_back(),
                                   ArrayRef(path.vertices).drop_front())) {
    auto [from, to] = pair;
    auto [fromX, fromY] = coordinates(from);
    auto [toX, toY] = coordinates(to);
    if (fromY == toY) {
      score += horizontalUsage[horizontalEdge(std::min(fromX, toX), fromY)] - 1;
      score += vertexVerticalUsage[to];
    } else {
      score += verticalUsage[verticalEdge(fromX, std::min(fromY, toY))] - 1;
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

SmallVector<ArrowPath>
AvoidanceArrowRenderer::route(ArrayRef<ArrowRequest> requests,
                              ArrayRef<RoutingObstacle> obstacles,
                              const LayoutGeometry &layout) {
  buildGrid(requests, obstacles, layout);
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
    routed[requestID] = routeOne(requests[requestID]);
    addUsage(routed[requestID], 1);
  }

  for (unsigned pass = 0; pass < 2; ++pass) {
    llvm::stable_sort(order, [&](unsigned first, unsigned second) {
      return std::pair{-conflictScore(routed[first]), first} <
             std::pair{-conflictScore(routed[second]), second};
    });
    size_t rerouteLimit = std::min<size_t>(128, order.size());
    for (unsigned requestID : ArrayRef(order).take_front(rerouteLimit)) {
      if (conflictScore(routed[requestID]) <= 0)
        break;
      RoutedPath before = std::move(routed[requestID]);
      addUsage(before, -1);
      RoutedPath after = routeOne(requests[requestID]);
      if (!after.valid) {
        routed[requestID] = std::move(before);
      } else {
        routed[requestID] = std::move(after);
      }
      addUsage(routed[requestID], 1);
    }
  }

  SmallVector<ArrowPath> paths;
  for (const RoutedPath &path : routed) {
    if (!path.valid)
      continue;
    SmallVector<Point> corners = cornersOf(path);
    SmallVector<Point> bezier = toBezier(corners, kCornerRadius);
    if (!bezier.empty())
      paths.push_back({path.id, std::move(bezier)});
  }
  return paths;
}

} // namespace

std::unique_ptr<ArrowRenderer> createAvoidanceArrowRenderer() {
  return std::make_unique<AvoidanceArrowRenderer>();
}

} // namespace xilinx::AIE
