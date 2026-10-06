//===- AIECreatePathfindFlows.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"
#include "aie/Dialect/AIE/Transforms/AIERoutingDiagnostics.h"
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-create-pathfinder-flows"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEROUTEPATHFINDERFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using PhysPort = std::pair<TileID, Port>;

/// (mask, value, amsel) by slave port, in slot order.
using OverlayRules =
    std::map<PhysPort, SmallVector<std::tuple<int, int, int>, 2>>;

/// Whether a control-packet reload configures `d`: it installs
/// @ctrl_pkt_overlay first, and control packets through the overlay configure
/// the rest, so the overlay must keep the switch settings it takes alone.
static bool reloadsOverlay(DeviceOp d) {
  auto reload = d->getAttrOfType<BoolAttr>("has_ctrl_pkt_overlay");
  return reload && reload.getValue();
}

/// Whether `flow` stays out of the control overlay of a design a control-packet
/// reload configures. The reload installs @ctrl_pkt_overlay, which holds the
/// flows to and from TileControl ports alone, and configures the design's
/// other flows, prioritized or not, with the rest of it (#3837).
static bool reloadConfigures(PacketFlowOp flow) {
  auto control = [](auto end) {
    return end.getBundle() == WireBundle::TileControl;
  };
  return reloadsOverlay(flow->getParentOfType<DeviceOp>()) &&
         llvm::none_of(flow.getPorts().getOps<PacketSourceOp>(), control) &&
         llvm::none_of(flow.getPorts().getOps<PacketDestOp>(), control);
}

/// What the prioritized flows (the control overlay) take when routed alone,
/// which a reload keeps.
struct PinnedOverlay {
  PacketTrees trees;
  OverlayRules rules;
};

static OverlayRules overlayRules(DeviceOp d) {
  const int numArbiters = d.getTargetModel().getNumArbiters();
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(d.getContext());
  OverlayRules rules;
  for (SwitchboxOp box : d.getOps<SwitchboxOp>())
    for (PacketRulesOp port : box.getOps<PacketRulesOp>())
      for (PacketRuleOp rule : port.getRules().front().getOps<PacketRuleOp>())
        if (overlay.isAttrPresent(port) || overlay.isAttrPresent(rule)) {
          auto amsel = cast<AMSelOp>(rule.getAmsel().getDefiningOp());
          rules[{box.getTileOp().getTileID(),
                 {port.getSourceBundle(), port.sourceIndex()}}]
              .push_back(
                  {rule.maskInt(), rule.valueInt(),
                   amsel.arbiterIndex() + amsel.getMselValue() * numArbiters});
        }
  return rules;
}

// The backtracking searches for an arbiter plan and for a set of streams that
// cannot share an arbiter are exact up to this many steps.
constexpr int maxSearchSteps = 100000;
// The hold-cycle search replans a tile at most this many times.
constexpr int maxHoldCycleReplans = 256;

static constexpr llvm::StringLiteral sharedReceiverCycleReason =
    "Packet flows into receivers they share can deadlock holding arbiters "
    "across switchboxes, and no routing found avoids it: ";
static constexpr llvm::StringLiteral allowDeadlockProneHint =
    " Set allow-deadlock-prone (aiecc --allow-deadlock-prone-routing) to "
    "route them anyway.";

namespace {
struct PacketPlan;

/// \brief Routes flows in a device by lowering them to stream-switch
/// configurations.
///
/// Overall flow:
/// 1. Report flows that can deadlock however they are routed.
/// 2. Route the prioritized packet flows alone and pin their trees.
/// 3. Route every flow with Pathfinder, planning the arbiters and packet rules
///    on each routing found and steering off those that cannot be used,
///    relaxing the packet constraints if no routing is found.
/// 4. Lower the circuit flows to connects, and the packet flows as planned.
/// 5. Add the wires between switchboxes and tiles.
struct AIEPathfinderPass
    : xilinx::AIE::impl::AIERoutePathfinderFlowsBase<AIEPathfinderPass> {
  using Base =
      xilinx::AIE::impl::AIERoutePathfinderFlowsBase<AIEPathfinderPass>;
  AIEPathfinderPass() = default;
  AIEPathfinderPass(const AIERoutePathfinderFlowsOptions &options)
      : Base(options) {}
  void runOnOperation() override;
  LogicalResult runOnFlow(DeviceOp d, DynamicTileAnalysis &analyzer);
  /// Lowers the packet flows as `plan` says, on the routing `analyzer` found,
  /// warning of a hold cycle left through receivers flows share if `warn`.
  LogicalResult runOnPacketFlow(DeviceOp d, OpBuilder &builder,
                                DynamicTileAnalysis &analyzer,
                                const StreamConflicts &conflicts,
                                const PacketPlan &plan, bool warn);
  /// Routes the flows in `d`, planning the arbiters on each routing found,
  /// with the overlay in `pinned` kept as it is, and other prioritized flows
  /// routed first if `prioritize`. Returns the plan for the packet flows on
  /// the routing found.
  llvm::Expected<PacketPlan> route(DeviceOp d, DynamicTileAnalysis &analyzer,
                                   const StreamConflicts &conflicts,
                                   const PinnedOverlay &pinned,
                                   bool circuitSwitchHops,
                                   bool prioritize = true);
};

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {

    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    ConnectOp::create(rewriter, flowOp.getLoc(), inBundle, inIndex, outBundle,
                      outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  mlir::LogicalResult
  matchAndRewrite(FlowOp flowOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();
    DeviceOp d = flowOp->getParentOfType<DeviceOp>();
    if (!d) {
      flowOp->emitOpError("This operation must be contained within a device");
      return failure();
    }
    rewriter.setInsertionPoint(d.getBody()->getTerminator());

    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

    LLVM_DEBUG({
      auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
      llvm::dbgs() << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
                   << ", " << srcCoords.row << ")"
                   << stringifyWireBundle(srcBundle) << srcChannel << " -> ("
                   << dstTile.colIndex() << ", " << dstTile.rowIndex() << ")"
                   << stringifyWireBundle(flowOp.getDestBundle())
                   << flowOp.getDestChannel() << "\n\t";
    });

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    TileID srcSbId = {srcCoords.col, srcCoords.row};
    PathEndPoint srcPoint = {srcSbId, srcPort};
    if (analyzer.processedFlows.count(srcPoint)) {
      // This FlowOp is a broadcast sibling of a flow whose route was already
      // materialized (the analyzer merges all destinations sharing a source
      // into one net, so the first sibling emitted connections for every
      // destination). Erase it and report success so the erase is committed.
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");
      rewriter.eraseOp(Op);
      return success();
    }
    const SwitchSettings &settings = analyzer.routing.settings.at(srcPoint);
    // add connections for all the Switchboxes in SwitchSettings
    for (const auto &[tileId, setting] : settings) {
      int col = tileId.col;
      int row = tileId.row;
      SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, col, row);
      int shimCh = srcChannel;
      bool isShim = analyzer.getTile(rewriter, tileId).isShimNOCorPLTile();

      // TODO: must reserve N3, N7, S2, S3 for DMA connections
      if (isShim && tileId == srcSbId) {

        shimCh = shimMuxChannelFrom(srcPort);
        ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, col);
        addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                      flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
      }
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];

        // A flow can start and end at one shim (see shimMuxChannelFrom).
        if (isShim && tileId == srcSbId && src == srcPort)
          src = {WireBundle::South, shimCh};
        if (isShim && (dest.bundle == WireBundle::DMA ||
                       dest.bundle == WireBundle::PLIO ||
                       dest.bundle == WireBundle::NOC)) {
          int destCh = shimMuxChannelTo(dest);

          ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, col);
          addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                        flowOp, WireBundle::North, destCh, dest.bundle,
                        dest.channel);
          dest = {WireBundle::South, destCh};
        }
        addConnection(rewriter, cast<Interconnect>(swOp.getOperation()), flowOp,
                      src.bundle, src.channel, dest.bundle, dest.channel);
      }

      LLVM_DEBUG(llvm::dbgs() << tileId << ": " << setting << " | "
                              << "\n");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");

    analyzer.processedFlows.insert(srcPoint);
    rewriter.eraseOp(Op);
    return success();
  }
};

} // namespace

LogicalResult AIEPathfinderPass::runOnFlow(DeviceOp d,
                                           DynamicTileAnalysis &analyzer) {
  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer);
  if (failed(applyPartialConversion(d, target, std::move(patterns))))
    return failure();
  return success();
}

/// The switchbox a master port feeds, and the slave port it feeds there.
static std::optional<std::pair<TileID, Port>> linkedInput(TileID tile,
                                                          Port out) {
  switch (out.bundle) {
  case WireBundle::East:
    return std::pair{TileID{tile.col + 1, tile.row},
                     Port{WireBundle::West, out.channel}};
  case WireBundle::West:
    return std::pair{TileID{tile.col - 1, tile.row},
                     Port{WireBundle::East, out.channel}};
  case WireBundle::North:
    return std::pair{TileID{tile.col, tile.row + 1},
                     Port{WireBundle::South, out.channel}};
  case WireBundle::South:
    return std::pair{TileID{tile.col, tile.row - 1},
                     Port{WireBundle::North, out.channel}};
  default:
    return std::nullopt;
  }
}

/// Whether the connection out of `out` at `tile` leads, through `settings`, to
/// `finalPort` at `finalTile`.
static bool findPathToDest(const SwitchSettings &settings, TileID tile,
                           Port out, TileID finalTile, Port finalPort) {
  if (tile == finalTile && out == finalPort)
    return true;
  std::optional<std::pair<TileID, Port>> next = linkedInput(tile, out);
  if (!next)
    return false;
  TileID neighbour = next->first;
  Port in = next->second;
  auto setting = settings.find(neighbour);
  if (setting == settings.end())
    return false;
  return llvm::any_of(
      llvm::zip(setting->second.srcs, setting->second.dsts), [&](auto link) {
        auto [src, dest] = link;
        return src == in &&
               findPathToDest(settings, neighbour, dest, finalTile, finalPort);
      });
}

/// Every node `successors` reaches from `start`, with the node a breadth-first
/// search first reaches it from; `start` maps to itself. Stops once `goal` is
/// reached, if given.
template <typename Node, typename Successors>
static std::map<Node, Node>
breadthFirstParents(Node start, Successors successors,
                    std::optional<Node> goal = std::nullopt) {
  std::map<Node, Node> parent{{start, start}};
  std::deque<Node> work{start};
  while (!work.empty() && !(goal && parent.count(*goal))) {
    Node u = work.front();
    work.pop_front();
    for (Node v : successors(u))
      if (parent.try_emplace(v, u).second)
        work.push_back(v);
  }
  return parent;
}

namespace {
struct CoverCube {
  int mask;
  int value;
  // cov[i] is set iff this cube matches match id i.
  llvm::SmallBitVector cov;
};
} // namespace

// Minimum set cover by branch-and-bound: find the fewest `cubes` whose coverage
// (cube.cov) unions to every match id still set in `uncovered`. `sel` is the
// current partial pick; `bestSel`/`bestSize` hold the smallest full cover so
// far and are updated in place.
static void bnbMinCover(ArrayRef<CoverCube> cubes,
                        const llvm::SmallBitVector &uncovered,
                        SmallVectorImpl<int> &sel, int &bestSize,
                        SmallVectorImpl<int> &bestSel) {

  // base case
  if (uncovered.none()) {
    if (static_cast<int>(sel.size()) < bestSize) {
      bestSize = static_cast<int>(sel.size());
      bestSel.assign(sel.begin(), sel.end());
    }
    return;
  }

  // bound: adding a cube would only tie the best
  if (static_cast<int>(sel.size()) + 1 >= bestSize) {
    return;
  }

  // most constrained variable: find the least-covered, uncovered matchId
  int pick = -1;
  int pickCount = static_cast<int>(cubes.size()) + 1;

  for (int id = uncovered.find_first(); id != -1;
       id = uncovered.find_next(id)) {
    int c = 0;

    for (const CoverCube &cb : cubes) {
      if (cb.cov.test(id)) {
        c++;
      }
    }

    if (c < pickCount) {
      pickCount = c;
      pick = id;
    }
  }

  // branch: try each cube that covers the chosen id, recursing on the rest.
  for (int i = 0; i < static_cast<int>(cubes.size()); ++i) {
    if (!cubes[i].cov.test(pick)) {
      continue;
    }

    sel.push_back(i);
    llvm::SmallBitVector next = uncovered;
    next.reset(cubes[i].cov); // next &= ~cov
    bnbMinCover(cubes, next, sel, bestSize, bestSel);
    sel.pop_back();
  }
}

static bool cubesIntersect(std::pair<int, int> a, std::pair<int, int> b) {
  return ((a.second ^ b.second) & a.first & b.first) == 0;
}

// Cover `matchIds` (a group's packet ids) with the fewest (mask, value) rules
// that match every specified id and no cube in `avoidCubes` (what the other
// groups on the same slave port claim). A rule matches id x iff
// (x & mask) == value; ids no cube claims are don't-cares, free to over-claim.
//
// Based on Quine-McCluskey:
//   1. Enumerate prime implicants: maximal cubes (a cube is one (mask, value))
//      that match >=1 matchIds and no avoid cube.
//   2. Pick the fewest of those cubes that cover all match ids (bnbMinCover).
//   3. Tighten each chosen cube to the smallest enclosing cube of the ids it
//      took, so it claims no more don't-cares than necessary; the one-cube case
//      reduces to the old common-bits mask.
static SmallVector<std::pair<int, int>>
computeSubcubeCover(const SmallVector<int, 4> &matchIds,
                    ArrayRef<std::pair<int, int>> avoidCubes, int idBits) {

  // id space and full mask, sized from the target's packet-id width.
  const int numIds = 1 << idBits;
  const int idMask = numIds - 1;

  auto hitsAvoid = [&](int mask, int value) {
    return llvm::any_of(avoidCubes, [&](std::pair<int, int> c) {
      return cubesIntersect({mask, value}, c);
    });
  };

  llvm::SmallBitVector matchMask(numIds);
  for (int id : matchIds) {
    matchMask.set(id);
  }

  // phase 1: enumerate prime implicants. small (3^idBits), so brute force.
  SmallVector<CoverCube> primes;
  for (int mask = 0; mask < numIds; ++mask) {
    for (int value = 0; value < numIds; ++value) {
      // valid cube := value covered by mask, avoids properly
      if ((value & mask) != value) {
        continue;
      }
      if (hitsAvoid(mask, value)) {
        continue;
      }

      // record covering
      llvm::SmallBitVector cov(numIds);
      for (int id : matchIds) {
        if ((id & mask) == value) {
          cov.set(id);
        }
      }

      if (cov.none()) {
        continue;
      }

      // prime = maximal, i.e., no checked mask bit can be dropped without
      // covering an avoidId.
      bool isPrime = true;
      for (int b = 0; b < idBits; ++b) {
        if (!((mask >> b) & 1)) {
          continue;
        }

        if (!hitsAvoid(mask & ~(1 << b), value & ~(1 << b))) {
          isPrime = false;
          break;
        }
      }

      if (isPrime) {
        primes.push_back({mask, value, std::move(cov)});
      }
    }
  }

  // phase 2: pick the fewest primes that cover every match id (bnb).
  SmallVector<int> sel, bestSel;
  int bestSize = static_cast<int>(matchIds.size()) + 1;

  bnbMinCover(primes, matchMask, sel, bestSize, bestSel);

  SmallVector<std::pair<int, int>> chosen;
  for (int i : bestSel) {
    chosen.push_back({primes[i].mask, primes[i].value});
  }

  if (chosen.empty()) {
    // unreachable: a single-id cube always covers
    for (int id : matchIds) {
      chosen.push_back({idMask, id});
    }
  }

  // phase 3: tighten each chosen cube to match only the ids it took.
  // maximal -> over-claim don't-cares -> more mask bits narrow the cube.
  //
  // e.g., {2,4} avoid {0}:
  //   prime (mask 00010, value 00010): 2 & 00010 = 00010 == value -> hit
  //                                    6 & 00010 = 00010 == value -> also hit
  //   tight (mask 11111, value 00010): 2 & 11111 = 00010 == value -> hit
  //                                    6 & 11111 = 00110 != value -> dropped
  //                                    0 & 11111 = 00000 != value -> avoided

  SmallVector<SmallVector<int, 4>, 4> assigned(chosen.size());
  for (int id : matchIds) {
    for (int c = 0; c < static_cast<int>(chosen.size()); ++c) {
      if ((id & chosen[c].first) == chosen[c].second) {
        assigned[c].push_back(id);
        break;
      }
    }
  }

  SmallVector<std::pair<int, int>> cover;
  for (const SmallVector<int, 4> &ids : assigned) {
    if (ids.empty()) {
      continue;
    }

    int mask = idMask;
    for (int i = 0; i < idBits; ++i) {
      int bit = (ids.front() >> i) & 1;
      for (int id : ids) {
        if (((id >> i) & 1) != bit) {
          mask &= ~(1 << i);
          break;
        }
      }
    }
    cover.push_back({mask, ids.front() & mask});
  }

  return cover;
}

// A group's rules (see GroupClaims): the stated ones, then a cover of the
// derived ids that avoids them and `avoid`, whatever else claims ids on the
// port.
static SmallVector<std::pair<int, int>>
groupRules(ArrayRef<std::pair<int, int>> stated, ArrayRef<int> derived,
           ArrayRef<std::pair<int, int>> avoid, int idBits) {
  SmallVector<std::pair<int, int>> rules(stated);
  if (derived.empty())
    return rules;
  SmallVector<std::pair<int, int>> derivedAvoid(avoid);
  derivedAvoid.append(stated.begin(), stated.end());
  llvm::append_range(rules, computeSubcubeCover(SmallVector<int, 4>(derived),
                                                derivedAvoid, idBits));
  return rules;
}

namespace {
/// What a group of flows on one slave port claims: the rules its flows state,
/// and the ids they leave to the router; and whether they are the control
/// overlay's.
struct GroupClaims {
  ArrayRef<std::pair<int, int>> stated;
  ArrayRef<int> derived;
  bool overlay = false;
};

/// A packet rule of a slave port, and the group it selects.
struct PortRule {
  int mask;
  int value;
  size_t group;
};
} // namespace

static uint64_t cubeIds(int mask, int value, int idBits) {
  uint64_t ids = 0;
  for (int id = 0; id < (1 << idBits); ++id)
    if ((id & mask) == (value & mask))
      ids |= uint64_t(1) << id;
  return ids;
}

// The fewest rules, at most `budget`, that send every id a group claims to
// that group, where a packet takes the first rule it matches. A rule may then
// claim ids of another group that an earlier rule already takes.
static std::optional<SmallVector<PortRule>>
orderedRules(ArrayRef<GroupClaims> groups,
             ArrayRef<std::pair<int, int>> existing, int idBits,
             size_t budget) {
  if (idBits > 6)
    return std::nullopt;
  const int idMask = (1 << idBits) - 1;
  SmallVector<uint64_t> claims;
  for (const GroupClaims &group : groups) {
    uint64_t ids = 0;
    for (auto [mask, value] : group.stated)
      ids |= cubeIds(mask, value, idBits);
    for (int id : group.derived)
      ids |= uint64_t(1) << id;
    claims.push_back(ids);
  }
  uint64_t claimed = 0;
  for (uint64_t ids : claims)
    claimed |= ids;
  uint64_t taken = 0;
  for (auto [mask, value] : existing)
    taken |= cubeIds(mask, value, idBits);
  if (taken & claimed)
    return std::nullopt;
  SmallVector<uint64_t> cubes;
  for (int mask = 0; mask <= idMask; ++mask)
    for (int value = 0; value <= idMask; ++value)
      if ((value & mask) == value)
        cubes.push_back(cubeIds(mask, value, idBits));

  // A rule is worth only the claimed ids it newly takes, all of one group, so
  // the tightest cube around them serves, and fewer ids of the same group
  // never serve better.
  SmallVector<PortRule> rules;
  std::set<std::pair<uint64_t, size_t>> dead;
  auto search = [&](auto &self, uint64_t taken, size_t left) -> bool {
    uint64_t open = claimed & ~taken;
    size_t openGroups =
        llvm::count_if(claims, [&](uint64_t ids) { return ids & open; });
    if (openGroups == 0)
      return true;
    if (openGroups > left || dead.count({taken, left}))
      return false;
    SmallVector<std::pair<uint64_t, size_t>> moves;
    for (uint64_t cube : cubes)
      for (auto [g, ids] : llvm::enumerate(claims))
        if ((cube & open) && (cube & open & ~ids) == 0 &&
            !llvm::is_contained(moves, std::pair{cube & open, g}))
          moves.push_back({cube & open, g});
    llvm::stable_sort(moves, [](auto a, auto b) {
      return llvm::popcount(a.first) > llvm::popcount(b.first);
    });
    SmallVector<std::pair<uint64_t, size_t>> tried;
    for (auto [own, g] : moves) {
      if (llvm::any_of(tried, [&, own = own, g = g](auto t) {
            return t.second == g && (t.first & own) == own;
          }))
        continue;
      tried.push_back({own, g});
      int all = idMask, any = 0;
      for (int id = 0; id <= idMask; ++id)
        if ((own >> id) & 1) {
          all &= id;
          any |= id;
        }
      int mask = idMask & ~(all ^ any);
      rules.push_back({mask, all, g});
      if (self(self, taken | own, left - 1))
        return true;
      rules.pop_back();
    }
    dead.insert({taken, left});
    return false;
  };
  for (size_t length = 1; length <= budget; ++length)
    if (search(search, taken, length))
      return rules;
  return std::nullopt;
}

// The rules of a slave port, in slot order after the `existing` ones. Each
// group takes its own cover, which claims no id another group claims, unless
// those outnumber the free slots and first-match order fits them.
static SmallVector<PortRule> portRules(ArrayRef<GroupClaims> groups,
                                       ArrayRef<std::pair<int, int>> existing,
                                       int idBits, size_t slots) {
  const int idMask = (1 << idBits) - 1;
  SmallVector<PortRule> rules;
  for (auto [g, group] : llvm::enumerate(groups)) {
    SmallVector<std::pair<int, int>> avoid(existing);
    for (auto [o, other] : llvm::enumerate(groups)) {
      if (o == g)
        continue;
      avoid.append(other.stated.begin(), other.stated.end());
      for (int id : other.derived)
        avoid.push_back({idMask, id});
    }
    for (auto [mask, value] :
         groupRules(group.stated, group.derived, avoid, idBits))
      rules.push_back({mask, value, g});
  }
  if (existing.size() + rules.size() <= slots || existing.size() >= slots)
    return rules;
  if (std::optional<SmallVector<PortRule>> ordered =
          orderedRules(groups, existing, idBits, slots - existing.size()))
    return *ordered;
  return rules;
}

// The rules of a slave port, in slot order after the `existing` ones. The
// overlay's groups take the rules they take alone, in the first slots (see
// reloadsOverlay), and the other groups' rules follow.
static SmallVector<PortRule>
slavePortRules(ArrayRef<GroupClaims> groups,
               ArrayRef<std::pair<int, int>> existing, int idBits,
               size_t slots) {
  SmallVector<std::pair<int, int>> taken(existing);
  SmallVector<PortRule> rules;
  for (bool overlay : {true, false}) {
    SmallVector<size_t, 4> members;
    SmallVector<GroupClaims> claims;
    for (auto [g, group] : llvm::enumerate(groups))
      if (group.overlay == overlay) {
        members.push_back(g);
        claims.push_back(group);
      }
    if (claims.empty())
      continue;
    for (const PortRule &r : portRules(claims, taken, idBits, slots))
      rules.push_back({r.mask, r.value, members[r.group]});
    for (const PortRule &r : rules)
      if (groups[r.group].overlay == overlay)
        taken.push_back({r.mask, r.value});
  }
  return rules;
}

namespace {
/// Packets with one id entering a switchbox on one slave port, and the master
/// ports they leave by.
struct SlaveFlow {
  Port slave;
  int id;
  SmallVector<Port, 4> masters;
  bool isCtrlPkt;
};

/// The amsel each slave flow takes, and the master ports each amsel selects
/// (see planArbiters).
struct ArbiterPlan {
  std::map<std::pair<Port, int>, int> slaveAmsels;
  std::map<int, SmallVector<Port, 4>> amselMasters;
};

/// The master ports `flows` tie to one arbiter: a slave flow's arbiter takes
/// every master port it leaves by (see planArbiters).
template <typename SlaveFlows>
llvm::EquivalenceClasses<Port> tiedMasters(SlaveFlows &&flows) {
  llvm::EquivalenceClasses<Port> tied;
  for (const SlaveFlow &f : flows)
    for (Port m : f.masters)
      tied.unionSets(f.masters.front(), m);
  return tied;
}

/// Chooses arbiters for a switchbox's slave flows so that no two flows that
/// can deadlock (`conflict`) and enter on different slave ports share one.
/// A master port is tied to one arbiter, so every master port a slave flow
/// leaves by takes that flow's arbiter; the master ports linked that way form
/// a unit, and each master set in a unit takes an msel. A flow never takes
/// an arbiter `excluded` rules out for it. Units are colored onto arbiters by
/// backtracking, which is exact up to a step budget. On failure, fills
/// `blocking` with the conflicting pairs that stood in the way; it stays empty
/// when the master sets alone outnumber the free msels.
std::optional<ArbiterPlan>
planArbiters(const AIETargetModel &targetModel, ArrayRef<SlaveFlow> flows,
             llvm::function_ref<bool(size_t, size_t)> conflict,
             llvm::function_ref<bool(size_t, int)> excluded,
             const std::set<int> &reservedAmsels,
             SmallVectorImpl<std::pair<size_t, size_t>> &blocking) {
  const int numArbiters = targetModel.getNumArbiters();
  const int numMselsPerArbiter = targetModel.getNumMselsPerArbiter();
  auto amselOf = [&](int arbiter, int msel) {
    return arbiter + msel * numArbiters;
  };

  llvm::EquivalenceClasses<Port> tied = tiedMasters(flows);

  struct Unit {
    SmallVector<size_t, 4> flows;
    SmallVector<SmallVector<Port, 4>, 2> masterSets;
    bool isCtrlPkt = false;
  };
  std::vector<Unit> units;
  std::map<Port, size_t> unitOf;
  SmallVector<size_t, 8> flowUnit;
  for (auto [i, f] : llvm::enumerate(flows)) {
    auto [it, inserted] =
        unitOf.try_emplace(tied.getLeaderValue(f.masters.front()), 0);
    if (inserted) {
      it->second = units.size();
      units.emplace_back();
    }
    Unit &unit = units[it->second];
    flowUnit.push_back(it->second);
    unit.flows.push_back(i);
    unit.isCtrlPkt |= f.isCtrlPkt;
    if (!llvm::is_contained(unit.masterSets, f.masters))
      unit.masterSets.push_back(f.masters);
  }
  for (Unit &unit : units) {
    auto *ctrlEnd = std::stable_partition(
        unit.masterSets.begin(), unit.masterSets.end(),
        [&](const SmallVector<Port, 4> &masters) {
          return llvm::any_of(unit.flows, [&](size_t f) {
            return flows[f].isCtrlPkt && flows[f].masters == masters;
          });
        });
    std::sort(unit.masterSets.begin(), ctrlEnd);
  }

  auto clash = [&](size_t a, size_t b) {
    return flows[a].slave != flows[b].slave && conflict(a, b);
  };
  for (const Unit &unit : units)
    for (auto [k, a] : llvm::enumerate(unit.flows))
      for (size_t b : llvm::drop_begin(unit.flows, k + 1))
        if (clash(a, b))
          blocking.push_back({a, b});
  if (!blocking.empty())
    return std::nullopt;

  std::vector<llvm::SmallDenseSet<size_t, 4>> neighbors(units.size());
  SmallVector<std::pair<size_t, size_t>, 4> crossPairs;
  for (size_t a = 0; a < flows.size(); a++)
    for (size_t b = a + 1; b < flows.size(); b++)
      if (flowUnit[a] != flowUnit[b] && clash(a, b)) {
        neighbors[flowUnit[a]].insert(flowUnit[b]);
        neighbors[flowUnit[b]].insert(flowUnit[a]);
        crossPairs.push_back({a, b});
      }

  SmallVector<SmallVector<int, 4>, 6> freeMsels(numArbiters);
  SmallVector<SmallVector<size_t, 4>, 6> excludedUnits(numArbiters);
  for (int a = 0; a < numArbiters; a++) {
    for (int m = 0; m < numMselsPerArbiter; m++)
      if (!reservedAmsels.count(amselOf(a, m)))
        freeMsels[a].push_back(m);
    for (size_t f = 0; f < flows.size(); f++)
      if (excluded(f, a) && !llvm::is_contained(excludedUnits[a], flowUnit[f]))
        excludedUnits[a].push_back(flowUnit[f]);
  }
  size_t mostFree = 0;
  for (const SmallVector<int, 4> &free : freeMsels)
    mostFree = std::max(mostFree, free.size());
  if (llvm::any_of(units, [&](const Unit &unit) {
        return unit.masterSets.size() > mostFree;
      }))
    return std::nullopt;

  // Most constrained first.
  SmallVector<size_t, 8> order(units.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    return std::make_pair(neighbors[a].size(), units[a].masterSets.size()) >
           std::make_pair(neighbors[b].size(), units[b].masterSets.size());
  });

  SmallVector<int, 8> arbiterOf(units.size(), -1);
  SmallVector<size_t, 6> load(numArbiters, 0);
  int steps = 0;
  auto place = [&](auto &self, size_t depth) -> bool {
    if (depth == order.size())
      return true;
    if (++steps > maxSearchSteps) {
      LLVM_DEBUG(llvm::dbgs() << "Arbiter search gave up after "
                              << maxSearchSteps << " steps\n");
      return false;
    }
    size_t u = order[depth];
    SmallVector<int, 6> candidates(numArbiters);
    std::iota(candidates.begin(), candidates.end(), 0);
    if (units[u].isCtrlPkt)
      std::reverse(candidates.begin(), candidates.end());
    else
      llvm::stable_sort(candidates, [&](int a, int b) {
        return load[a] + numMselsPerArbiter - freeMsels[a].size() <
               load[b] + numMselsPerArbiter - freeMsels[b].size();
      });
    std::set<std::pair<size_t, SmallVector<size_t, 4>>> triedEmpty;
    for (int a : candidates) {
      if (load[a] + units[u].masterSets.size() > freeMsels[a].size())
        continue;
      if (llvm::is_contained(excludedUnits[a], u) ||
          llvm::any_of(neighbors[u],
                       [&](size_t v) { return arbiterOf[v] == a; }))
        continue;
      // Empty arbiters with as many free msels, ruled out for the same units,
      // are interchangeable.
      if (load[a] == 0 &&
          !triedEmpty.insert({freeMsels[a].size(), excludedUnits[a]}).second)
        continue;
      arbiterOf[u] = a;
      load[a] += units[u].masterSets.size();
      if (self(self, depth + 1))
        return true;
      load[a] -= units[u].masterSets.size();
      arbiterOf[u] = -1;
    }
    return false;
  };
  if (!place(place, 0)) {
    blocking.append(crossPairs.begin(), crossPairs.end());
    return std::nullopt;
  }

  // Control packets take the highest msels, as they do elsewhere, in master
  // port order so the overlay's msels don't depend on the design's flows (see
  // reloadsOverlay).
  SmallVector<size_t, 8> mselOrder(units.size());
  std::iota(mselOrder.begin(), mselOrder.end(), 0);
  llvm::stable_sort(mselOrder, [&](size_t a, size_t b) {
    if (!units[a].isCtrlPkt || !units[b].isCtrlPkt)
      return units[a].isCtrlPkt > units[b].isCtrlPkt;
    return units[a].masterSets.front() < units[b].masterSets.front();
  });
  ArbiterPlan plan;
  SmallVector<size_t, 6> low(numArbiters, 0), high(numArbiters, 0);
  for (size_t u : mselOrder) {
    const Unit &unit = units[u];
    int a = arbiterOf[u];
    std::map<SmallVector<Port, 4>, int> setAmsel;
    for (const SmallVector<Port, 4> &masters : unit.masterSets) {
      int msel = unit.isCtrlPkt
                     ? freeMsels[a][freeMsels[a].size() - 1 - high[a]++]
                     : freeMsels[a][low[a]++];
      setAmsel[masters] = amselOf(a, msel);
      plan.amselMasters[amselOf(a, msel)] = masters;
    }
    for (size_t f : unit.flows)
      plan.slaveAmsels[{flows[f].slave, flows[f].id}] =
          setAmsel.at(flows[f].masters);
  }
  return plan;
}

/// The tiles other than `src` and `dst` that every route between them passes.
SmallVector<TileID> cutTiles(const AIETargetModel &targetModel, TileID src,
                             TileID dst) {
  auto neighbors = [&](TileID t) {
    SmallVector<TileID, 4> next;
    for (auto [bundle, dc, dr] : {std::tuple{WireBundle::North, 0, 1},
                                  {WireBundle::South, 0, -1},
                                  {WireBundle::East, 1, 0},
                                  {WireBundle::West, -1, 0}}) {
      TileID n{t.col + dc, t.row + dr};
      if (n.col >= 0 && n.col < targetModel.columns() && n.row >= 0 &&
          n.row < targetModel.rows() &&
          targetModel.getNumDestSwitchboxConnections(t.col, t.row, bundle) > 0)
        next.push_back(n);
    }
    return next;
  };
  auto reach = [&](std::optional<TileID> avoid) {
    return breadthFirstParents(
        src,
        [&](TileID t) {
          SmallVector<TileID, 4> next = neighbors(t);
          llvm::erase(next, avoid);
          return next;
        },
        std::optional{dst});
  };
  std::map<TileID, TileID> via = reach(std::nullopt);
  if (!via.count(dst))
    return {};
  SmallVector<TileID> interior;
  for (TileID t = via.at(dst); t != src; t = via.at(t))
    interior.push_back(t);
  llvm::erase_if(interior, [&](TileID t) { return reach(t).count(dst) > 0; });
  return interior;
}

/// The tile, port and id of each packet a prioritized flow (priority_route)
/// sends.
struct PrioritizedPackets {
  std::set<std::tuple<TileID, Port, int>> packets;
  bool contains(const RoutedStream &s) const {
    return s.packetID && packets.count({s.src.tile, s.src.port, *s.packetID});
  }
};

PrioritizedPackets prioritizedPackets(DeviceOp device) {
  PrioritizedPackets prioritized;
  for (PacketFlowOp flow : device.getOps<PacketFlowOp>())
    if (flow.getPriorityRoute().value_or(false))
      for (auto src : flow.getPorts().getOps<PacketSourceOp>())
        prioritized.packets.insert(
            {cast<TileOp>(src.getTile().getDefiningOp()).getTileID(),
             src.port(), flow.IDInt()});
  return prioritized;
}

/// The largest set of `candidates` that must be kept apart pairwise
/// (StreamConflicts::mustSeparate). The search stops once it finds one larger
/// than `limit` or runs out of steps.
SmallVector<size_t, 8> largestApartSet(const StreamConflicts &conflicts,
                                       ArrayRef<size_t> candidates,
                                       size_t limit) {
  SmallVector<size_t, 8> clique, best;
  int steps = 0;
  auto grow = [&](auto &self, ArrayRef<size_t> cands) -> void {
    if (clique.size() > best.size())
      best = clique;
    for (auto [k, s] : llvm::enumerate(cands)) {
      if (best.size() > limit || ++steps > maxSearchSteps ||
          clique.size() + cands.size() - k <= best.size())
        return;
      SmallVector<size_t, 8> next;
      for (size_t t : cands.drop_front(k + 1))
        if (conflicts.mustSeparate(s, t))
          next.push_back(t);
      clique.push_back(s);
      self(self, next);
      clique.pop_back();
    }
  };
  grow(grow, candidates);
  return best;
}

/// Packet streams take an arbiter at the tile they end at whatever the
/// routing, and where `pinsHops` says hops cannot be circuit switched, or the
/// stream is prioritized, at the tile they start at and every tile each of
/// their routes passes too. Two that must be kept apart pass any tile on
/// different slave ports -- sharing one means they merged, unsafely, upstream
/// -- so a set of them that must be kept apart pairwise needs an arbiter
/// apiece. Says why no routing can work if some tile has such a set larger
/// than its free arbiters.
std::optional<std::string>
tooFewArbiters(DeviceOp device, const StreamConflicts &conflicts,
               llvm::function_ref<bool(TileID)> pinsHops,
               const PrioritizedPackets &prioritized) {
  const AIETargetModel &targetModel = device.getTargetModel();
  const int numArbiters = targetModel.getNumArbiters();
  const int numMselsPerArbiter = targetModel.getNumMselsPerArbiter();
  std::map<TileID, SmallVector<size_t, 8>> pinned;
  for (auto [i, s] : llvm::enumerate(conflicts.getRequestedStreams())) {
    if (!s.packetID)
      continue;
    pinned[s.dst.tile].push_back(i);
    if (s.src.tile == s.dst.tile)
      continue;
    bool circuitless = prioritized.contains(s);
    if (circuitless || pinsHops(s.src.tile))
      pinned[s.src.tile].push_back(i);
    for (TileID t : cutTiles(targetModel, s.src.tile, s.dst.tile))
      if (circuitless || pinsHops(t))
        pinned[t].push_back(i);
  }
  std::set<std::pair<TileID, int>> reserved;
  for (auto swboxOp : device.getOps<SwitchboxOp>())
    for (auto amselOp : swboxOp.getConnections().getOps<AMSelOp>())
      reserved.insert(
          {swboxOp.getTileOp().getTileID(),
           amselOp.arbiterIndex() + amselOp.getMselValue() * numArbiters});

  ArrayRef<RoutedStream> streams = conflicts.getStreams();
  for (const auto &[tileId, candidates] : pinned) {
    size_t free = 0;
    for (int a = 0; a < numArbiters; a++)
      free += llvm::any_of(
          llvm::seq(numMselsPerArbiter), [&, tileId = tileId](int m) {
            return !reserved.count({tileId, a + m * numArbiters});
          });
    // Streams that must be kept apart have distinct sources and destinations.
    std::set<std::pair<TileID, Port>> srcs, dsts;
    for (size_t s : candidates) {
      srcs.insert({streams[s].src.tile, streams[s].src.port});
      dsts.insert({streams[s].dst.tile, streams[s].dst.port});
    }
    if (std::min(srcs.size(), dsts.size()) <= free)
      continue;
    SmallVector<size_t, 8> best = largestApartSet(conflicts, candidates, free);
    if (best.size() <= free)
      continue;

    std::string reason;
    llvm::raw_string_ostream os(reason);
    if (best.size() == 1) {
      os << "at tile (" << tileId.col << ", " << tileId.row << "), "
         << describeStream(streams[best[0]])
         << " takes an arbiter whatever the routing, but the switchbox has "
            "none free";
      return reason;
    }
    os << "at tile (" << tileId.col << ", " << tileId.row << "), no two of ";
    llvm::interleave(
        best, os, [&](size_t s) { os << describeStream(streams[s]); }, ", ");
    os << " can share an arbiter, and each takes one there whatever the "
          "routing, but the switchbox has "
       << free << " free. For example, " << conflicts.explain(best[0], best[1]);
    return reason;
  }
  return std::nullopt;
}

/// Where a stream leaves a tile it is known to pass: by which master port from
/// which slave port, if known.
struct Leaving {
  size_t stream;
  std::optional<Port> slave;
  Port master;
};

/// Where each stream leaves each tile on the trees in `pinnedTrees`, and the
/// tiles those trees pass.
std::pair<std::map<TileID, SmallVector<Leaving, 8>>, std::set<TileID>>
leavingPinnedTrees(const StreamConflicts &conflicts,
                   const PacketTrees &pinnedTrees,
                   const PrioritizedPackets &prioritized) {
  // Each hop of a pinned tree by the hop before it, unset where two lead to it.
  std::map<PathEndPoint, std::map<PathEndPoint, std::optional<PathEndPoint>>>
      preds;
  for (const auto &[src, hops] : pinnedTrees)
    for (const TreeHop &hop : hops) {
      auto [it, first] = preds[src].try_emplace(hop.to, hop.from);
      if (!first && it->second && !(*it->second == hop.from))
        it->second.reset();
    }
  std::map<TileID, SmallVector<Leaving, 8>> leaving;
  std::set<TileID> onTrees;
  for (auto [i, s] : llvm::enumerate(conflicts.getRequestedStreams())) {
    if (!s.packetID)
      continue;
    // Only the prioritized packets of a source take its tree.
    auto tree = prioritized.contains(s) ? preds.find({s.src.tile, s.src.port})
                                        : preds.end();
    SmallVector<std::pair<Port, PathEndPoint>, 8> hops;
    PathEndPoint at{s.dst.tile, s.dst.port}, src{s.src.tile, s.src.port};
    for (size_t n = 0;
         tree != preds.end() && !(at == src) && n <= tree->second.size(); n++) {
      auto pred = tree->second.find(at);
      if (pred == tree->second.end() || !pred->second)
        break;
      if (pred->second->coords == at.coords)
        hops.push_back({pred->second->port, at});
      at = *pred->second;
    }
    if (!(at == src)) {
      leaving[s.dst.tile].push_back({i, std::nullopt, s.dst.port});
      continue;
    }
    for (auto [slave, hop] : hops) {
      leaving[hop.coords].push_back({i, slave, hop.port});
      onTrees.insert(hop.coords);
    }
  }
  return {std::move(leaving), std::move(onTrees)};
}

/// A visit of `here` on one arbiter with another: the other visit, and the
/// master port they leave by or the slave port they share a packet rule on.
struct ArbiterLink {
  size_t visit;
  Port port;
  bool rule;
};

/// Says why `a` and `b`, which must be kept apart, end up on one arbiter at
/// `tileId` through the visits in `chain`.
std::string explainSharedArbiter(TileID tileId, size_t a, size_t b,
                                 ArrayRef<ArbiterLink> chain,
                                 ArrayRef<Leaving> here,
                                 const StreamConflicts &conflicts) {
  ArrayRef<RoutedStream> streams = conflicts.getStreams();
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "at tile (" << tileId.col << ", " << tileId.row
     << "), the routes prioritized flows (priority_route) keep put "
     << describeStream(streams[a]) << " and " << describeStream(streams[b])
     << " on one arbiter: " << describeStream(streams[a]);
  for (auto [k, link] : llvm::enumerate(chain))
    os << (k == 0 ? "" : ", which")
       << (link.rule ? " takes one packet rule on " : " leaves by ")
       << describePort(link.port) << " with "
       << describeStream(streams[here[link.visit].stream]);
  os << ", and a master port or packet rule takes one arbiter. "
     << conflicts.explain(a, b);
  return reason;
}

/// Says why no routing can work if two streams that must be kept apart leave
/// `tileId` by master ports that streams leaving by one each join, which puts
/// them on one arbiter.
std::optional<std::string> sharedArbiterAt(TileID tileId,
                                           ArrayRef<Leaving> here,
                                           const StreamConflicts &conflicts) {
  ArrayRef<RoutedStream> streams = conflicts.getStreams();
  // A stream packet switched at a master port takes its one arbiter, and
  // packets with one id on a slave port take one packet rule's arbiter.
  // The visits on one arbiter with each. A stream passing a tile twice takes
  // an arbiter on each visit, so the nodes are visits.
  std::map<size_t, SmallVector<ArbiterLink, 4>> with;
  for (auto [x, l] : llvm::enumerate(here))
    for (auto [y, m] : llvm::enumerate(here)) {
      if (x == y)
        continue;
      if (l.master == m.master)
        with[x].push_back({y, l.master, false});
      else if (l.slave && m.slave && *l.slave == *m.slave &&
               streams[l.stream].packetID == streams[m.stream].packetID)
        with[x].push_back({y, *l.slave, true});
    }
  for (size_t x : llvm::make_first_range(with)) {
    // The chain of visits that joins `x` to each visit it reaches.
    std::map<size_t, size_t> via = breadthFirstParents(x, [&](size_t u) {
      return llvm::map_range(
          with.at(u), [](const ArbiterLink &link) { return link.visit; });
    });
    size_t a = here[x].stream;
    for (size_t y : llvm::make_first_range(via)) {
      size_t b = here[y].stream;
      if (b <= a || !conflicts.mustSeparate(a, b))
        continue;
      SmallVector<ArbiterLink, 4> chain;
      for (size_t v = y; v != x; v = via.at(v))
        chain.push_back(
            *llvm::find_if(with.at(via.at(v)), [&](const ArbiterLink &link) {
              return link.visit == v;
            }));
      std::reverse(chain.begin(), chain.end());
      return explainSharedArbiter(tileId, a, b, chain, here, conflicts);
    }
  }
  return std::nullopt;
}

/// A RoutingFailure if no routing can work because the arbiters packet streams
/// take whatever the routing do not fit: too few free ones at a tile, or, as a
/// PinnedRoutingFailure, two streams that must be kept apart put on one by the
/// trees in `pinnedTrees`, which prioritized sources keep.
llvm::Error unroutableArbiters(DeviceOp device,
                               const StreamConflicts &conflicts,
                               llvm::function_ref<bool(TileID)> pinsHops,
                               const PacketTrees &pinnedTrees) {
  PrioritizedPackets prioritized = prioritizedPackets(device);
  if (std::optional<std::string> reason =
          tooFewArbiters(device, conflicts, pinsHops, prioritized))
    return llvm::make_error<RoutingFailure>(std::move(*reason));
  if (pinnedTrees.empty())
    return llvm::Error::success();
  auto [leaving, onTrees] =
      leavingPinnedTrees(conflicts, pinnedTrees, prioritized);
  for (TileID tileId : onTrees)
    if (std::optional<std::string> reason =
            sharedArbiterAt(tileId, leaving[tileId], conflicts))
      return llvm::make_error<PinnedRoutingFailure>(std::move(*reason));
  return llvm::Error::success();
}
} // namespace

static void getOrCreateConnect(OpBuilder &builder, ShimMuxOp shimMux,
                               Location loc, WireBundle srcBundle, int srcCh,
                               WireBundle destBundle, int destCh) {
  for (auto connect : shimMux.getConnections().getOps<ConnectOp>())
    if (connect.getSourceBundle() == srcBundle &&
        connect.getSourceChannel() == srcCh &&
        connect.getDestBundle() == destBundle &&
        connect.getDestChannel() == destCh)
      return;
  ConnectOp::create(builder, loc, srcBundle, srcCh, destBundle, destCh);
}

static PathEndPoint sourceOf(FlowOp flow) {
  auto tile = cast<TileOp>(flow.getSource().getDefiningOp());
  return PathEndPoint{tile.getTileID(),
                      {flow.getSourceBundle(), flow.getSourceChannel()}};
}

// The router names a shim DMA by its own port; move the rules and master sets
// that use one behind the shim mux, onto the South channel it reaches the
// switchbox by (see shimMuxChannelFrom).
static void lowerShimDMAPorts(DeviceOp device, OpBuilder &builder,
                              DynamicTileAnalysis &analyzer) {
  for (auto switchbox : device.getOps<SwitchboxOp>()) {
    TileOp tileOp = switchbox.getTileOp();
    if (!tileOp.isShimNOCTile())
      continue;
    auto connectInMux = [&](Port src, Port dst) {
      builder.setInsertionPointAfter(tileOp);
      ShimMuxOp shimMux = analyzer.getShimMux(builder, tileOp.colIndex());
      builder.setInsertionPointToStart(&shimMux.getConnections().front());
      getOrCreateConnect(builder, shimMux, tileOp.getLoc(), src.bundle,
                         src.channel, dst.bundle, dst.channel);
    };
    for (Operation &op : switchbox.getConnections().front()) {
      if (auto rules = dyn_cast<PacketRulesOp>(op);
          rules && rules.getSourceBundle() == WireBundle::DMA) {
        int ch = shimMuxChannelFrom(rules.sourcePort());
        connectInMux(rules.sourcePort(), {WireBundle::North, ch});
        rules.setSourceBundle(WireBundle::South);
        rules.setSourceChannel(ch);
      }
      if (auto masterSet = dyn_cast<MasterSetOp>(op);
          masterSet && masterSet.getDestBundle() == WireBundle::DMA) {
        int ch = shimMuxChannelTo(masterSet.destPort());
        connectInMux({WireBundle::North, ch}, masterSet.destPort());
        masterSet.setDestBundle(WireBundle::South);
        masterSet.setDestChannel(ch);
      }
    }
  }
}

namespace {
// How runOnPacketFlow lowers the packet flows on one routing: the connections
// the routing lays down and the arbiters planned for them.
struct PacketPlan {
  LogicalResult emit(DeviceOp device, OpBuilder &builder,
                     DynamicTileAnalysis &analyzer) const;
  SmallVector<std::pair<int, int>, 2>
  statedCubes(const std::pair<PhysPort, int> &slaveFlow, int idMask) const;

  // A hold cycle through waits at receivers the trees share that no arbiter
  // assignment found breaks.
  std::optional<HoldCycle> sharedReceiverCycle;

  // The master ports packets with each flow ID leave by, keyed by the slave
  // port they enter.
  std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
  SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
  // Flag to keep packet header at packet flow destination
  DenseMap<PhysPort, BoolAttr> keepPktHeaderAttr;
  // The slave ports and IDs that carry prioritized packets. A switchbox routes
  // on the ID alone, so a flow sharing both with a priority flow is one too.
  DenseSet<std::pair<PhysPort, int>> prioritizedFlows;
  // The prioritized slave ports and IDs a control-packet reload keeps, as the
  // control overlay's (see reloadConfigures).
  DenseSet<std::pair<PhysPort, int>> overlayFlows;
  // The slave ports and IDs of the control overlay's flows. Their routing
  // carries the overlay marker even when they route like the others; see
  // AIEFindFlows.
  DenseSet<std::pair<PhysPort, int>> markedFlows;
  // Set of master ports that belong to control packet overlay flows
  DenseSet<PhysPort> ctrlPktOverlayMasterPorts;
  // The ports priority_route flows start at. Only their own source feeds them,
  // so their rules can mark the prioritized flows, which a shared master set
  // cannot.
  DenseSet<PhysPort> prioritizedSourcePorts;

  // Packet-rule masks the flows state, keyed by the slave port the stream
  // enters and the flow ID. One ID may reach a port under two masks, so the
  // ID alone does not identify the claim, and flows sharing both may each
  // state a mask of their own, which the port then claims together.
  std::map<std::pair<PhysPort, int>, std::set<int>> pinnedMasks;

  // A switchbox with more packet master ports than free arbiters has to put
  // two master ports on one arbiter, and an arbiter holds its grant until
  // tlast, so one stalled packet then holds up an unrelated flow. A hop that
  // alone uses both its ports needs no arbiter: a circuit connection passes the
  // header and tlast through, and the next switchbox routes the packet as
  // before. The last hop stays packet-switched, since it drops the header.
  // A circuit may broadcast, so a slave port qualifies with all the master
  // ports it reaches, provided it alone feeds them and every packet goes to
  // all of them.
  std::map<TileID, SmallVector<Connect, 4>> circuitHops;
  std::map<TileID, ArbiterPlan> plans;
};

// Plans the packet flows on a routing, leaving the IR alone. Where no plan
// fits, the faults say which connections are in the way.
struct PacketFlowRouting : PacketPlan {
  using FlowKey = std::pair<Port, int>;

  PacketFlowRouting(DeviceOp device, const Routing &routing,
                    const StreamConflicts &conflicts,
                    const PinnedOverlay &pinned, bool routeCircuit,
                    bool circuitSwitchHops, bool prioritize)
      : device(device), routing(routing), conflicts(conflicts), pinned(pinned),
        prioritize(prioritize),
        keepOverlay(!pinned.rules.empty() && reloadsOverlay(device)),
        routeCircuit(routeCircuit), circuitSwitchHops(circuitSwitchHops),
        targetModel(device.getTargetModel()),
        numArbiters(targetModel.getNumArbiters()),
        numMselsPerArbiter(targetModel.getNumMselsPerArbiter()),
        maxPacketId(targetModel.getMaxPacketId()),
        idBits(llvm::Log2_32_Ceil(maxPacketId + 1)), idMask((1 << idBits) - 1) {
  }

  llvm::Expected<PacketPlan> plan();
  void collectFlows();
  void findCircuitHops();
  void collectSlaveFlows();
  void checkRules();
  void planTiles();
  std::optional<std::string> breakHoldCycles();
  RoutingFaults faults() const;

  const SwitchSettings &settingsOf(const PathEndPoint &src,
                                   std::optional<int> id = std::nullopt);
  std::optional<std::pair<size_t, size_t>>
  conflictingStreams(TileID tileId, const SlaveFlow &a, const SlaveFlow &b);
  std::optional<ArbiterPlan>
  planTile(TileID tileId, SmallVectorImpl<std::pair<size_t, size_t>> &blocking);
  void moveFlow(TileID tileId, FlowKey key, bool overlay = false);
  void moveUnit(TileID tileId, FlowKey key);
  void splitFlow(TileID tileId, FlowKey key, ArrayRef<FlowKey> partners);

  DeviceOp device;
  const Routing &routing;
  const StreamConflicts &conflicts;
  const PinnedOverlay &pinned;
  // Whether prioritized flows not pinned are prioritized here.
  bool prioritize;
  // A reload keeps the overlay's amsels, so no plan moves them.
  bool keepOverlay;
  // Why the overlay cannot keep its amsels at a tile.
  std::map<TileID, std::string> unkeptOverlay;
  bool routeCircuit;
  bool circuitSwitchHops;
  const AIETargetModel &targetModel;
  const int numArbiters;
  const int numMselsPerArbiter;
  const uint32_t maxPacketId;
  const int idBits;
  const int idMask;

  // The streams each slave flow carries, for asking whether two flows can
  // deadlock on an arbiter. Flows with one source, destination and id may
  // state different masks, so carry different ids, and each is a stream of
  // its own.
  struct StreamEnds {
    PathEndPoint src, dst;
    int id;
    bool operator<(const StreamEnds &rhs) const {
      return std::tie(src, dst, id) < std::tie(rhs.src, rhs.dst, rhs.id);
    }
  };
  std::map<StreamEnds, SmallVector<size_t, 1>> packetStreamIndex;
  DenseMap<std::pair<PhysPort, int>, SmallVector<size_t, 2>> slaveFlowStreams;
  // Each source's part of a SlaveFlow (see SlaveFlow). A switchbox routes on
  // the id alone, so sources sharing an id there go everywhere any of them
  // does.
  std::map<std::pair<PhysPort, int>, std::map<PathEndPoint, std::set<Port>>>
      slaveFlowSources;
  // Every stream's hops, source first; the requested ones as this routing
  // lays them, the rest as the design already does.
  std::vector<SmallVector<StreamHop, 8>> routes;
  // Sources routed as circuits, which runOnFlow lowers.
  std::set<PathEndPoint> circuitSources;
  const SwitchSettings noSettings;

  // The logical model of all the switchboxes.
  std::map<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
  // <arbiter, msel> slots (per tile) that packet-switch configuration in the
  // input IR already occupies. Kept apart from masterAMSels so the allocator
  // run does not re-emit those ops.
  std::map<TileID, std::set<int>> reservedAmsels;
  // Each switchbox has its arbiters planned as a whole (see planArbiters).
  // Where that is impossible the routing is unusable, and the connections of
  // the flows in the way are what the router has to move.
  std::map<TileID, std::map<FlowKey, SlaveFlow>> tileSlaveFlows;
  std::map<TileID, SmallVector<SlaveFlow, 8>> tileFlows;
  // What the search below learns: flows to keep on different arbiters, and
  // flows to keep off an arbiter that flows already in the design take.
  std::set<std::tuple<TileID, FlowKey, FlowKey>> apart;
  std::set<std::tuple<TileID, FlowKey, int>> offArbiter;
  std::set<std::pair<TileID, Connect>> hazardConnections;
  // Those moved off the overlay's master ports.
  std::set<std::pair<TileID, Connect>> overlayConnections;
  // The TreeSplits (see TreeSplit) the routing check asks for. `key` branches
  // from the flows of its source that tie it to one of `partners`, where
  // nothing else does. The next routing may reach the tile by another slave
  // port, so the split holds for the tile.
  std::set<TreeSplit> hazardSplits;
  std::set<std::pair<PathEndPoint, PathEndPoint>> hazardApart;
  std::set<std::pair<PathEndPoint, PathEndPoint>> hazardTogether;
  std::set<TileID> crowdedTiles;
  std::optional<std::string> planFailure;
  // Whether planFailure blames the overlay a reload keeps.
  bool overlayFailure = false;
  // The first flow whose source the routing leaves unconnected. The routing
  // check plans around it, but route fails a routing with one.
  std::optional<std::pair<Location, std::string>> incomplete;
};
} // namespace

void PacketFlowRouting::collectFlows() {
  for (auto [i, s] : llvm::enumerate(conflicts.getStreams()))
    if (s.packetID)
      packetStreamIndex[{{s.src.tile, s.src.port},
                         {s.dst.tile, s.dst.port},
                         *s.packetID}]
          .push_back(i);
  for (const RoutedStream &s : conflicts.getStreams())
    routes.push_back(s.hops);

  if (routeCircuit)
    for (FlowOp flow : device.getOps<FlowOp>())
      circuitSources.insert(sourceOf(flow));

  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    int flowID = pktFlowOp.IDInt();
    SmallVector<std::pair<TileID, Port>, 4> sources;

    // Pass 1: collect all sources (order-independent; supports fan-in).
    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        auto srcTile = cast<TileOp>(pktSource.getTile().getDefiningOp());
        sources.push_back(
            {{srcTile.colIndex(), srcTile.rowIndex()}, pktSource.port()});
      }
    }
    // Pass 2: lower each (source, destination) pair so fan-in flows lay
    // down switchbox connections for every source, not just the last one.
    for (Operation &Op : b.getOperations()) {
      auto pktDest = dyn_cast<PacketDestOp>(Op);
      if (!pktDest)
        continue;
      auto destTile = cast<TileOp>(pktDest.getTile().getDefiningOp());
      Port destPort = pktDest.port();
      TileID destCoords = {destTile.colIndex(), destTile.rowIndex()};
      // Assign "keep_pkt_header flag"
      auto keep = pktFlowOp.getKeepPktHeader();
      keepPktHeaderAttr[{destTile.getTileID(), destPort}] =
          keep ? BoolAttr::get(Op.getContext(), *keep) : nullptr;

      for (auto &[srcCoords, srcPort] : sources) {
        TileID srcSB = {srcCoords.col, srcCoords.row};
        PathEndPoint srcPoint = {srcSB, srcPort};
        if (circuitSources.count(srcPoint))
          continue;
        const SwitchSettings &settings = settingsOf(srcPoint, flowID);
        // The requested streams this flow asks for here; with none, the
        // stream the design already routes.
        SmallVector<size_t, 1> flowStreams;
        if (auto stream = packetStreamIndex.find(
                {srcPoint, {destCoords, destPort}, flowID});
            stream != packetStreamIndex.end()) {
          int mask = static_cast<int>(pktFlowOp.getMask().value_or(~0));
          for (size_t i : stream->second)
            if (i < conflicts.getRequestedStreams().size() &&
                conflicts.getStreams()[i].packetMask == mask)
              flowStreams.push_back(i);
          if (flowStreams.empty())
            flowStreams.push_back(stream->second.front());
        }
        for (size_t stream : flowStreams) {
          SmallVector<StreamHop, 8> &hops = routes[stream];
          hops.clear();
          // A route may pass a switchbox more than once, but no connection.
          size_t connections = 0;
          for (const auto &[_, setting] : settings)
            connections += setting.srcs.size();
          std::optional<std::pair<TileID, Port>> at{{srcSB, srcPort}};
          while (at && hops.size() <= connections) {
            auto [tile, input] = *at;
            hops.push_back({tile, input, std::nullopt});
            at.reset();
            auto setting = settings.find(tile);
            if (setting == settings.end())
              break;
            for (auto [src, dest] :
                 llvm::zip(setting->second.srcs, setting->second.dsts))
              if (src == input && !(tile == destCoords && dest == destPort) &&
                  findPathToDest(settings, tile, dest, destCoords, destPort)) {
                at = linkedInput(tile, dest);
                break;
              }
          }
        }
        // Track whether the source's own switch connection survives routing.
        bool srcRouted = false;
        // add connections for all the Switchboxes in SwitchSettings
        for (const auto &[curr, setting] : settings) {
          assert(setting.srcs.size() == setting.dsts.size());
          TileID currTile = {curr.col, curr.row};
          for (size_t i = 0; i < setting.srcs.size(); i++) {
            Port src = setting.srcs[i];
            Port dest = setting.dsts[i];
            // reject false broadcast
            if (!findPathToDest(settings, currTile, dest, destCoords, destPort))
              continue;
            if (currTile == srcSB && src.bundle == srcPort.bundle &&
                src.channel == srcPort.channel)
              srcRouted = true;
            Connect connect = {{src.bundle, src.channel},
                               {dest.bundle, dest.channel}};
            if (std::find(
                    switchboxes[currTile].begin(), switchboxes[currTile].end(),
                    std::pair{connect, flowID}) == switchboxes[currTile].end())
              switchboxes[currTile].push_back({connect, flowID});
            // Keyed by slave port as well as ID, since IDs are reused across
            // unrelated flows.
            PhysPort slavePort = {currTile, {src.bundle, src.channel}};
            std::pair<PhysPort, int> slaveFlow = {slavePort, flowID};
            for (size_t stream : flowStreams)
              if (!llvm::is_contained(slaveFlowStreams[slaveFlow], stream))
                slaveFlowStreams[slaveFlow].push_back(stream);
            slaveFlowSources[slaveFlow][srcPoint].insert(dest);
            if (std::optional<uint8_t> mask = pktFlowOp.getMask()) {
              pinnedMasks[{{currTile, {src.bundle, src.channel}}, flowID}]
                  .insert(*mask);
            }
            if (pktFlowOp.getPriorityRoute().value_or(false) &&
                !reloadConfigures(pktFlowOp))
              markedFlows.insert(slaveFlow);
            if (pktFlowOp.getPriorityRoute().value_or(false) &&
                (prioritize || pinned.trees.count(srcPoint))) {
              prioritizedFlows.insert(slaveFlow);
              if (!reloadConfigures(pktFlowOp))
                overlayFlows.insert(slaveFlow);
              if (slavePort == PhysPort{srcSB, srcPort})
                prioritizedSourcePorts.insert(slavePort);
            }
          }
        }
        if (!srcRouted && !incomplete)
          incomplete = {
              pktFlowOp.getLoc(),
              "packet flow source " + describeTilePort(srcCoords, srcPort) +
                  " could not be routed to destination " +
                  describeTilePort(destCoords, destPort) +
                  "; the pathfinder produced an incomplete routing for this "
                  "placement."};
      }
    }
  }
}

// The rules the flows through `slaveFlow` state. A full-width mask selects the
// id alone, which the cover states just as well, so only a wider claim becomes
// a rule of its own.
SmallVector<std::pair<int, int>, 2>
PacketPlan::statedCubes(const std::pair<PhysPort, int> &slaveFlow,
                        int idMask) const {
  SmallVector<std::pair<int, int>, 2> cubes;
  auto it = pinnedMasks.find(slaveFlow);
  if (it != pinnedMasks.end())
    for (int mask : it->second)
      if (mask != idMask)
        cubes.push_back({mask, slaveFlow.second});
  return cubes;
}

const SwitchSettings &PacketFlowRouting::settingsOf(const PathEndPoint &src,
                                                    std::optional<int> id) {
  if (id)
    if (auto own = routing.idSettings.find({src, *id});
        own != routing.idSettings.end())
      return own->second;
  auto it = routing.settings.find(src);
  return it == routing.settings.end() ? noSettings : it->second;
}

void PacketFlowRouting::findCircuitHops() {
  // Seed the reserved set from the packet-switch configuration the switchboxes
  // already carry, so this run allocates around it instead of over it.
  for (auto swboxOp : device.getOps<SwitchboxOp>()) {
    TileID tileId = swboxOp.getTileOp().getTileID();
    // An amsel no master set uses still has rules steering packets to it.
    for (auto amselOp : swboxOp.getConnections().getOps<AMSelOp>())
      reservedAmsels[tileId].insert(amselOp.arbiterIndex() +
                                    amselOp.getMselValue() * numArbiters);
  }

  // Ports the input's own aie.switchbox ops already drive, circuit or packet.
  std::set<PhysPort> claimedSlaves, claimedMasters;
  for (auto swbox : device.getOps<SwitchboxOp>()) {
    TileID tileId = swbox.getTileOp().getTileID();
    Region &ops = swbox.getConnections();
    for (auto connect : ops.getOps<ConnectOp>()) {
      claimedSlaves.insert({tileId, connect.sourcePort()});
      claimedMasters.insert({tileId, connect.destPort()});
    }
    for (auto rules : ops.getOps<PacketRulesOp>())
      claimedSlaves.insert({tileId, rules.sourcePort()});
    for (auto masterSet : ops.getOps<MasterSetOp>())
      claimedMasters.insert({tileId, masterSet.destPort()});
  }
  // Circuits not yet lowered claim their ports all the same.
  if (routeCircuit)
    for (FlowOp flow : device.getOps<FlowOp>())
      for (const auto &[tileId, setting] : settingsOf(sourceOf(flow))) {
        for (Port p : setting.srcs)
          claimedSlaves.insert({tileId, p});
        for (Port p : setting.dsts)
          claimedMasters.insert({tileId, p});
      }
  for (auto &[tileId, connects] : switchboxes) {
    if (!circuitSwitchHops ||
        targetModel.isShimNOCorPLTile(tileId.col, tileId.row))
      continue;
    std::set<Port> masters;
    for (const auto &[conn, flowID] : connects)
      masters.insert(conn.dst);
    SmallVector<Port, 8> slaves;
    for (const auto &[conn, flowID] : connects)
      if (!llvm::is_contained(slaves, conn.src))
        slaves.push_back(conn.src);
    const std::set<int> &reserved = reservedAmsels[tileId];
    size_t freeArbiters = llvm::count_if(llvm::seq(0, numArbiters), [&](int a) {
      return llvm::none_of(llvm::seq(0, numMselsPerArbiter), [&](int m) {
        return reserved.count(a + m * numArbiters);
      });
    });
    for (Port slave : slaves) {
      if (masters.size() <= freeArbiters)
        break;
      std::set<Port> reached;
      std::map<int, std::set<Port>> reachedByID;
      for (const auto &[conn, flowID] : connects)
        if (conn.src == slave) {
          reached.insert(conn.dst);
          reachedByID[flowID].insert(conn.dst);
        }
      bool exclusive =
          !claimedSlaves.count({tileId, slave}) &&
          llvm::all_of(reached,
                       [&, tileId = tileId](Port m) {
                         return isDirectional(m.bundle) &&
                                !claimedMasters.count({tileId, m});
                       }) &&
          llvm::all_of(
              reachedByID,
              [&](const auto &ids) { return ids.second == reached; }) &&
          llvm::all_of(connects, [&, tileId = tileId](const auto &entry) {
            const auto &[other, otherID] = entry;
            if (!reached.count(other.dst))
              return true;
            return other.src == slave &&
                   !prioritizedFlows.contains({{tileId, slave}, otherID});
          });
      if (!exclusive)
        continue;
      for (Port m : reached) {
        circuitHops[tileId].push_back({slave, m});
        masters.erase(m);
      }
      llvm::erase_if(connects, [&](const auto &entry) {
        return entry.first.src == slave;
      });
    }
  }
}

void PacketFlowRouting::collectSlaveFlows() {
  LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

  for (const auto &[tileId, connects] : switchboxes) {
    LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << tileId.col << " "
                            << tileId.row << '\n');
    for (const auto &[conn, flowID] : connects) {
      Port sourcePort = conn.src;
      Port destPort = conn.dst;
      auto sourceFlow =
          std::make_pair(std::make_pair(tileId, sourcePort), flowID);
      packetFlows[sourceFlow].push_back({tileId, destPort});
      if (markedFlows.contains(sourceFlow))
        ctrlPktOverlayMasterPorts.insert({tileId, destPort});
      slavePorts.push_back(sourceFlow);
      LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                              << stringifyWireBundle(sourcePort.bundle) << " "
                              << sourcePort.channel << " -> "
                              << stringifyWireBundle(destPort.bundle) << " "
                              << destPort.channel << "\n");
    }
  }

  for (const auto &[flow, dests] : packetFlows) {
    SlaveFlow &f = tileSlaveFlows[flow.first.first]
                       .try_emplace({flow.first.second, flow.second},
                                    SlaveFlow{flow.first.second,
                                              flow.second,
                                              {},
                                              overlayFlows.contains(flow)})
                       .first->second;
    for (const PhysPort &dest : dests)
      if (!llvm::is_contained(f.masters, dest.second))
        f.masters.push_back(dest.second);
    llvm::sort(f.masters);
  }
  for (const auto &[tileId, byFlow] : tileSlaveFlows)
    for (const auto &[key, f] : byFlow)
      tileFlows[tileId].push_back(f);
}

std::optional<std::pair<size_t, size_t>>
PacketFlowRouting::conflictingStreams(TileID tileId, const SlaveFlow &a,
                                      const SlaveFlow &b) {
  auto as = slaveFlowStreams.find({{tileId, a.slave}, a.id});
  auto bs = slaveFlowStreams.find({{tileId, b.slave}, b.id});
  if (as == slaveFlowStreams.end() || bs == slaveFlowStreams.end())
    return std::nullopt;
  for (size_t s : as->second)
    for (size_t t : bs->second)
      if (conflicts.conflict(s, t))
        return std::pair{s, t};
  return std::nullopt;
}

std::optional<ArbiterPlan> PacketFlowRouting::planTile(
    TileID tileId, SmallVectorImpl<std::pair<size_t, size_t>> &blocking) {
  ArrayRef<SlaveFlow> flows = tileFlows.at(tileId);
  auto key = [&](size_t f) { return FlowKey{flows[f].slave, flows[f].id}; };
  auto conflict = [&](size_t a, size_t b) {
    return apart.count(
               {tileId, std::min(key(a), key(b)), std::max(key(a), key(b))}) ||
           conflictingStreams(tileId, flows[a], flows[b]);
  };
  auto excluded = [&](size_t f, int arbiter) {
    return offArbiter.count({tileId, key(f), arbiter}) > 0;
  };
  SmallVector<size_t, 8> overlay, others;
  for (auto [i, f] : llvm::enumerate(flows))
    (f.isCtrlPkt ? overlay : others).push_back(i);
  if (overlay.empty() || (others.empty() && pinned.rules.empty()))
    return planArbiters(targetModel, flows, conflict, excluded,
                        reservedAmsels[tileId], blocking);

  auto subset = [&](ArrayRef<size_t> members) {
    SmallVector<SlaveFlow, 8> sub;
    for (size_t m : members)
      sub.push_back(flows[m]);
    return sub;
  };
  // The overlay's flows take the amsels they take on their own where they
  // can, and the other flows plan around them. A reload keeps them;
  // otherwise, if the design rules them out, they are planned anew.
  auto keepPinned = [&]() -> std::optional<ArbiterPlan> {
    std::string reason;
    llvm::raw_string_ostream os(reason);
    auto unkept = [&](size_t f) -> llvm::raw_ostream & {
      return os << "packet flow " << flows[f].id << " into "
                << describePort(flows[f].slave) << ' ';
    };
    ArbiterPlan kept;
    for (size_t f : overlay) {
      // The overlay's rules name a shim DMA by the mux port it enters by.
      Port slave = flows[f].slave;
      if (slave.bundle == WireBundle::DMA &&
          targetModel.isShimNOCTile(tileId.col, tileId.row))
        slave = {WireBundle::South, shimMuxChannelFrom(slave)};
      auto rules = pinned.rules.find({tileId, slave});
      const std::tuple<int, int, int> *rule = nullptr;
      if (rules != pinned.rules.end())
        rule = llvm::find_if(rules->second, [&](const auto &r) {
          auto [mask, value, amsel] = r;
          return (flows[f].id & mask) == (value & mask);
        });
      if (!rule || rule == rules->second.end()) {
        unkept(f) << "takes no packet rule there alone.";
      } else if (int amsel = std::get<2>(*rule);
                 excluded(f, amsel % numArbiters)) {
        unkept(f) << "takes arbiter " << amsel % numArbiters
                  << " alone, and here can deadlock holding it.";
      } else if (reservedAmsels[tileId].count(amsel)) {
        unkept(f) << "takes amsel<" << amsel % numArbiters << "> ("
                  << amsel / numArbiters
                  << ") alone, and the design's own switchbox takes it.";
      } else if (kept.amselMasters.try_emplace(amsel, flows[f].masters)
                     .first->second != flows[f].masters) {
        unkept(f) << "takes amsel<" << amsel % numArbiters << "> ("
                  << amsel / numArbiters
                  << ") alone, to other master ports than here.";
      } else {
        kept.slaveAmsels[key(f)] = amsel;
        continue;
      }
      unkeptOverlay[tileId] = std::move(reason);
      return std::nullopt;
    }
    for (auto [k, a] : llvm::enumerate(overlay))
      for (size_t b : llvm::drop_begin(overlay, k + 1))
        if (kept.slaveAmsels.at(key(a)) % numArbiters ==
                kept.slaveAmsels.at(key(b)) % numArbiters &&
            flows[a].slave != flows[b].slave && conflict(a, b)) {
          unkept(a) << "and packet flow " << flows[b].id << " into "
                    << describePort(flows[b].slave)
                    << " share an arbiter alone, and here can deadlock on it.";
          unkeptOverlay[tileId] = std::move(reason);
          return std::nullopt;
        }
    return kept;
  };
  SmallVector<std::pair<size_t, size_t>, 4> stuck;
  std::optional<ArbiterPlan> plan;
  if (!pinned.rules.empty())
    plan = keepPinned();
  if (!plan && keepOverlay)
    return std::nullopt;
  if (!plan) {
    plan = planArbiters(
        targetModel, subset(overlay),
        [&](size_t a, size_t b) { return conflict(overlay[a], overlay[b]); },
        [&](size_t f, int arbiter) { return excluded(overlay[f], arbiter); },
        reservedAmsels[tileId], stuck);
    if (!plan) {
      for (auto [a, b] : stuck)
        blocking.push_back({overlay[a], overlay[b]});
      return std::nullopt;
    }
  }
  if (others.empty())
    return plan;

  // A flow that leaves by the overlay's master ports alone takes the amsel
  // that selects them (checkRules lets no other flow reach them).
  auto around = [&](ArbiterPlan plan, ArrayRef<size_t> planned,
                    ArrayRef<size_t> pending,
                    SmallVectorImpl<std::pair<size_t, size_t>> &blocking)
      -> std::optional<ArbiterPlan> {
    auto arbiterOf = [&](size_t f) {
      return plan.slaveAmsels.at(key(f)) % numArbiters;
    };
    SmallVector<size_t, 8> placed(planned), rest;
    for (size_t f : pending) {
      auto same = llvm::find_if(plan.amselMasters, [&](const auto &entry) {
        return entry.second == flows[f].masters;
      });
      if (same == plan.amselMasters.end()) {
        rest.push_back(f);
        continue;
      }
      for (size_t g : placed)
        if (arbiterOf(g) == same->first % numArbiters &&
            flows[g].slave != flows[f].slave && conflict(f, g))
          blocking.push_back({g, f});
      if (excluded(f, same->first % numArbiters))
        blocking.push_back({planned.front(), f});
      plan.slaveAmsels[key(f)] = same->first;
      placed.push_back(f);
    }
    if (!blocking.empty())
      return std::nullopt;
    if (rest.empty())
      return plan;

    // A master port is tied to one arbiter, so a flow leaving by a master port
    // the plan has tied takes that arbiter.
    llvm::EquivalenceClasses<Port> tied = tiedMasters(flows);
    std::map<Port, std::set<int>> tiedArbiters;
    std::set<int> reserved = reservedAmsels[tileId];
    for (const auto &[amsel, masters] : plan.amselMasters) {
      reserved.insert(amsel);
      tiedArbiters[tied.getLeaderValue(masters.front())].insert(amsel %
                                                                numArbiters);
    }
    SmallVector<std::pair<size_t, size_t>, 4> stuck;
    std::optional<ArbiterPlan> rested = planArbiters(
        targetModel, subset(rest),
        [&](size_t a, size_t b) { return conflict(rest[a], rest[b]); },
        [&](size_t f, int arbiter) {
          auto arbiters = tiedArbiters.find(
              tied.getLeaderValue(flows[rest[f]].masters.front()));
          return excluded(rest[f], arbiter) ||
                 (arbiters != tiedArbiters.end() &&
                  (arbiters->second.size() > 1 ||
                   !arbiters->second.count(arbiter))) ||
                 llvm::any_of(placed, [&](size_t g) {
                   return arbiterOf(g) == arbiter &&
                          flows[g].slave != flows[rest[f]].slave &&
                          conflict(rest[f], g);
                 });
        },
        reserved, stuck);
    if (!rested) {
      for (auto [a, b] : stuck)
        blocking.push_back({rest[a], rest[b]});
      for (size_t f : rest)
        for (size_t g : placed)
          if (flows[g].slave != flows[f].slave && conflict(f, g))
            blocking.push_back({g, f});
      return std::nullopt;
    }
    plan.slaveAmsels.insert(rested->slaveAmsels.begin(),
                            rested->slaveAmsels.end());
    plan.amselMasters.insert(rested->amselMasters.begin(),
                             rested->amselMasters.end());
    return plan;
  };
  SmallVector<std::pair<size_t, size_t>, 4> first;
  if (std::optional<ArbiterPlan> placed = around(*plan, overlay, others, first))
    return placed;
  if (keepOverlay) {
    blocking.append(first.begin(), first.end());
    return std::nullopt;
  }

  // The flows that take the overlay's arbiters may conflict where the
  // overlay's flows do not, so they are planned with them.
  llvm::EquivalenceClasses<Port> tied = tiedMasters(flows);
  SmallVector<size_t, 8> joint(overlay), rest;
  for (size_t f : others) {
    bool tiedToOverlay = llvm::any_of(overlay, [&](size_t g) {
      return tied.isEquivalent(flows[g].masters.front(),
                               flows[f].masters.front());
    });
    (tiedToOverlay ? joint : rest).push_back(f);
  }
  stuck.clear();
  std::optional<ArbiterPlan> together;
  if (joint.size() > overlay.size())
    together = planArbiters(
        targetModel, subset(joint),
        [&](size_t a, size_t b) { return conflict(joint[a], joint[b]); },
        [&](size_t f, int arbiter) { return excluded(joint[f], arbiter); },
        reservedAmsels[tileId], stuck);
  if (!together) {
    blocking.append(first.begin(), first.end());
    return std::nullopt;
  }
  return around(*together, joint, rest, blocking);
}

void PacketFlowRouting::moveFlow(TileID tileId, FlowKey key, bool overlay) {
  auto move = [&](const Connect &c) {
    hazardConnections.insert({tileId, c});
    if (overlay)
      overlayConnections.insert({tileId, c});
  };
  auto flows = tileSlaveFlows.find(tileId);
  if (flows != tileSlaveFlows.end())
    if (auto f = flows->second.find(key); f != flows->second.end())
      for (Port m : f->second.masters)
        move({key.first, m});
  for (const Connect &hop : circuitHops[tileId])
    if (hop.src == key.first)
      move(hop);
}

// Moves `key` with its whole unit (see planArbiters): a flow's arbiter is
// fixed by every flow reaching its master ports too.
void PacketFlowRouting::moveUnit(TileID tileId, FlowKey key) {
  moveFlow(tileId, key);
  auto flows = tileSlaveFlows.find(tileId);
  if (flows == tileSlaveFlows.end())
    return;
  auto self = flows->second.find(key);
  if (self == flows->second.end())
    return;
  llvm::EquivalenceClasses<Port> tied =
      tiedMasters(llvm::make_second_range(flows->second));
  for (const auto &[other, f] : flows->second)
    if (f.masters.size() > 1 &&
        tied.isEquivalent(f.masters.front(), self->second.masters.front()))
      moveFlow(tileId, other);
}

void PacketFlowRouting::splitFlow(TileID tileId, FlowKey key,
                                  ArrayRef<FlowKey> partners) {
  auto flows = tileSlaveFlows.find(tileId);
  if (flows == tileSlaveFlows.end())
    return;
  auto self = flows->second.find(key);
  if (self == flows->second.end())
    return;
  auto onlySource = [&](FlowKey k) -> std::optional<PathEndPoint> {
    auto sources = slaveFlowSources.find({{tileId, k.first}, k.second});
    if (sources == slaveFlowSources.end() || sources->second.size() != 1)
      return std::nullopt;
    return sources->second.begin()->first;
  };
  std::optional<PathEndPoint> src = onlySource(key);
  if (!src)
    return;
  auto sharesMaster = [&](const auto &f, const std::set<Port> &ports) {
    return llvm::any_of(f.masters, [&](Port m) { return ports.count(m); });
  };
  std::set<Port> own(self->second.masters.begin(), self->second.masters.end());
  std::set<FlowKey> siblings;
  for (const auto &[other, f] : flows->second)
    if (other != key && other.first == key.first && sharesMaster(f, own) &&
        onlySource(other) == src)
      siblings.insert(other);
  // The masters tied to `partner`'s arbiter, not through `key`, nor through
  // its siblings if `skipSiblings`.
  auto tied = [&](FlowKey partner, bool skipSiblings) {
    llvm::EquivalenceClasses<Port> classes =
        tiedMasters(llvm::make_filter_range(
            llvm::make_second_range(flows->second), [&](const SlaveFlow &f) {
              FlowKey other{f.slave, f.id};
              return other != key && !(skipSiblings && siblings.count(other));
            }));
    auto unit = classes.members(flows->second.at(partner).masters.front());
    return std::set<Port>(unit.begin(), unit.end());
  };
  auto split = [&](FlowKey other,
                   std::optional<PathEndPoint> apart = std::nullopt) {
    int a = key.second, b = other.second;
    if (!apart && a > b)
      std::swap(a, b);
    hazardSplits.insert({*src, tileId, a, b, apart});
  };
  for (FlowKey partner : partners) {
    if (partner == key || !flows->second.count(partner))
      continue;
    if (siblings.count(partner)) {
      split(partner);
      continue;
    }
    std::set<Port> partnerUnit = tied(partner, true);
    if (sharesMaster(self->second, partnerUnit))
      continue;
    std::set<Port> unit = tied(partner, false);
    for (FlowKey other : siblings) {
      const SlaveFlow &f = flows->second.find(other)->second;
      if (!sharesMaster(f, unit))
        continue;
      // Where the sibling only ties `key` to the partner's arbiter by
      // destinations on this tile, those can reach it by a slave port of
      // their own, and `key` keeps sharing the rest of the tree.
      SmallVector<PathEndPoint, 2> apart;
      for (Port m : f.masters)
        if (partnerUnit.count(m))
          apart.push_back({tileId, m});
      if (PathEndPoint{tileId, key.first} == *src || apart.empty() ||
          !llvm::all_of(apart, [&](const PathEndPoint &p) {
            return packetStreamIndex.count({*src, p, other.second});
          })) {
        split(other);
        continue;
      }
      for (const PathEndPoint &p : apart)
        split(other, p);
    }
  }
}

// Plans the arbiters of every switchbox on the routing. Where that fails, the
// RoutingFailure names the connections in the way.
llvm::Expected<PacketPlan> PacketFlowRouting::plan() {
  collectFlows();
  findCircuitHops();
  collectSlaveFlows();
  checkRules();
  planTiles();
  if (!planFailure)
    planFailure = breakHoldCycles();
  if (planFailure && overlayFailure)
    return llvm::make_error<PinnedRoutingFailure>(std::move(*planFailure),
                                                  faults());
  if (planFailure)
    return llvm::make_error<RoutingFailure>(std::move(*planFailure), faults());
  return PacketPlan(std::move(*this));
}

RoutingFaults PacketFlowRouting::faults() const {
  RoutingFaults faults;
  faults.connections.assign(hazardConnections.begin(), hazardConnections.end());
  faults.splits.assign(hazardSplits.begin(), hazardSplits.end());
  faults.crowded.assign(crowdedTiles.begin(), crowdedTiles.end());
  faults.apart.assign(hazardApart.begin(), hazardApart.end());
  faults.together.assign(hazardTogether.begin(), hazardTogether.end());
  faults.onlyOverlayMasters = !overlayConnections.empty() &&
                              hazardSplits.empty() && crowdedTiles.empty() &&
                              hazardApart.empty() &&
                              hazardConnections == overlayConnections;
  return faults;
}

void PacketFlowRouting::checkRules() {
  // A slave port holds a few packet rules, and how many its flows need depends
  // on which ids leave by the same master ports.
  std::map<PhysPort, SmallVector<std::pair<int, int>>> existingCubes;
  for (auto swbox : device.getOps<SwitchboxOp>())
    for (auto rules : swbox.getConnections().getOps<PacketRulesOp>())
      for (auto rule : rules.getRules().getOps<PacketRuleOp>())
        existingCubes[{swbox.getTileOp().getTileID(), rules.sourcePort()}]
            .push_back({rule.maskInt(), rule.valueInt()});
  struct RuleGroup {
    SmallVector<std::pair<int, int>, 4> stated;
    SmallVector<int, 4> derived;
  };
  for (const auto &[slaveFlow, sources] : slaveFlowSources) {
    const auto &[first, firstMasters] = *sources.begin();
    auto other = llvm::find_if(
        sources, [&, &firstMasters = firstMasters](const auto &source) {
          return source.second != firstMasters;
        });
    if (other == sources.end())
      continue;
    const auto &[slavePort, id] = slaveFlow;
    TileID tileId = slavePort.first;
    moveFlow(tileId, {slavePort.second, id});
    if (planFailure)
      continue;
    const PathEndPoint &second = other->first;
    planFailure = llvm::formatv(
        "at tile ({0}, {1}), packets with id {2} from {3} and {4} enter on {5} "
        "and leave by different ports; a switchbox routes on the id alone, so "
        "each source's packets would also go where the other's do.",
        tileId.col, tileId.row, id, describeTilePort(first.coords, first.port),
        describeTilePort(second.coords, second.port),
        describePort(slavePort.second));
  }
  for (const auto &[tileId, byFlow] : tileSlaveFlows) {
    // The packet rules the flows entering on `slave` add, without the packets
    // `without` sends to a master port if set, or only the control overlay's.
    // `groupMasters`, if set, gets the master ports each rule group selects.
    auto newRules =
        [&, tileId = tileId, &byFlow = byFlow](
            Port slave, std::optional<std::pair<PathEndPoint, Port>> without,
            bool overlayOnly,
            SmallVector<SmallVector<Port, 4>> *groupMasters = nullptr) {
          std::map<std::pair<bool, SmallVector<Port, 4>>, RuleGroup> groups;
          for (const auto &[key, f] : byFlow) {
            if (f.slave != slave || (overlayOnly && !f.isCtrlPkt))
              continue;
            SmallVector<Port, 4> masters = f.masters;
            auto sources = slaveFlowSources.find({{tileId, slave}, f.id});
            if (without && sources != slaveFlowSources.end()) {
              std::set<Port> kept;
              for (const auto &[src, ports] : sources->second)
                for (Port m : ports)
                  if (!(src == without->first) || m != without->second)
                    kept.insert(m);
              masters.assign(kept.begin(), kept.end());
            }
            if (masters.empty())
              continue;
            RuleGroup &group = groups[{f.isCtrlPkt, masters}];
            auto cubes = statedCubes({{tileId, f.slave}, f.id}, idMask);
            if (cubes.empty())
              group.derived.push_back(f.id);
            for (auto cube : cubes)
              if (!llvm::is_contained(group.stated, cube))
                group.stated.push_back(cube);
          }
          SmallVector<GroupClaims> claims;
          for (const auto &[masters, group] : groups) {
            claims.push_back({group.stated, group.derived, masters.first});
            if (groupMasters)
              groupMasters->push_back(masters.second);
          }
          return slavePortRules(claims, existingCubes[{tileId, slave}], idBits,
                                targetModel.getNumSlaveSlots());
        };
    auto rulesNeeded =
        [&, tileId = tileId](
            Port slave,
            std::optional<std::pair<PathEndPoint, Port>> without = {}) {
          return existingCubes[{tileId, slave}].size() +
                 newRules(slave, without, false).size();
        };
    // Packets for a destination on the tile that reach it by a slave port of
    // their own take their rules with them. The source's tree splits apart the
    // destination that frees the most rules on `slave`, if one frees any.
    auto splitRules = [&, tileId = tileId, &byFlow = byFlow](Port slave) {
      std::map<PathEndPoint, std::map<PathEndPoint, std::set<int>>> dstIds;
      for (const auto &[key, f] : byFlow) {
        if (f.slave != slave)
          continue;
        auto streams = slaveFlowStreams.find({{tileId, slave}, f.id});
        if (streams == slaveFlowStreams.end())
          continue;
        for (size_t i : streams->second) {
          const RoutedStream &s = conflicts.getStreams()[i];
          dstIds[{s.src.tile, s.src.port}][{s.dst.tile, s.dst.port}].insert(
              f.id);
        }
      }
      std::optional<std::pair<PathEndPoint, PathEndPoint>> best;
      size_t fewest = rulesNeeded(slave);
      for (const auto &[src, dsts] : dstIds)
        for (const auto &[dst, ids] : dsts)
          if (dst.coords == tileId && dsts.size() > 1)
            if (size_t n = rulesNeeded(slave, {{src, dst.port}}); n < fewest) {
              fewest = n;
              best = {src, dst};
            }
      if (!best)
        return false;
      const auto &[src, apart] = *best;
      const std::set<int> &apartIds = dstIds[src][apart];
      for (const auto &[dst, ids] : dstIds[src]) {
        auto carries = [&, &src = src, &dst = dst](int id) {
          return packetStreamIndex.count({src, dst, id}) > 0;
        };
        auto b = llvm::find_if_not(apartIds, carries);
        if (!(dst == apart) && b != apartIds.end())
          hazardSplits.insert({src, tileId, *ids.begin(), *b, apart});
      }
      return true;
    };
    // Flows on one slave port that leave by different master ports must not
    // claim a common id, and a mask written on an aie.packet_flow claims every
    // id it matches.
    auto claim = [&, tileId = tileId](const SlaveFlow &f) {
      auto cubes = statedCubes({{tileId, f.slave}, f.id}, idMask);
      if (cubes.empty())
        cubes.push_back({idMask, f.id});
      return cubes;
    };
    for (auto a = byFlow.begin(); a != byFlow.end(); ++a)
      for (auto b = std::next(a); b != byFlow.end(); ++b) {
        const SlaveFlow &fa = a->second, &fb = b->second;
        if (fa.slave != fb.slave || fa.masters == fb.masters)
          continue;
        std::optional<std::pair<std::pair<int, int>, std::pair<int, int>>>
            overlap;
        for (auto own : claim(fa))
          for (auto other : claim(fb))
            if (!overlap && cubesIntersect(own, other))
              overlap = {own, other};
        if (!overlap)
          continue;
        auto [own, other] = *overlap;
        moveFlow(tileId, a->first);
        moveFlow(tileId, b->first);
        if (planFailure)
          continue;
        int witness = (own.second & own.first) |
                      (other.second & other.first & ~own.first);
        planFailure = llvm::formatv(
            "at tile ({0}, {1}), packet flows through {2} claim rule (mask "
            "0x{3:X-}, id 0x{4:X-}) and rule (mask 0x{5:X-}, id 0x{6:X-}), "
            "which both match id 0x{7:X-}; widen one mask to carry both, or "
            "route them apart.",
            tileId.col, tileId.row, describePort(fa.slave), own.first,
            own.second, other.first, other.second, witness);
      }
    std::set<Port> slaves;
    for (const auto &[key, f] : byFlow)
      slaves.insert(f.slave);
    // A reload keeps the overlay's master sets, so another flow reaches the
    // overlay's master ports only by one of them.
    // Without a reload, they hold only while the overlay is pinned.
    std::set<Port> overlayMasters;
    std::set<SmallVector<Port, 4>> overlaySets;
    for (const auto &[key, f] : byFlow)
      if (f.isCtrlPkt && !pinned.rules.empty()) {
        overlayMasters.insert(f.masters.begin(), f.masters.end());
        overlaySets.insert(f.masters);
      }
    for (const auto &[key, f] : byFlow) {
      const Port *shared = llvm::find_if(
          f.masters, [&](Port m) { return overlayMasters.count(m); });
      if (f.isCtrlPkt || shared == f.masters.end() ||
          overlaySets.count(f.masters))
        continue;
      moveFlow(tileId, key, /*overlay=*/true);
      if (!planFailure)
        planFailure = llvm::formatv(
            "at tile ({0}, {1}), packets with id {2} entering on {3} leave by "
            "{4}, a master port of the prioritized flows (the control "
            "overlay), but not by one of their master sets, which a "
            "control-packet reload keeps.",
            tileId.col, tileId.row, f.id, describePort(f.slave),
            describePort(*shared));
    }
    // A reload keeps the overlay's rules, which match first, so they must send
    // no other flow's packets elsewhere.
    for (Port slave : pinned.rules.empty() ? std::set<Port>{} : slaves) {
      SmallVector<SmallVector<Port, 4>> ruleMasters;
      SmallVector<PortRule> overlay = newRules(slave, {}, true, &ruleMasters);
      for (const auto &[key, f] : byFlow) {
        if (f.slave != slave || f.isCtrlPkt)
          continue;
        for (auto own : claim(f)) {
          const auto *rule =
              llvm::find_if(overlay, [&, &f = f](const PortRule &r) {
                return cubesIntersect(own, {r.mask, r.value}) &&
                       ruleMasters[r.group] != f.masters;
              });
          if (rule == overlay.end())
            continue;
          moveFlow(tileId, key);
          if (!planFailure)
            planFailure = llvm::formatv(
                "at tile ({0}, {1}), the packet rule (mask 0x{2:X-}, id "
                "0x{3:X-}) of the prioritized flows (the control overlay) on "
                "{4} also matches packet id 0x{5:X-}, and a control-packet "
                "reload keeps their rules; use an id the rule does not match, "
                "or route the flow apart.",
                tileId.col, tileId.row, rule->mask, rule->value,
                describePort(slave), f.id);
          break;
        }
      }
    }
    for (Port slave : slaves) {
      size_t needed = rulesNeeded(slave);
      if (needed <= targetModel.getNumSlaveSlots())
        continue;
      for (const auto &[key, f] : byFlow)
        if (f.slave == slave)
          moveFlow(tileId, key);
      if (!splitRules(slave))
        crowdedTiles.insert(tileId);
      if (planFailure)
        continue;
      planFailure = llvm::formatv(
          "at tile ({0}, {1}), the packet flows entering on {2} need {3} "
          "packet rules, and a slave port holds {4}.",
          tileId.col, tileId.row, describePort(slave), needed,
          targetModel.getNumSlaveSlots());
    }
  }
}

void PacketFlowRouting::planTiles() {
  for (const auto &[tileId, flows] : tileFlows) {
    SmallVector<std::pair<size_t, size_t>, 4> blocking;
    if (std::optional<ArbiterPlan> plan = planTile(tileId, blocking)) {
      plans[tileId] = std::move(*plan);
      continue;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "No arbiter plan at tile (" << tileId.col << ", "
                   << tileId.row << "):\n";
      for (const SlaveFlow &f : flows) {
        llvm::dbgs() << "  " << describePort(f.slave) << " id " << f.id
                     << " ->";
        for (Port m : f.masters)
          llvm::dbgs() << ' ' << describePort(m);
        llvm::dbgs() << '\n';
      }
      for (auto [a, b] : blocking)
        llvm::dbgs() << "  blocking " << a << ' ' << b << '\n';
    });
    std::string reason;
    llvm::raw_string_ostream os(reason);
    if (auto unkept = unkeptOverlay.find(tileId);
        keepOverlay && unkept != unkeptOverlay.end()) {
      if (!planFailure) {
        overlayFailure = true;
        planFailure = "the control overlay keeps the packet rules it takes "
                      "alone, as in @ctrl_pkt_overlay, since a control-packet "
                      "reload (has_ctrl_pkt_overlay) installs them first, but "
                      "at tile (" +
                      std::to_string(tileId.col) + ", " +
                      std::to_string(tileId.row) + "), " + unkept->second;
      }
      continue;
    }
    if (blocking.empty()) {
      // Ids a source sends out by master sets that overlap tie those sets to
      // one arbiter, so its tree branches them apart to free msels.
      for (const SlaveFlow &f : flows)
        for (const SlaveFlow &g : flows)
          if (f.slave == g.slave && f.id < g.id && f.masters != g.masters &&
              llvm::any_of(
                  f.masters,
                  [&](Port m) { return llvm::is_contained(g.masters, m); })) {
            FlowKey keys[] = {{f.slave, f.id}, {g.slave, g.id}};
            splitFlow(tileId, keys[0], keys);
          }
      for (const SlaveFlow &f : flows)
        moveFlow(tileId, {f.slave, f.id});
      crowdedTiles.insert(tileId);
      if (planFailure)
        continue;
      os << "at tile (" << tileId.col << ", " << tileId.row
         << "), the packet flows need more arbiter msels than the switchbox "
            "has free.";
      planFailure = std::move(reason);
      continue;
    }
    for (auto [a, b] : blocking) {
      FlowKey keys[] = {{flows[a].slave, flows[a].id},
                        {flows[b].slave, flows[b].id}};
      for (FlowKey key : keys) {
        moveUnit(tileId, key);
        splitFlow(tileId, key, keys);
      }
    }
    if (planFailure)
      continue;
    std::optional<std::pair<size_t, size_t>> pair = conflictingStreams(
        tileId, flows[blocking.front().first], flows[blocking.front().second]);
    assert(pair && apart.empty() && offArbiter.empty() &&
           "before the hold-cycle search, only stream conflicts block a plan");
    auto [s, t] = *pair;
    os << describeStream(conflicts.getStreams()[s]) << " and "
       << describeStream(conflicts.getStreams()[t])
       << " can deadlock if they share an arbiter, and no routing found keeps "
          "them apart (last tried: tile ("
       << tileId.col << ", " << tileId.row << ")). " << conflicts.explain(s, t);
    if (keepOverlay && (flows[blocking.front().first].isCtrlPkt ||
                        flows[blocking.front().second].isCtrlPkt))
      os << " The control overlay keeps the arbiters it takes alone, as in "
            "@ctrl_pkt_overlay, since a control-packet reload "
            "(has_ctrl_pkt_overlay) installs them first.";
    planFailure = std::move(reason);
  }
}

std::optional<std::string> PacketFlowRouting::breakHoldCycles() {
  // A packet holds every arbiter it has taken until its tail passes, so waits
  // chain from switchbox to switchbox, and planning each alone can close a
  // cycle of them. Breaking any one shared arbiter of such a cycle breaks it,
  // so the search tries each in turn, up to a budget.
  auto arbitrate = [&]() {
    for (size_t s = 0; s < conflicts.getRequestedStreams().size(); s++) {
      if (!conflicts.getStreams()[s].packetID)
        continue;
      for (StreamHop &hop : routes[s]) {
        hop.arbiter.reset();
        auto plan = plans.find(hop.tile);
        if (plan == plans.end())
          continue;
        auto amsel = plan->second.slaveAmsels.find(
            {hop.input, *conflicts.getStreams()[s].packetID});
        if (amsel != plan->second.slaveAmsels.end())
          hop.arbiter = amsel->second % numArbiters;
      }
    }
  };
  ArrayRef<RoutedStream> streams = conflicts.getStreams();
  size_t numRequested = conflicts.getRequestedStreams().size();
  // Every cycle the search meets: the routing moves the flows of each, as the
  // first may run only through flows it cannot move.
  SmallVector<HoldCycle, 2> cycles;
  int replans = 0;
  bool forcedWaits = false, definite = false;
  auto search = [&](auto &self) -> bool {
    arbitrate();
    std::optional<HoldCycle> cycle =
        conflicts.holdCycle(routes, forcedWaits, definite);
    // Waits `definite` drops still count where no receiver is shared, so a
    // replan must not close a cycle the first search broke.
    if (!cycle && definite)
      cycle = conflicts.holdCycle(routes);
    if (!cycle)
      return true;
    cycles.push_back(*cycle);
    LLVM_DEBUG(llvm::dbgs()
               << "Hold cycle: " << conflicts.explain(*cycle) << '\n');
    for (const HoldCycle::Step &step : cycle->steps) {
      if (step.wait != HoldCycle::Wait::Arbiter || step.forced)
        continue;
      FlowKey sharer{step.sharerInput, *streams[step.sharer].packetID};
      FlowKey holder{step.holderInput, *streams[step.holding].packetID};
      bool sharerNew = step.sharer < numRequested;
      bool holderNew = step.holding < numRequested;
      std::optional<std::tuple<TileID, FlowKey, FlowKey>> pair;
      std::optional<std::tuple<TileID, FlowKey, int>> off;
      if (sharerNew && holderNew)
        pair = {step.tile, std::min(sharer, holder), std::max(sharer, holder)};
      else if (sharerNew || holderNew)
        off = {step.tile, sharerNew ? sharer : holder, step.arbiter};
      else
        continue;
      if (replans == maxHoldCycleReplans)
        return false;
      if ((pair && !apart.insert(*pair).second) ||
          (off && !offArbiter.insert(*off).second))
        continue;
      replans++;
      ArbiterPlan saved = plans.at(step.tile);
      SmallVector<std::pair<size_t, size_t>, 4> blocking;
      if (std::optional<ArbiterPlan> plan = planTile(step.tile, blocking)) {
        plans[step.tile] = std::move(*plan);
        if (self(self))
          return true;
        plans[step.tile] = std::move(saved);
      }
      if (pair)
        apart.erase(*pair);
      if (off)
        offArbiter.erase(*off);
    }
    return false;
  };
  // The routing moves the flows of each cycle the search met, but for those
  // waiting at a receiver they share.
  auto move = [&] {
    arbitrate();
    for (const HoldCycle &cycle : cycles)
      for (const HoldCycle::Step &step : cycle.steps) {
        if (step.wait == HoldCycle::Wait::Arbiter && step.forced &&
            step.sharer < numRequested && step.holding < numRequested &&
            streams[step.sharer].packetID && streams[step.holding].packetID) {
          PathEndPoint a{streams[step.sharer].src.tile,
                         streams[step.sharer].src.port},
              b{streams[step.holding].src.tile, streams[step.holding].src.port};
          if (!(a == b))
            hazardTogether.insert(std::minmax(a, b));
        }
        if (step.wait != HoldCycle::Wait::Drain && !step.forced) {
          SmallVector<FlowKey, 3> keys;
          for (auto [s, input] : {std::pair{step.waiting, step.sharerInput},
                                  std::pair{step.sharer, step.sharerInput},
                                  std::pair{step.holding, step.holderInput}})
            if (std::optional<int> id = streams[s].packetID;
                s < numRequested && id)
              keys.push_back({input, *id});
          for (FlowKey key : keys) {
            moveUnit(step.tile, key);
            splitFlow(step.tile, key, keys);
          }
          if (step.sharer >= numRequested || step.holding >= numRequested ||
              !streams[step.sharer].packetID || !streams[step.holding].packetID)
            continue;
          PathEndPoint a{streams[step.sharer].src.tile,
                         streams[step.sharer].src.port},
              b{streams[step.holding].src.tile, streams[step.holding].src.port};
          if (!(a == b))
            hazardApart.insert(std::minmax(a, b));
        }
      }
  };
  if (!search(search)) {
    move();
    return "packet flows can deadlock holding arbiters across "
           "switchboxes, and no arbiter assignment found avoids it. " +
           conflicts.explain(cycles.front());
  }
  // Trees into a receiver they share wait on each other at its master port
  // whatever the plan, so a cycle through such waits can only be broken
  // elsewhere along it. Where none of it can, the routing fails, unless some
  // plan has such cycles only by assumption.
  cycles.clear();
  replans = 0;
  forcedWaits = true;
  if (!search(search)) {
    LLVM_DEBUG(llvm::dbgs() << "Hold cycle at a shared receiver: "
                            << conflicts.explain(cycles.front()) << '\n');
    cycles.clear();
    replans = 0;
    definite = true;
    if (!search(search)) {
      move();
      sharedReceiverCycle = cycles.front();
    }
  }
  return std::nullopt;
}

LogicalResult PacketPlan::emit(DeviceOp device, OpBuilder &builder,
                               DynamicTileAnalysis &analyzer) const {
  const AIETargetModel &targetModel = device.getTargetModel();
  const int numArbiters = targetModel.getNumArbiters();
  const int numMselsPerArbiter = targetModel.getNumMselsPerArbiter();
  const uint32_t maxPacketId = targetModel.getMaxPacketId();
  const int idBits = llvm::Log2_32_Ceil(maxPacketId + 1);
  const int idMask = (1 << idBits) - 1;
  AIEDialect::IsCtrlPktOverlayAttrHelper overlay(builder.getContext());
  AIEDialect::PriorityRouteAttrHelper priorityRoute(builder.getContext());
  UnitAttr unit = builder.getUnitAttr();

  // A master port can only be associated with one arbiter, and each arbiter
  // has numMselsPerArbiter msels, so a tile has numArbiters x
  // numMselsPerArbiter "logical" arbiters.
  //
  // A map from Tile and master selectValue to the ports targetted by that
  // master select.
  std::map<std::pair<TileID, int>, SmallVector<Port, 4>> masterAMSels;
  DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
  for (const auto &[tileId, plan] : plans) {
    for (const auto &[amsel, masters] : plan.amselMasters)
      masterAMSels[{tileId, amsel}].append(masters.begin(), masters.end());
    for (const auto &[flow, amsel] : plan.slaveAmsels)
      slaveAMSels[{{tileId, flow.first}, flow.second}] = amsel;
  }

  // Compute the master set IDs
  // A map from a switchbox output port to its associated amsel values
  std::map<PhysPort, SmallVector<int, 4>> mastersets;
  for (const auto &[tileAmsel, ports] : masterAMSels)
    for (Port port : ports)
      mastersets[{tileAmsel.first, port}].push_back(tileAmsel.second);

  LLVM_DEBUG({
    llvm::dbgs() << "CHECK mastersets\n";
    for (const auto &[master, values] : mastersets) {
      llvm::dbgs() << "master " << master.first << " "
                   << stringifyWireBundle(master.second.bundle) << " : "
                   << master.second.channel << '\n';
      for (int value : values)
        llvm::dbgs() << "amsel: " << value << '\n';
    }
  });

  // Overlay flows group apart from the others; the ids of a group share its
  // packet rules.
  SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
  SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
  while (!workList.empty()) {
    std::pair<PhysPort, int> slave = workList.pop_back_val();
    const SmallVector<PhysPort, 4> &dests = packetFlows.at(slave);
    auto *group = llvm::find_if(slaveGroups, [&](const auto &candidate) {
      const SmallVector<PhysPort, 4> &groupDests =
          packetFlows.at(candidate.front());
      return candidate.front().first.second == slave.first.second &&
             overlayFlows.contains(candidate.front()) ==
                 overlayFlows.contains(slave) &&
             groupDests.size() == dests.size() &&
             llvm::all_of(dests, [&](const PhysPort &dest) {
               return llvm::is_contained(groupDests, dest);
             });
    });
    if (group != slaveGroups.end())
      group->push_back(slave);
    else
      slaveGroups.push_back({slave});
  }

  // What each group claims on its slave port, as cubes, split into the rules
  // its flows state and the ids left for the cover to describe.
  SmallVector<SmallVector<std::pair<int, int>, 4>, 4> statedRules(
      slaveGroups.size());
  SmallVector<SmallVector<int, 4>, 4> derivedIds(slaveGroups.size());
  for (size_t gi = 0; gi < slaveGroups.size(); ++gi) {
    for (auto member : slaveGroups[gi]) {
      auto cubes = statedCubes(member, idMask);
      if (cubes.empty()) {
        derivedIds[gi].push_back(member.second);
        continue;
      }
      for (auto cube : cubes)
        if (!llvm::is_contained(statedRules[gi], cube))
          statedRules[gi].push_back(cube);
    }
  }

  // Everything a group claims, for the other groups on its port to avoid.
  [[maybe_unused]] auto claimsOf = [&](size_t gi) {
    SmallVector<std::pair<int, int>, 8> claims(statedRules[gi].begin(),
                                               statedRules[gi].end());
    for (int id : derivedIds[gi])
      claims.push_back({idMask, id});
    return claims;
  };

  // Realize the routes in MLIR

  // Update tiles map if any new tile op declaration is needed for constructing
  // the flow.
  std::map<TileID, Operation *> tiles;
  for (auto tileOp : device.getOps<TileOp>())
    tiles[tileOp.getTileID()] = tileOp;
  for (const PhysPort &master : llvm::make_first_range(mastersets))
    tiles.try_emplace(master.first, analyzer.getTile(builder, master.first));

  for (Operation *tileOp : llvm::make_second_range(tiles)) {
    TileOp tile = cast<TileOp>(tileOp);
    TileID tileId = tile.getTileID();
    Location tileLoc = tile.getLoc();

    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox =
        analyzer.getSwitchbox(builder, tile.colIndex(), tile.rowIndex());
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder, tileLoc);
    Block &b = swbox.getConnections().front();
    builder.setInsertionPoint(b.getTerminator());

    if (auto hops = circuitHops.find(tileId); hops != circuitHops.end())
      for (const Connect &hop : hops->second)
        ConnectOp::create(builder, tileLoc, hop.src.bundle, hop.src.channel,
                          hop.dst.bundle, hop.dst.channel);

    // The master sets of this tile, in port order, and the amsels they take.
    auto tileMastersets =
        llvm::make_filter_range(mastersets, [&](const auto &entry) {
          return entry.first.first == tileId;
        });
    llvm::SmallBitVector amselNeeded(numMselsPerArbiter * numArbiters);
    for (const auto &[master, amsels] : tileMastersets)
      for (int amsel : amsels)
        amselNeeded.set(amsel);
    std::map<int, AMSelOp> amselOps;
    for (int msel = 0; msel < numMselsPerArbiter; msel++)
      for (int arbiter = 0; arbiter < numArbiters; arbiter++)
        if (int amsel = arbiter + msel * numArbiters; amselNeeded.test(amsel))
          amselOps[amsel] = AMSelOp::create(builder, tileLoc, arbiter, msel);
    for (const auto &[master, msels] : tileMastersets) {
      SmallVector<Value, 4> amsels;
      for (int msel : msels)
        amsels.push_back(amselOps.at(msel));
      auto msOp = MasterSetOp::create(
          builder, tileLoc, builder.getIndexType(), master.second.bundle,
          master.second.channel, amsels, keepPktHeaderAttr.lookup(master));
      if (ctrlPktOverlayMasterPorts.contains(master))
        overlay.setAttr(msOp, unit);
    }

    // Generate the packet rules, adding to any the switchbox already has.
    DenseMap<Port, PacketRulesOp> slaveRules;
    DenseMap<Port, SmallVector<PacketRuleOp>> existingRules;
    struct PortPlan {
      SmallVector<size_t, 4> groups;
      SmallVector<PortRule> rules;
      size_t emitted = 0;
    };
    std::map<Port, PortPlan> portPlans;
    for (auto rulesOp : b.getOps<PacketRulesOp>()) {
      slaveRules[rulesOp.sourcePort()] = rulesOp;
      llvm::append_range(existingRules[rulesOp.sourcePort()],
                         rulesOp.getRules().getOps<PacketRuleOp>());
    }
    for (size_t gi = 0; gi < slaveGroups.size(); ++gi) {
      const auto &group = slaveGroups[gi];
      builder.setInsertionPoint(b.getTerminator());

      auto port = group.front().first;
      if (tileId != port.first)
        continue;

      Port slave = port.second;

      SmallVector<int, 4> matchIds =
          llvm::to_vector<4>(llvm::make_second_range(group));
      for (int id : matchIds)
        if (id > static_cast<int>(maxPacketId))
          return mlir::emitError(tileLoc)
                 << "packet id " << id << " exceeds the maximum of "
                 << maxPacketId;

      // Rules the switchbox already has, e.g. hand-authored, match first.
      for (PacketRuleOp rule : existingRules.lookup(slave))
        for (int id : matchIds)
          if ((id & rule.maskInt()) == rule.valueInt()) {
            rule->emitOpError("can lead to false packet id match for id ")
                << id << ", which is not supposed to pass through this port.";
            rule->emitRemark("Please consider changing all uses of packet id ")
                << id << " to avoid deadlock.";
            return failure();
          }

      // Groups on one slave port carry different destination sets, so
      // checkRules let no two of them claim the same id.
      assert(llvm::all_of(
                 llvm::seq<size_t>(0, slaveGroups.size()),
                 [&](size_t oi) {
                   return oi == gi || slaveGroups[oi].front().first != port ||
                          llvm::none_of(claimsOf(gi), [&](auto own) {
                            return llvm::any_of(claimsOf(oi), [&](auto other) {
                              return cubesIntersect(own, other);
                            });
                          });
                 }) &&
             "groups on one slave port claim the same id");

      // The rules of the slave port, planned at its first group, go out in
      // order: each group adds those up to its last.
      auto [planIt, fresh] = portPlans.try_emplace(slave);
      PortPlan &plan = planIt->second;
      if (fresh) {
        SmallVector<GroupClaims> claims;
        for (size_t oi = 0; oi < slaveGroups.size(); ++oi) {
          if (slaveGroups[oi].front().first != port)
            continue;
          plan.groups.push_back(oi);
          claims.push_back({statedRules[oi], derivedIds[oi],
                            overlayFlows.contains(slaveGroups[oi].front())});
        }
        SmallVector<std::pair<int, int>> existing;
        for (PacketRuleOp rule : existingRules.lookup(slave))
          existing.push_back({rule.maskInt(), rule.valueInt()});
        plan.rules = slavePortRules(claims, existing, idBits,
                                    targetModel.getNumSlaveSlots());
        LLVM_DEBUG({
          llvm::dbgs() << "packet rules " << describePort(slave) << ":";
          for (const PortRule &r : plan.rules)
            llvm::dbgs() << " rule(" << r.mask << ", " << r.value
                         << ") -> group " << plan.groups[r.group];
          llvm::dbgs() << '\n';
        });
      }

      // Every id the group claims takes a rule to its amsel first: its own,
      // or one of the control overlay's for the same destinations.
      assert(
          llvm::all_of(
              llvm::seq(0, idMask + 1),
              [&](int id) {
                bool own = llvm::is_contained(derivedIds[gi], id) ||
                           llvm::any_of(statedRules[gi], [&](auto c) {
                             return (id & c.first) == (c.second & c.first);
                           });
                const auto *first =
                    llvm::find_if(plan.rules, [&](const PortRule &r) {
                      return (id & r.mask) == r.value;
                    });
                return !own ||
                       (first != plan.rules.end() &&
                        slaveAMSels.at(
                            slaveGroups[plan.groups[first->group]].front()) ==
                            slaveAMSels.at(slaveGroups[gi].front()));
              }) &&
          "packet rules send a claimed id elsewhere");

      size_t last = plan.emitted;
      for (size_t r = plan.emitted; r < plan.rules.size(); ++r)
        if (plan.groups[plan.rules[r].group] == gi)
          last = r + 1;

      PacketRulesOp packetrules = slaveRules.lookup(slave);
      if (!packetrules) {
        packetrules = PacketRulesOp::create(builder, tileLoc, slave.bundle,
                                            slave.channel);
        PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                        tileLoc);
        slaveRules[slave] = packetrules;
      } else {
        // After the amsels its new rules use.
        packetrules->moveBefore(b.getTerminator());
      }

      Block &rules = packetrules.getRules().front();

      assert(llvm::range_size(rules.getOps<PacketRuleOp>()) +
                     (last - plan.emitted) <=
                 targetModel.getNumSlaveSlots() &&
             "checkRules lets a slave port take more rules than it holds");

      builder.setInsertionPoint(rules.getTerminator());
      for (; plan.emitted < last; ++plan.emitted) {
        const PortRule &r = plan.rules[plan.emitted];
        const auto &ruleGroup = slaveGroups[plan.groups[r.group]];
        auto rule = PacketRuleOp::create(
            builder, tileLoc, r.mask, r.value,
            amselOps.at(slaveAMSels.at(ruleGroup.front())));
        if (llvm::all_of(ruleGroup, [&](const auto &member) {
              return markedFlows.contains(member);
            }))
          overlay.setAttr(rule, unit);
        if (prioritizedSourcePorts.contains(port) &&
            llvm::any_of(ruleGroup, [&](const auto &member) {
              return prioritizedFlows.contains(member);
            }))
          priorityRoute.setAttr(rule, unit);
      }
    }

    // A reload skips the overlay's rules. A port of its rules alone says so
    // once; a port it shares says so rule by rule.
    for (auto rulesOp : b.getOps<PacketRulesOp>()) {
      auto rules = rulesOp.getRules().getOps<PacketRuleOp>();
      bool blockTagged = overlay.isAttrPresent(rulesOp);
      auto tagged = [&](PacketRuleOp rule) {
        return blockTagged || overlay.isAttrPresent(rule);
      };
      bool all = llvm::all_of(rules, tagged);
      for (PacketRuleOp rule : rules) {
        if (!all && tagged(rule))
          overlay.setAttr(rule, unit);
        else
          rule->removeDiscardableAttr(overlay.getName());
      }
      if (all && !rules.empty())
        overlay.setAttr(rulesOp, unit);
      else
        rulesOp->removeDiscardableAttr(overlay.getName());
    }
  }
  return success();
}

LogicalResult AIEPathfinderPass::runOnPacketFlow(
    DeviceOp device, OpBuilder &builder, DynamicTileAnalysis &analyzer,
    const StreamConflicts &conflicts, const PacketPlan &plan, bool warn) {
  if (warn && plan.sharedReceiverCycle)
    emitWarning(device.getLoc(), sharedReceiverCycleReason)
        << conflicts.explain(*plan.sharedReceiverCycle);
  if (failed(plan.emit(device, builder, analyzer)))
    return failure();
  lowerShimDMAPorts(device, builder, analyzer);
  for (PacketFlowOp flow :
       llvm::make_early_inc_range(device.getOps<PacketFlowOp>()))
    flow.erase();
  return success();
}

// A copy of `d` in a module like its own, as the analyses look up through it.
static DeviceOp cloneInScratch(DeviceOp d, OwningOpRef<ModuleOp> &scratch) {
  scratch = ModuleOp::create(d.getLoc());
  if (Operation *parent = d->getParentOp())
    (*scratch)->setAttrs(parent->getAttrDictionary());
  DeviceOp copy = d.clone();
  scratch->push_back(copy);
  return copy;
}

// The router names a shim DMA by its own port, and the end of
// runOnPacketFlow moves it behind the shim mux onto a South channel. Rules and
// master sets a previous run left there are moved back first, so new flows on
// the same DMA channel share them.
static void unmuxShimDMAPacketPorts(DeviceOp device) {
  for (auto shimMux : device.getOps<ShimMuxOp>()) {
    auto hasConnect = [&](Port src, Port dst) {
      return llvm::any_of(shimMux.getConnections().getOps<ConnectOp>(),
                          [&](ConnectOp c) {
                            return c.sourcePort() == src && c.destPort() == dst;
                          });
    };
    for (auto switchbox : device.getOps<SwitchboxOp>()) {
      if (switchbox.getTileOp() != shimMux.getTileOp())
        continue;
      for (auto rules : switchbox.getConnections().getOps<PacketRulesOp>())
        for (int ch : {0, 1}) {
          Port dma{WireBundle::DMA, ch};
          Port north{WireBundle::North, shimMuxChannelFrom(dma)};
          if (rules.sourcePort() == Port{WireBundle::South, north.channel} &&
              hasConnect(dma, north)) {
            rules.setSourceBundle(WireBundle::DMA);
            rules.setSourceChannel(ch);
            break;
          }
        }
      for (auto masterSet : switchbox.getConnections().getOps<MasterSetOp>())
        for (int ch : {0, 1}) {
          Port dma{WireBundle::DMA, ch};
          Port north{WireBundle::North, shimMuxChannelTo(dma)};
          if (masterSet.destPort() == Port{WireBundle::South, north.channel} &&
              hasConnect(north, dma)) {
            masterSet.setDestBundle(WireBundle::DMA);
            masterSet.setDestChannel(ch);
            break;
          }
        }
    }
  }
}

llvm::Expected<PacketPlan> AIEPathfinderPass::route(
    DeviceOp d, DynamicTileAnalysis &analyzer, const StreamConflicts &conflicts,
    const PinnedOverlay &pinned, bool circuitSwitchHops, bool prioritize) {
  // Packet flows that can deadlock must not share an arbiter. Every routing
  // the router finds is checked by planning the arbiters on it, and one that
  // cannot be planned counts as illegal, so routing and allocation agree.
  std::map<PathEndPoint, SmallVector<size_t, 4>> streamsFrom;
  PacketConstraints constraints;
  constraints.pinned = pinned.trees;
  constraints.prioritize = prioritize;
  if (clRoutePacket && !d.getOps<PacketFlowOp>().empty()) {
    const AIETargetModel &targetModel = d.getTargetModel();
    if (llvm::Error err = unroutableArbiters(
            d, conflicts,
            [&](TileID tile) {
              return !circuitSwitchHops ||
                     targetModel.isShimNOCorPLTile(tile.col, tile.row);
            },
            pinned.trees))
      return std::move(err);
    for (auto [i, s] : llvm::enumerate(conflicts.getRequestedStreams()))
      if (s.packetID)
        streamsFrom[{s.src.tile, s.src.port}].push_back(i);
    constraints.conflict = [&](const PathEndPoint &a, const std::set<int> &aIds,
                               const PathEndPoint &b,
                               const std::set<int> &bIds) {
      auto as = streamsFrom.find(a), bs = streamsFrom.find(b);
      if (as == streamsFrom.end() || bs == streamsFrom.end())
        return false;
      ArrayRef<RoutedStream> streams = conflicts.getStreams();
      for (size_t s : as->second)
        for (size_t t : bs->second)
          if (aIds.count(*streams[s].packetID) &&
              bIds.count(*streams[t].packetID) && conflicts.conflict(s, t))
            return true;
      return false;
    };
    constraints.check = [&](const Routing &routing) -> llvm::Error {
      PacketFlowRouting check(d, routing, conflicts, pinned, clRouteCircuit,
                              circuitSwitchHops, prioritize);
      llvm::Expected<PacketPlan> plan = check.plan();
      if (!plan)
        return plan.takeError();
      RoutingFaults all = check.faults(), faults;
      faults.connections = std::move(all.connections);
      faults.apart = std::move(all.apart);
      faults.together = std::move(all.together);
      if (!plan->sharedReceiverCycle ||
          (faults.connections.empty() && faults.apart.empty() &&
           faults.together.empty()))
        return llvm::Error::success();
      return llvm::make_error<DeadlockProneRouting>(
          conflicts.explain(*plan->sharedReceiverCycle), std::move(faults));
    };
  }
  analyzer.pathfinder.setPacketConstraints(std::move(constraints));
  llvm::Error routed = analyzer.runAnalysis(d);
  analyzer.pathfinder.setPacketConstraints({});
  if (routed)
    return std::move(routed);
  if (!clRoutePacket)
    return PacketPlan();
  PacketFlowRouting found(d, analyzer.routing, conflicts, pinned,
                          clRouteCircuit, circuitSwitchHops, prioritize);
  llvm::Expected<PacketPlan> plan = found.plan();
  if (found.incomplete) {
    llvm::consumeError(plan.takeError());
    return llvm::make_error<RoutingFailure>(std::move(found.incomplete->second),
                                            RoutingFaults{},
                                            found.incomplete->first);
  }
  // The routing found may still leave a hold cycle through receivers flows
  // share, as the check lets one through where it names nothing to move.
  if (!plan || clAllowDeadlockProne)
    return plan;
  const std::optional<HoldCycle> &cycle = plan->sharedReceiverCycle;
  if (!cycle)
    return plan;
  return llvm::make_error<DeadlockProneRouting>(
      (llvm::Twine(sharedReceiverCycleReason) + conflicts.explain(*cycle) +
       allowDeadlockProneHint)
          .str());
}

void AIEPathfinderPass::runOnOperation() {

  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(verifyDMAChannelsResolved(d)))
    return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
  if (clRoutePacket)
    unmuxShimDMAPacketPorts(d);

  StreamConflicts conflicts(d);
  if (auto pairs = conflicts.unavoidable(); !pairs.empty()) {
    InFlightDiagnostic diag =
        (clAllowDeadlockProne ? emitWarning(d.getLoc()) : emitError(d.getLoc()))
        << "Flows can deadlock however they are routed: "
        << conflicts.explain(pairs[0].first, pairs[0].second);
    if (pairs.size() > 1)
      diag << " So can " << pairs.size() - 1 << " other pair"
           << (pairs.size() > 2 ? "s" : "") << " of flows.";
    if (!clAllowDeadlockProne) {
      diag << allowDeadlockProneHint;
      return signalPassFailure();
    }
  }
  DynamicTileAnalysis analyzer;
  // A prioritized flow keeps the route it takes alone, so route the
  // prioritized flows without the rest of the design first, and pin their
  // trees for the rest to route around. Packets sharing a source and id with
  // a prioritized flow go with it (see prioritizedFlows); other packets from
  // their sources route freely.
  auto sourcesOf = [](PacketFlowOp flow) {
    return llvm::map_range(
        flow.getPorts().getOps<PacketSourceOp>(), [](PacketSourceOp src) {
          return PathEndPoint{
              cast<TileOp>(src.getTile().getDefiningOp()).getTileID(),
              src.port()};
        });
  };
  std::set<PathEndPoint> prioritized;
  std::set<std::pair<PathEndPoint, int>> prioritizedIds;
  for (PacketFlowOp flow : d.getOps<PacketFlowOp>())
    if (flow.getPriorityRoute().value_or(false))
      for (PathEndPoint src : sourcesOf(flow)) {
        prioritized.insert(src);
        prioritizedIds.insert({src, flow.IDInt()});
      }
  auto isPrioritized = [&](PacketFlowOp flow) {
    return llvm::any_of(sourcesOf(flow), [&](const PathEndPoint &src) {
      return prioritizedIds.count({src, flow.IDInt()}) > 0;
    });
  };
  // The last flow to end at a port sets its keep_pkt_header; a reload keeps
  // the overlay's.
  bool reload = reloadsOverlay(d);
  std::map<PathEndPoint, std::pair<PacketFlowOp, PacketFlowOp>> lastTo;
  for (PacketFlowOp flow : d.getOps<PacketFlowOp>())
    for (PacketDestOp dst : flow.getPorts().getOps<PacketDestOp>()) {
      PathEndPoint port{cast<TileOp>(dst.getTile().getDefiningOp()).getTileID(),
                        dst.port()};
      auto &[prio, last] = lastTo[port];
      last = flow;
      if (reload && isPrioritized(flow) && !reloadConfigures(flow))
        prio = flow;
    }
  for (const auto &[dst, flows] : lastTo) {
    auto [prio, last] = flows;
    if (!prio ||
        keepsPktHeader(dst.coords, dst.port, prio.getKeepPktHeader()) ==
            keepsPktHeader(dst.coords, dst.port, last.getKeepPktHeader()))
      continue;
    last.emitError() << "packet flow " << last.IDInt()
                     << " is the last to end at "
                     << describeTilePort(dst.coords, dst.port)
                     << ", so it sets whether packets keep their header "
                        "there, but the prioritized flows (the control "
                        "overlay) ending there set it otherwise, and a "
                        "control-packet reload keeps their setting.";
    return signalPassFailure();
  }
  // The prioritized flows routed without the rest of the design. A reload
  // installs them as @ctrl_pkt_overlay first, so they keep that route;
  // otherwise they keep it, if they have one, where the other flows route
  // around it.
  bool pinned = false;
  PinnedOverlay overlay;
  if (clRoutePacket && !prioritized.empty() &&
      (!d.getOps<FlowOp>().empty() ||
       !llvm::all_of(d.getOps<PacketFlowOp>(), isPrioritized))) {
    OwningOpRef<ModuleOp> scratch;
    // @ctrl_pkt_overlay holds the overlay's flows and the tiles alone, without
    // the design's own switchboxes. Otherwise the prioritized flows route
    // around those.
    DeviceOp alone = cloneInScratch(d, scratch);
    SmallVector<Operation *> others;
    for (Operation &op : alone.getBody()->without_terminator()) {
      auto flow = dyn_cast<PacketFlowOp>(op);
      if (flow ? !isPrioritized(flow) || reloadConfigures(flow)
               : isa<FlowOp>(op) || (reload && !isa<TileOp>(op)))
        others.push_back(&op);
    }
    for (Operation *op : others)
      op->dropAllReferences();
    for (Operation *op : others)
      op->erase();
    DynamicTileAnalysis aloneAnalyzer;
    StreamConflicts aloneConflicts(alone);
    llvm::Expected<PacketPlan> alonePlan =
        route(alone, aloneAnalyzer, aloneConflicts, {},
              /*circuitSwitchHops=*/false);
    if (!alonePlan) {
      if (reload) {
        llvm::handleAllErrors(alonePlan.takeError(), [&](RoutingFailure &f) {
          f.reason = "prioritized packet flows (priority_route) in a design a "
                     "control-packet reload configures (has_ctrl_pkt_overlay) "
                     "keep the route they take alone, as in @ctrl_pkt_overlay, "
                     "and alone they have none" +
                     (f.reason.empty() ? "." : ": " + f.reason);
          emitError(f.loc.value_or(d.getLoc())) << f.message();
        });
        return signalPassFailure();
      }
      llvm::consumeError(alonePlan.takeError());
    } else {
      // The design's own routing keeps these trees, so it warns of any hold
      // cycle they leave.
      OpBuilder aloneBuilder = OpBuilder::atBlockTerminator(alone.getBody());
      if (failed(runOnPacketFlow(alone, aloneBuilder, aloneAnalyzer,
                                 aloneConflicts, *alonePlan, /*warn=*/false)))
        return signalPassFailure();
      overlay =
          PinnedOverlay{aloneAnalyzer.routing.packetTrees, overlayRules(alone)};
      pinned = true;
    }
  }
  // If routing fails, the router relaxes how it routes packet flows (see
  // Pathfinder::relax) and tries again. The error is the first attempt's, or
  // the first to find a routing allow-deadlock-prone would take, as that is
  // all that keeps the design from routing.
  auto routeRelaxing = [&](DeviceOp dev, DynamicTileAnalysis &an,
                           const StreamConflicts &c, const PinnedOverlay &pin,
                           bool prioritize =
                               true) -> llvm::Expected<PacketPlan> {
    llvm::Expected<PacketPlan> plan =
        route(dev, an, c, pin, clCircuitSwitchHops, prioritize);
    if (plan)
      return plan;
    llvm::Error first = plan.takeError();
    while (an.pathfinder.relax()) {
      plan = route(dev, an, c, pin, clCircuitSwitchHops, prioritize);
      if (plan) {
        llvm::consumeError(std::move(first));
        return plan;
      }
      llvm::Error err = plan.takeError();
      if (err.isA<DeadlockProneRouting>() && !first.isA<DeadlockProneRouting>())
        std::swap(first, err);
      llvm::consumeError(std::move(err));
    }
    return std::move(first);
  };
  llvm::Expected<PacketPlan> plan =
      routeRelaxing(d, analyzer, conflicts, overlay);
  // If routing fails with the prioritized flows first, they route like the
  // others, but for the overlay a reload keeps. Without a reload the error is
  // that routing's, unless only the first found one allow-deadlock-prone takes.
  if (!plan && !prioritized.empty()) {
    llvm::Error err = plan.takeError();
    if (!reload)
      overlay = {};
    DynamicTileAnalysis freeAnalyzer;
    plan = routeRelaxing(d, freeAnalyzer, conflicts, overlay,
                         /*prioritize=*/false);
    if (plan) {
      llvm::consumeError(std::move(err));
      analyzer = std::move(freeAnalyzer);
    } else {
      llvm::Error freeErr = plan.takeError();
      if (!reload && (freeErr.isA<DeadlockProneRouting>() ||
                      !err.isA<DeadlockProneRouting>()))
        std::swap(err, freeErr);
      llvm::consumeError(std::move(freeErr));
      plan = std::move(err);
    }
    pinned &= reload;
  }
  if (!plan) {
    llvm::handleAllErrors(plan.takeError(), [&](RoutingFailure &f) {
      // Say whether the pinned trees are what stands in the way: the design
      // routes if they may move.
      if (pinned && !f.reason.empty() &&
          !f.isA(PinnedRoutingFailure::classID())) {
        OwningOpRef<ModuleOp> scratch;
        DeviceOp free = cloneInScratch(d, scratch);
        DynamicTileAnalysis freeAnalyzer;
        StreamConflicts freeConflicts(free);
        llvm::Expected<PacketPlan> freePlan = routeRelaxing(
            free, freeAnalyzer, freeConflicts, {}, /*prioritize=*/false);
        if (!freePlan) {
          llvm::consumeError(freePlan.takeError());
        } else {
          f.reason = describePrioritized(llvm::to_vector(llvm::map_range(
                         prioritized,
                         [](const PathEndPoint &src) {
                           return describeTilePort(src.coords, src.port);
                         }))) +
                     ", and the other flows route only if it moves. Around "
                     "it, " +
                     f.reason;
        }
      }
      emitError(f.loc.value_or(d.getLoc())) << f.message();
    });
    signalPassFailure();
    return;
  }

  if (clRouteCircuit && failed(runOnFlow(d, analyzer))) {
    signalPassFailure();
    return;
  }
  if (clRoutePacket && failed(runOnPacketFlow(d, builder, analyzer, conflicts,
                                              *plan, /*warn=*/true))) {
    signalPassFailure();
    return;
  }
  if (pinned && reload) {
    OverlayRules rules = overlayRules(d);
    for (const auto &[port, portRules] : overlay.rules)
      rules.try_emplace(port);
    for (const auto &[port, portRules] : rules)
      if (portRules != overlay.rules[port]) {
        emitError(d.getLoc())
            << "the control overlay takes other packet rules at "
            << describeTilePort(port.first, port.second)
            << " than it takes alone, as in @ctrl_pkt_overlay, which a "
               "control-packet reload (has_ctrl_pkt_overlay) installs first.";
        signalPassFailure();
        return;
      }
  }

  // Populate wires between switchboxes and tiles. Re-running the pass on
  // already-routed IR must not duplicate the wires a previous run emitted, so
  // key the wires already present and emit only the missing ones.
  builder.setInsertionPoint(d.getBody()->getTerminator());
  llvm::DenseSet<std::tuple<Value, int, Value, int>> existingWires;
  // aie.wire is a physical connection, so the IR may spell one adjacency in
  // either operand order.
  auto recordWire = [&](Value source, int sourceBundle, Value dest,
                        int destBundle) {
    bool absent =
        !existingWires.contains({source, sourceBundle, dest, destBundle}) &&
        !existingWires.contains({dest, destBundle, source, sourceBundle});
    existingWires.insert({source, sourceBundle, dest, destBundle});
    existingWires.insert({dest, destBundle, source, sourceBundle});
    return absent;
  };
  for (auto wire : d.getOps<WireOp>()) {
    recordWire(wire.getSource(), static_cast<int>(wire.getSourceBundle()),
               wire.getDest(), static_cast<int>(wire.getDestBundle()));
  }
  auto wire = [&](Location loc, Value source, WireBundle sourceBundle,
                  Value dest, WireBundle destBundle) {
    if (recordWire(source, static_cast<int>(sourceBundle), dest,
                   static_cast<int>(destBundle))) {
      WireOp::create(builder, loc, source, sourceBundle, dest, destBundle);
    }
  };
  for (int col = 0; col <= analyzer.getMaxCol(); col++) {
    for (int row = 0; row <= analyzer.getMaxRow(); row++) {
      TileOp tile = analyzer.lookupTile({col, row});
      SwitchboxOp sw = analyzer.lookupSwitchbox({col, row});
      if (!tile || !sw)
        continue;
      Location loc = tile.getLoc();
      // connections east-west between stream switches
      if (col > 0)
        if (SwitchboxOp westsw = analyzer.lookupSwitchbox({col - 1, row}))
          wire(loc, westsw, WireBundle::East, sw, WireBundle::West);
      if (row > 0) {
        // connections between abstract 'core' of tile
        wire(loc, tile, WireBundle::Core, sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        wire(loc, tile, WireBundle::DMA, sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (SwitchboxOp southsw = analyzer.lookupSwitchbox({col, row - 1}))
          wire(loc, southsw, WireBundle::North, sw, WireBundle::South);
      } else if (tile.isShimNOCTile()) {
        if (ShimMuxOp shimsw = analyzer.lookupShimMux(col)) {
          wire(loc, shimsw,
               WireBundle::North, // Changed to connect into the north
               sw, WireBundle::South);
          // abstract 'DMA' connection on tile is attached to shim mux ( in
          // row 0 )
          wire(loc, tile, WireBundle::DMA, shimsw, WireBundle::DMA);
        }
      }
    }
  }
}

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEPathfinderPass(const AIERoutePathfinderFlowsOptions &options) {
  return std::make_unique<AIEPathfinderPass>(options);
}
