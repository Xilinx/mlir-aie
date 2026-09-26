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
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <queue>
#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-create-pathfinder-flows"

namespace {
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

#ifndef NDEBUG
    auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    auto dstBundle = flowOp.getDestBundle();
    auto dstChannel = flowOp.getDestChannel();
    LLVM_DEBUG(llvm::dbgs()
               << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
               << ", " << srcCoords.row << ")" << stringifyWireBundle(srcBundle)
               << srcChannel << " -> (" << dstCoords.col << ", "
               << dstCoords.row << ")" << stringifyWireBundle(dstBundle)
               << dstChannel << "\n\t");
#endif

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    TileID srcSbId = {srcCoords.col, srcCoords.row};
    PathEndPoint srcPoint = {srcSbId, srcPort};
    if (analyzer.processedFlows[srcPoint]) {
      // This FlowOp is a broadcast sibling of a flow whose route was already
      // materialized (the analyzer merges all destinations sharing a source
      // into one net, so the first sibling emitted connections for every
      // destination). Erase it and report success so the erase is committed.
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");
      rewriter.eraseOp(Op);
      return success();
    }
    // std::map<TileID, SwitchSetting>
    SwitchSettings settings = analyzer.flowSolutions[srcPoint];
    // add connections for all the Switchboxes in SwitchSettings
    for (const auto &[tileId, setting] : settings) {
      int col = tileId.col;
      int row = tileId.row;
      SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, col, row);
      int shimCh = srcChannel;
      bool isShim = analyzer.getTile(rewriter, tileId).isShimNOCorPLTile();

      // TODO: must reserve N3, N7, S2, S3 for DMA connections
      if (isShim && tileId == srcSbId) {

        // shim DMAs at start of flows
        if (srcBundle == WireBundle::DMA)
          // must be either DMA0 -> N3 or DMA1 -> N7
          shimCh = srcChannel == 0 ? 3 : 7;
        else if (srcBundle == WireBundle::NOC)
          // must be NOC0/NOC1 -> N2/N3 or NOC2/NOC3 -> N6/N7
          shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
        else if (srcBundle == WireBundle::PLIO)
          shimCh = srcChannel;

        ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, col);
        addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                      flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
      }
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];

        // A shim's own DMA, NOC and PLIO ports reach its switchbox through
        // the shim mux, on South channels; a flow can start and end at one.
        if (isShim && tileId == srcSbId && src == srcPort)
          src = {WireBundle::South, shimCh};
        if (isShim && (dest.bundle == WireBundle::DMA ||
                       dest.bundle == WireBundle::PLIO ||
                       dest.bundle == WireBundle::NOC)) {
          int destCh = dest.channel;
          if (dest.bundle == WireBundle::DMA)
            // must be either N2 -> DMA0 or N3 -> DMA1
            destCh = dest.channel == 0 ? 2 : 3;
          else if (dest.bundle == WireBundle::NOC)
            // must be either N2/3/4/5 -> NOC0/1/2/3
            destCh = dest.channel + 2;

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

    analyzer.processedFlows[srcPoint] = true;
    rewriter.eraseOp(Op);
    return success();
  }
};

} // namespace

namespace xilinx::AIE {

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

template <typename MyOp>
struct AIEOpRemoval : OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  explicit AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

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

bool AIEPathfinderPass::findPathToDest(const SwitchSettings &settings,
                                       TileID currTile,
                                       WireBundle currDestBundle,
                                       int currDestChannel, TileID finalTile,
                                       WireBundle finalDestBundle,
                                       int finalDestChannel) {

  if ((currTile == finalTile) && (currDestBundle == finalDestBundle) &&
      (currDestChannel == finalDestChannel)) {
    return true;
  }

  std::optional<std::pair<TileID, Port>> next =
      linkedInput(currTile, {currDestBundle, currDestChannel});
  if (!next)
    return false;
  auto [neighbourTile, neighbourSource] = *next;
  WireBundle neighbourSourceBundle = neighbourSource.bundle;
  int neighbourSourceChannel = neighbourSource.channel;
  for (const auto &[sbNode, setting] : settings) {
    TileID tile = {sbNode.col, sbNode.row};
    if (tile == neighbourTile) {
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];
        if ((src.bundle == neighbourSourceBundle) &&
            (src.channel == neighbourSourceChannel)) {
          if (findPathToDest(settings, neighbourTile, dest.bundle, dest.channel,
                             finalTile, finalDestBundle, finalDestChannel)) {
            return true;
          }
        }
      }
    }
  }

  return false;
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

// The rules a group of flows entering one slave port takes: those its flows
// state, then a cover of the ids they leave to the router, which avoids those
// rules and whatever else claims ids on the port.
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
/// and the ids they leave to the router.
struct GroupClaims {
  ArrayRef<std::pair<int, int>> stated;
  ArrayRef<int> derived;
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
  std::function<bool(uint64_t, size_t)> search = [&](uint64_t taken,
                                                     size_t left) {
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
      if (llvm::any_of(tried, [&](auto t) {
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
      if (search(taken | own, left - 1))
        return true;
      rules.pop_back();
    }
    dead.insert({taken, left});
    return false;
  };
  for (size_t length = 1; length <= budget; ++length)
    if (search(taken, length))
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

namespace {
constexpr int numArbiters = 6;
constexpr int numMselsPerArbiter = 4;

/// Packets with one id entering a switchbox on one slave port, and the master
/// ports they leave by.
struct SlaveFlow {
  Port slave;
  int id;
  SmallVector<Port, 4> masters;
  bool isCtrlPkt;
};

/// The amsel each slave flow of a switchbox takes, and the master ports each
/// amsel selects.
struct ArbiterPlan {
  std::map<std::pair<Port, int>, int> slaveAmsels;
  std::map<int, SmallVector<Port, 4>> amselMasters;
};

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
planArbiters(ArrayRef<SlaveFlow> flows,
             llvm::function_ref<bool(size_t, size_t)> conflict,
             llvm::function_ref<bool(size_t, int)> excluded,
             const std::set<int> &reservedAmsels,
             SmallVectorImpl<std::pair<size_t, size_t>> &blocking) {
  auto amselOf = [](int arbiter, int msel) {
    return arbiter + msel * numArbiters;
  };

  std::map<Port, Port> leader;
  std::function<Port(Port)> find = [&](Port p) {
    auto [it, inserted] = leader.try_emplace(p, p);
    if (it->second == p)
      return p;
    return it->second = find(it->second);
  };
  for (const SlaveFlow &f : flows)
    for (Port m : f.masters)
      leader[find(m)] = find(f.masters.front());

  struct Unit {
    SmallVector<size_t, 4> flows;
    SmallVector<SmallVector<Port, 4>, 2> masterSets;
    bool isCtrlPkt = false;
  };
  std::vector<Unit> units;
  std::map<Port, size_t> unitOf;
  SmallVector<size_t, 8> flowUnit;
  for (auto [i, f] : llvm::enumerate(flows)) {
    auto [it, inserted] = unitOf.try_emplace(find(f.masters.front()), 0);
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
  for (Unit &unit : units)
    std::stable_partition(unit.masterSets.begin(), unit.masterSets.end(),
                          [&](const SmallVector<Port, 4> &masters) {
                            return llvm::any_of(unit.flows, [&](size_t f) {
                              return flows[f].isCtrlPkt &&
                                     flows[f].masters == masters;
                            });
                          });

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

  SmallVector<SmallVector<int, 4>, numArbiters> freeMsels(numArbiters);
  SmallVector<SmallVector<size_t, 4>, numArbiters> excludedUnits(numArbiters);
  for (int a = 0; a < numArbiters; a++) {
    for (int m = 0; m < numMselsPerArbiter; m++)
      if (!reservedAmsels.count(amselOf(a, m)))
        freeMsels[a].push_back(m);
    for (size_t f = 0; f < flows.size(); f++)
      if (excluded(f, a) && !llvm::is_contained(excludedUnits[a], flowUnit[f]))
        excludedUnits[a].push_back(flowUnit[f]);
  }

  // Most constrained first.
  SmallVector<size_t, 8> order(units.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    return std::make_pair(neighbors[a].size(), units[a].masterSets.size()) >
           std::make_pair(neighbors[b].size(), units[b].masterSets.size());
  });

  SmallVector<int, 8> arbiterOf(units.size(), -1);
  SmallVector<size_t, numArbiters> load(numArbiters, 0);
  int steps = 0;
  constexpr int stepBudget = 100000;
  std::function<bool(size_t)> place = [&](size_t depth) {
    if (depth == order.size())
      return true;
    if (++steps > stepBudget)
      return false;
    size_t u = order[depth];
    SmallVector<int, numArbiters> candidates(numArbiters);
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
      if (place(depth + 1))
        return true;
      load[a] -= units[u].masterSets.size();
      arbiterOf[u] = -1;
    }
    return false;
  };
  if (!place(0)) {
    blocking.append(crossPairs.begin(), crossPairs.end());
    return std::nullopt;
  }

  // Control packets take the highest msels, as they do elsewhere.
  ArbiterPlan plan;
  SmallVector<size_t, numArbiters> low(numArbiters, 0), high(numArbiters, 0);
  for (const Unit &unit : units) {
    int a = arbiterOf[&unit - units.data()];
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
  auto path = [&](std::optional<TileID> avoid) {
    std::map<TileID, TileID> via{{src, src}};
    std::queue<TileID> queue;
    queue.push(src);
    while (!queue.empty() && !via.count(dst)) {
      TileID t = queue.front();
      queue.pop();
      for (TileID n : neighbors(t))
        if (n != avoid && via.try_emplace(n, t).second)
          queue.push(n);
    }
    SmallVector<TileID> tiles;
    if (via.count(dst))
      for (TileID t = via.at(dst); t != src; t = via.at(t))
        tiles.push_back(t);
    return std::pair{via.count(dst) > 0, tiles};
  };
  auto [reachable, interior] = path(std::nullopt);
  if (!reachable)
    return {};
  llvm::erase_if(interior, [&](TileID t) { return path(t).first; });
  return interior;
}

/// Packet streams take an arbiter at the tile they end at whatever the
/// routing, and where `pinsHops` says hops cannot be circuit switched, at the
/// tile they start at and every tile each of their routes passes too. Two that
/// conflict pass any tile on different slave ports -- sharing one means they
/// merged, unsafely, upstream -- so a set of them conflicting pairwise needs
/// an arbiter apiece. Says why no routing can work if some tile has such a set
/// larger than its free arbiters.
std::optional<std::string>
unroutableArbiters(DeviceOp device, StreamConflicts &conflicts,
                   llvm::function_ref<bool(TileID)> pinsHops) {
  const AIETargetModel &targetModel = device.getTargetModel();
  std::map<TileID, SmallVector<size_t, 8>> pinned;
  for (auto [i, s] : llvm::enumerate(conflicts.getRequestedStreams())) {
    if (!s.packetID)
      continue;
    pinned[s.dst.tile].push_back(i);
    if (s.src.tile == s.dst.tile)
      continue;
    if (pinsHops(s.src.tile))
      pinned[s.src.tile].push_back(i);
    for (TileID t : cutTiles(targetModel, s.src.tile, s.dst.tile))
      if (pinsHops(t))
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
      free += llvm::any_of(llvm::seq(numMselsPerArbiter), [&](int m) {
        return !reserved.count({tileId, a + m * numArbiters});
      });
    // Pairwise conflicting streams have distinct sources and destinations.
    std::set<std::pair<TileID, Port>> srcs, dsts;
    for (size_t s : candidates) {
      srcs.insert({streams[s].src.tile, streams[s].src.port});
      dsts.insert({streams[s].dst.tile, streams[s].dst.port});
    }
    if (std::min(srcs.size(), dsts.size()) <= free)
      continue;

    SmallVector<size_t, 8> clique, best;
    int steps = 0;
    constexpr int stepBudget = 100000;
    std::function<void(ArrayRef<size_t>)> grow = [&](ArrayRef<size_t> cands) {
      if (clique.size() > best.size())
        best = clique;
      for (auto [k, s] : llvm::enumerate(cands)) {
        if (best.size() > free || ++steps > stepBudget ||
            clique.size() + cands.size() - k <= best.size())
          return;
        SmallVector<size_t, 8> next;
        for (size_t t : cands.drop_front(k + 1))
          if (conflicts.conflict(s, t))
            next.push_back(t);
        clique.push_back(s);
        grow(next);
        clique.pop_back();
      }
    };
    grow(candidates);
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

LogicalResult AIEPathfinderPass::runOnPacketFlow(
    DeviceOp device, OpBuilder &builder, DynamicTileAnalysis &analyzer,
    const std::map<PathEndPoint, SwitchSettings> &solution,
    StreamConflicts &conflicts, RoutingHazards *hazards) {

  ConversionTarget target(getContext());

  std::map<TileID, mlir::Operation *> tiles;

  // Map from a port and flowID to
  std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
  std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> ctrlPacketFlows;
  SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
  DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
  // Flag to keep packet header at packet flow destination
  DenseMap<PhysPort, BoolAttr> keepPktHeaderAttr;
  // The slave ports and IDs that carry control packets. A switchbox routes on
  // the ID alone, so a flow sharing both with a priority flow is one too.
  DenseSet<std::pair<PhysPort, int>> ctrlPktFlows;
  // Set of master ports that belong to control packet overlay flows
  DenseSet<PhysPort> ctrlPktOverlayMasterPorts;

  // Packet-rule masks the flows state, keyed by the slave port the stream
  // enters and the flow ID. One ID may reach a port under two masks, so the
  // ID alone does not identify the claim.
  std::map<std::pair<PhysPort, int>, int> pinnedMasks;

  for (auto tileOp : device.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    tiles[{col, row}] = tileOp;
  }

  const AIETargetModel &targetModel = device.getTargetModel();

  // The streams each slave flow carries, for asking whether two flows can
  // deadlock on an arbiter.
  std::map<std::tuple<TileID, Port, TileID, Port, int>, size_t>
      packetStreamIndex;
  for (auto [i, s] : llvm::enumerate(conflicts.getStreams()))
    if (s.packetID)
      packetStreamIndex.try_emplace(
          {s.src.tile, s.src.port, s.dst.tile, s.dst.port, *s.packetID}, i);
  DenseMap<std::pair<PhysPort, int>, SmallVector<size_t, 2>> slaveFlowStreams;
  // The master ports each source's packets leave a slave port by. A switchbox
  // routes on the id alone, so sources sharing an id there go everywhere any
  // of them does.
  std::map<std::pair<PhysPort, int>, std::map<PathEndPoint, std::set<Port>>>
      slaveFlowSources;
  // Every stream's hops, source first; the requested ones as this routing
  // lays them, the rest as the design already does.
  std::vector<SmallVector<StreamHop, 8>> routes;
  for (const RoutedStream &s : conflicts.getStreams())
    routes.push_back(s.hops);

  // Sources routed as circuits, which runOnFlow lowers, or has lowered.
  std::set<PathEndPoint> circuitSources;
  for (const auto &[point, processed] : analyzer.processedFlows)
    if (processed)
      circuitSources.insert(point);
  auto sourceOf = [](FlowOp flow) {
    auto tile = cast<TileOp>(flow.getSource().getDefiningOp());
    return PathEndPoint{tile.getTileID(),
                        {flow.getSourceBundle(), flow.getSourceChannel()}};
  };
  if (clRouteCircuit)
    for (FlowOp flow : device.getOps<FlowOp>())
      circuitSources.insert(sourceOf(flow));
  const SwitchSettings noSettings;
  auto settingsOf = [&](const PathEndPoint &src) -> const SwitchSettings & {
    auto it = solution.find(src);
    return it == solution.end() ? noSettings : it->second;
  };

  // The logical model of all the switchboxes.
  std::map<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
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
    if (sources.empty()) {
      if (hazards)
        continue;
      return pktFlowOp.emitOpError("packet_flow has no packet_source");
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
        const SwitchSettings &settings = settingsOf(srcPoint);
        auto stream = packetStreamIndex.find(
            {srcCoords, srcPort, destCoords, destPort, flowID});
        if (stream != packetStreamIndex.end()) {
          SmallVector<StreamHop, 8> &hops = routes[stream->second];
          hops.clear();
          std::optional<std::pair<TileID, Port>> at{{srcSB, srcPort}};
          while (at && hops.size() <= settings.size()) {
            auto [tile, input] = *at;
            hops.push_back({tile, input, std::nullopt});
            at.reset();
            auto setting = settings.find(tile);
            if (setting == settings.end())
              break;
            for (auto [src, dest] :
                 llvm::zip(setting->second.srcs, setting->second.dsts))
              if (src == input && !(tile == destCoords && dest == destPort) &&
                  findPathToDest(settings, tile, dest.bundle, dest.channel,
                                 destCoords, destPort.bundle,
                                 destPort.channel)) {
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
            if (!findPathToDest(settings, currTile, dest.bundle, dest.channel,
                                destCoords, destPort.bundle, destPort.channel))
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
            if (stream != packetStreamIndex.end() &&
                !llvm::is_contained(slaveFlowStreams[slaveFlow],
                                    stream->second))
              slaveFlowStreams[slaveFlow].push_back(stream->second);
            slaveFlowSources[slaveFlow][srcPoint].insert(dest);
            if (std::optional<uint8_t> mask = pktFlowOp.getMask()) {
              pinnedMasks[{{currTile, {src.bundle, src.channel}}, flowID}] =
                  *mask;
            }
            if (pktFlowOp.getPriorityRoute().value_or(false))
              ctrlPktFlows.insert(slaveFlow);
          }
        }
        if (!srcRouted && !hazards)
          return pktFlowOp.emitOpError()
                 << "packet flow source (" << srcCoords.col << ", "
                 << srcCoords.row << ") " << stringifyWireBundle(srcPort.bundle)
                 << srcPort.channel << " could not be routed to destination ("
                 << destCoords.col << ", " << destCoords.row << ") "
                 << stringifyWireBundle(destPort.bundle) << destPort.channel
                 << "; the pathfinder produced an incomplete routing for this "
                    "placement.";
      }
    }
  }

  // <arbiter, msel> slots (per tile) that packet-switch configuration in the
  // input IR already occupies. Kept apart from masterAMSels so the allocator
  // run does not re-emit those ops.
  std::map<TileID, std::set<int>> reservedAmsels;

  // Seed the reserved set from the packet-switch configuration the switchboxes
  // already carry, so this run allocates around it instead of over it.
  for (auto swboxOp : device.getOps<SwitchboxOp>()) {
    TileID tileId = swboxOp.getTileOp().getTileID();
    // An amsel no master set uses still has rules steering packets to it.
    for (auto amselOp : swboxOp.getConnections().getOps<AMSelOp>())
      reservedAmsels[tileId].insert(amselOp.arbiterIndex() +
                                    amselOp.getMselValue() * numArbiters);
  }

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
  std::set<PhysPort> circuitPorts;
  for (auto swbox : device.getOps<SwitchboxOp>())
    for (auto connect : swbox.getConnections().getOps<ConnectOp>()) {
      TileID tileId = swbox.getTileOp().getTileID();
      circuitPorts.insert({tileId, connect.sourcePort()});
      circuitPorts.insert({tileId, connect.destPort()});
    }
  // Circuits not yet lowered claim their ports all the same.
  if (clRouteCircuit)
    for (FlowOp flow : device.getOps<FlowOp>())
      for (const auto &[tileId, setting] : settingsOf(sourceOf(flow))) {
        for (Port p : setting.srcs)
          circuitPorts.insert({tileId, p});
        for (Port p : setting.dsts)
          circuitPorts.insert({tileId, p});
      }
  auto isDirectional = [](WireBundle bundle) {
    return bundle == WireBundle::North || bundle == WireBundle::South ||
           bundle == WireBundle::East || bundle == WireBundle::West;
  };
  for (auto &[tileId, connects] : switchboxes) {
    if (!clCircuitSwitchHops ||
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
          !circuitPorts.count({tileId, slave}) &&
          llvm::all_of(reached,
                       [&](Port m) {
                         return isDirectional(m.bundle) &&
                                !circuitPorts.count({tileId, m});
                       }) &&
          llvm::all_of(
              reachedByID,
              [&](const auto &ids) { return ids.second == reached; }) &&
          llvm::all_of(connects, [&](const auto &entry) {
            const auto &[other, otherID] = entry;
            if (!reached.count(other.dst))
              return true;
            return other.src == slave &&
                   !ctrlPktFlows.contains({{tileId, slave}, otherID});
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

  LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

  for (const auto &[tileId, connects] : switchboxes) {
    LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << tileId.col << " "
                            << tileId.row << '\n');
    for (const auto &[conn, flowID] : connects) {
      Port sourcePort = conn.src;
      Port destPort = conn.dst;
      auto sourceFlow =
          std::make_pair(std::make_pair(tileId, sourcePort), flowID);
      if (ctrlPktFlows.contains(sourceFlow)) {
        ctrlPacketFlows[sourceFlow].push_back({tileId, destPort});
        ctrlPktOverlayMasterPorts.insert({tileId, destPort});
      } else {
        packetFlows[sourceFlow].push_back({tileId, destPort});
      }
      slavePorts.push_back(sourceFlow);
      LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                              << stringifyWireBundle(sourcePort.bundle) << " "
                              << sourcePort.channel << " -> "
                              << stringifyWireBundle(destPort.bundle) << " "
                              << destPort.channel << "\n");
    }
  }

  // A master port can only be associated with one arbiter, and each arbiter
  // has four msels, so a tile has 6 x 4 "logical" arbiters.

  // A map from Tile and master selectValue to the ports targetted by that
  // master select.
  std::map<std::pair<TileID, int>, SmallVector<Port, 4>> masterAMSels;

  // Each switchbox has its arbiters planned as a whole: no two flows that can
  // deadlock enter on different slave ports and share one, and every master
  // port a flow leaves by takes that flow's arbiter. Where that is impossible
  // the routing is unusable, and the connections of the flows in the way are
  // what the router has to move.
  using FlowKey = std::pair<Port, int>;
  std::map<TileID, std::map<FlowKey, SlaveFlow>> tileSlaveFlows;
  for (const auto *flows : {&ctrlPacketFlows, &packetFlows})
    for (const auto &[flow, dests] : *flows) {
      auto [it, inserted] = tileSlaveFlows[flow.first.first].try_emplace(
          {flow.first.second, flow.second},
          SlaveFlow{flow.first.second, flow.second, {}, false});
      SlaveFlow &f = it->second;
      f.isCtrlPkt |= flows == &ctrlPacketFlows;
      for (const PhysPort &dest : dests)
        if (!llvm::is_contained(f.masters, dest.second))
          f.masters.push_back(dest.second);
      llvm::sort(f.masters);
    }
  std::map<TileID, SmallVector<SlaveFlow, 8>> tileFlows;
  for (const auto &[tileId, byFlow] : tileSlaveFlows)
    for (const auto &[key, f] : byFlow)
      tileFlows[tileId].push_back(f);

  auto conflictingStreams =
      [&](TileID tileId, const SlaveFlow &a,
          const SlaveFlow &b) -> std::optional<std::pair<size_t, size_t>> {
    auto as = slaveFlowStreams.find({{tileId, a.slave}, a.id});
    auto bs = slaveFlowStreams.find({{tileId, b.slave}, b.id});
    if (as == slaveFlowStreams.end() || bs == slaveFlowStreams.end())
      return std::nullopt;
    for (size_t s : as->second)
      for (size_t t : bs->second)
        if (conflicts.conflict(s, t))
          return std::pair{s, t};
    return std::nullopt;
  };
  // What the search below learns: flows to keep on different arbiters, and
  // flows to keep off an arbiter that flows already in the design take.
  std::set<std::tuple<TileID, FlowKey, FlowKey>> apart;
  std::set<std::tuple<TileID, FlowKey, int>> offArbiter;
  auto planTile = [&](TileID tileId,
                      SmallVectorImpl<std::pair<size_t, size_t>> &blocking) {
    ArrayRef<SlaveFlow> flows = tileFlows.at(tileId);
    auto key = [&](size_t f) { return FlowKey{flows[f].slave, flows[f].id}; };
    return planArbiters(
        flows,
        [&](size_t a, size_t b) {
          return apart.count({tileId, std::min(key(a), key(b)),
                              std::max(key(a), key(b))}) ||
                 conflictingStreams(tileId, flows[a], flows[b]);
        },
        [&](size_t f, int arbiter) {
          return offArbiter.count({tileId, key(f), arbiter}) > 0;
        },
        reservedAmsels[tileId], blocking);
  };

  std::set<std::pair<TileID, Connect>> hazardConnections;
  auto moveFlow = [&](TileID tileId, FlowKey key) {
    auto flows = tileSlaveFlows.find(tileId);
    if (flows != tileSlaveFlows.end())
      if (auto f = flows->second.find(key); f != flows->second.end())
        for (Port m : f->second.masters)
          hazardConnections.insert({tileId, {key.first, m}});
    for (const Connect &hop : circuitHops[tileId])
      if (hop.src == key.first)
        hazardConnections.insert({tileId, hop});
  };
  // Flows branching to several master ports tie them to one arbiter, so a
  // flow's arbiter is fixed by every such flow reaching its master ports too.
  auto moveUnit = [&](TileID tileId, FlowKey key) {
    moveFlow(tileId, key);
    auto flows = tileSlaveFlows.find(tileId);
    if (flows == tileSlaveFlows.end())
      return;
    auto self = flows->second.find(key);
    if (self == flows->second.end())
      return;
    std::set<Port> unit(self->second.masters.begin(),
                        self->second.masters.end());
    std::set<FlowKey> joined;
    for (bool grew = true; grew;) {
      grew = false;
      for (const auto &[other, f] : flows->second)
        if (f.masters.size() > 1 && !joined.count(other) &&
            llvm::any_of(f.masters, [&](Port m) { return unit.count(m); })) {
          joined.insert(other);
          unit.insert(f.masters.begin(), f.masters.end());
          moveFlow(tileId, other);
          grew = true;
        }
    }
  };
  auto fail = [&](std::string reason) -> LogicalResult {
    if (!hazards)
      return device.emitError("Unable to find a legal routing: ") << reason;
    hazards->connections.assign(hazardConnections.begin(),
                                hazardConnections.end());
    hazards->reason = std::move(reason);
    return success();
  };

  const uint32_t maxPacketId = targetModel.getMaxPacketId();
  const int idBits = llvm::Log2_32_Ceil(maxPacketId + 1);
  const int idMask = (1 << idBits) - 1;

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
  std::optional<std::string> planFailure;
  for (const auto &[slaveFlow, sources] : slaveFlowSources) {
    const auto &[first, firstMasters] = *sources.begin();
    auto other = llvm::find_if(sources, [&](const auto &source) {
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
        "at tile ({0}, {1}), packets with id {2} from ({3}, {4}) {5}:{6} and "
        "({7}, {8}) {9}:{10} enter on {11}:{12} and leave by different ports; "
        "a switchbox routes on the id alone, so each source's packets would "
        "also go where the other's do.",
        tileId.col, tileId.row, id, first.coords.col, first.coords.row,
        stringifyWireBundle(first.port.bundle), first.port.channel,
        second.coords.col, second.coords.row,
        stringifyWireBundle(second.port.bundle), second.port.channel,
        stringifyWireBundle(slavePort.second.bundle), slavePort.second.channel);
  }
  for (const auto &[tileId, byFlow] : tileSlaveFlows) {
    std::map<Port, std::map<SmallVector<Port, 4>, RuleGroup>> ports;
    for (const auto &[key, f] : byFlow) {
      RuleGroup &group = ports[f.slave][f.masters];
      auto mask = pinnedMasks.find({{tileId, f.slave}, f.id});
      if (mask == pinnedMasks.end() || mask->second == idMask)
        group.derived.push_back(f.id);
      else if (!llvm::is_contained(group.stated, std::pair{mask->second, f.id}))
        group.stated.push_back({mask->second, f.id});
    }
    for (const auto &[slave, groups] : ports) {
      SmallVector<std::pair<int, int>> existing =
          existingCubes[{tileId, slave}];
      SmallVector<GroupClaims> claims;
      for (const auto &[masters, group] : groups)
        claims.push_back({group.stated, group.derived});
      size_t needed =
          existing.size() +
          portRules(claims, existing, idBits, targetModel.getNumSlaveSlots())
              .size();
      if (needed <= targetModel.getNumSlaveSlots())
        continue;
      for (const auto &[key, f] : byFlow)
        if (f.slave == slave)
          moveFlow(tileId, key);
      if (planFailure)
        continue;
      planFailure = llvm::formatv(
          "at tile ({0}, {1}), the packet flows entering on {2}:{3} need {4} "
          "packet rules, and a slave port holds {5}.",
          tileId.col, tileId.row, stringifyWireBundle(slave.bundle),
          slave.channel, needed, targetModel.getNumSlaveSlots());
    }
  }

  std::map<TileID, ArbiterPlan> plans;
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
        llvm::dbgs() << "  " << stringifyWireBundle(f.slave.bundle) << ':'
                     << f.slave.channel << " id " << f.id << " ->";
        for (Port m : f.masters)
          llvm::dbgs() << ' ' << stringifyWireBundle(m.bundle) << ':'
                       << m.channel;
        llvm::dbgs() << '\n';
      }
      for (auto [a, b] : blocking)
        llvm::dbgs() << "  blocking " << a << ' ' << b << '\n';
    });
    std::string reason;
    llvm::raw_string_ostream os(reason);
    if (blocking.empty()) {
      for (const SlaveFlow &f : flows)
        moveFlow(tileId, {f.slave, f.id});
      if (planFailure)
        continue;
      os << "at tile (" << tileId.col << ", " << tileId.row
         << "), the packet flows need more arbiter msels than the switchbox "
            "has free.";
      planFailure = std::move(reason);
      continue;
    }
    for (auto [a, b] : blocking)
      for (size_t f : {a, b})
        moveUnit(tileId, {flows[f].slave, flows[f].id});
    if (planFailure)
      continue;
    auto [s, t] = *conflictingStreams(tileId, flows[blocking.front().first],
                                      flows[blocking.front().second]);
    os << describeStream(conflicts.getStreams()[s]) << " and "
       << describeStream(conflicts.getStreams()[t])
       << " can deadlock if they share an arbiter, and no routing found keeps "
          "them apart (last tried: tile ("
       << tileId.col << ", " << tileId.row << ")). " << conflicts.explain(s, t);
    planFailure = std::move(reason);
  }
  if (planFailure)
    return fail(std::move(*planFailure));

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
  std::optional<HoldCycle> firstCycle;
  int budget = 256;
  std::function<bool()> search = [&]() {
    arbitrate();
    std::optional<HoldCycle> cycle = conflicts.holdCycle(routes);
    if (!cycle)
      return true;
    if (!firstCycle)
      firstCycle = cycle;
    LLVM_DEBUG(llvm::dbgs()
               << "Hold cycle: " << conflicts.explain(*cycle) << '\n');
    for (const HoldCycle::Step &step : cycle->steps) {
      if (step.wait != HoldCycle::Wait::Arbiter)
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
      if (budget <= 0)
        return false;
      if ((pair && !apart.insert(*pair).second) ||
          (off && !offArbiter.insert(*off).second))
        continue;
      budget--;
      ArbiterPlan saved = plans.at(step.tile);
      SmallVector<std::pair<size_t, size_t>, 4> blocking;
      if (std::optional<ArbiterPlan> plan = planTile(step.tile, blocking)) {
        plans[step.tile] = std::move(*plan);
        if (search())
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
  if (!search()) {
    arbitrate();
    for (const HoldCycle::Step &step : firstCycle->steps)
      if (step.wait != HoldCycle::Wait::Drain)
        for (auto [s, input] : {std::pair{step.waiting, step.sharerInput},
                                std::pair{step.sharer, step.sharerInput},
                                std::pair{step.holding, step.holderInput}})
          if (s < numRequested)
            moveUnit(step.tile, {input, *streams[s].packetID});
    return fail("packet flows can deadlock holding arbiters across "
                "switchboxes, and no arbiter assignment found avoids it. " +
                conflicts.explain(*firstCycle));
  }
  if (hazards)
    return success();

  for (const auto &[tileId, plan] : plans) {
    for (const auto &[amsel, masters] : plan.amselMasters)
      masterAMSels[{tileId, amsel}].append(masters.begin(), masters.end());
    for (const auto &[flow, amsel] : plan.slaveAmsels)
      slaveAMSels[{{tileId, flow.first}, flow.second}] = amsel;
  }
  packetFlows.insert(ctrlPacketFlows.begin(), ctrlPacketFlows.end());

  // Compute the master set IDs
  // A map from a switchbox output port to its associated amsel values
  std::map<PhysPort, SmallVector<int, 4>> mastersets;
  for (const auto &[physPort, ports] : masterAMSels) {
    TileID tileId = physPort.first;
    int amselValue = physPort.second;
    for (auto port : ports) {
      PhysPort physPort = {tileId, port};
      mastersets[physPort].push_back(amselValue);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
#ifndef NDEBUG
  for (const auto &[physPort, values] : mastersets) {
    TileID tileId = physPort.first;
    WireBundle bundle = physPort.second.bundle;
    int channel = physPort.second.channel;
    LLVM_DEBUG(llvm::dbgs()
               << "master " << tileId << " " << stringifyWireBundle(bundle)
               << " : " << channel << '\n');
    for (auto value : values)
      LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');
  }
#endif

  // Compute mask values
  // Merging as many stream flows as possible
  // The flows must originate from the same source port and have different IDs
  // Two flows can be merged if they share the same destinations
  SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
  SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
  while (!workList.empty()) {
    auto slave1 = workList.pop_back_val();
    Port slavePort1 = slave1.first.second;

    bool foundgroup = false;
    for (auto &group : slaveGroups) {
      auto slave2 = group.front();
      if (Port slavePort2 = slave2.first.second; slavePort1 != slavePort2)
        continue;

      bool matched = true;
      auto dests1 = packetFlows[slave1];
      auto dests2 = packetFlows[slave2];
      if (dests1.size() != dests2.size())
        continue;

      for (auto dest1 : dests1) {
        if (llvm::find(dests2, dest1) == dests2.end()) {
          matched = false;
          break;
        }
      }

      if (matched) {
        group.push_back(slave1);
        foundgroup = true;
        break;
      }
    }

    if (!foundgroup) {
      SmallVector<std::pair<PhysPort, int>, 4> group({slave1});
      slaveGroups.push_back(group);
    }
  }

  // What each group claims on its slave port, as cubes, split into the rules
  // its flows state and the ids left for the cover to describe.
  SmallVector<SmallVector<std::pair<int, int>, 4>, 4> statedRules(
      slaveGroups.size());
  SmallVector<SmallVector<int, 4>, 4> derivedIds(slaveGroups.size());
  for (size_t gi = 0; gi < slaveGroups.size(); ++gi) {
    for (auto member : slaveGroups[gi]) {
      auto it = pinnedMasks.find(member);
      // A full-width mask selects the id alone, which the cover states just as
      // well, so only a wider claim becomes a rule of its own.
      if (it == pinnedMasks.end() || it->second == idMask) {
        derivedIds[gi].push_back(member.second);
        continue;
      }
      std::pair<int, int> cube = {it->second, member.second};
      if (!llvm::is_contained(statedRules[gi], cube)) {
        statedRules[gi].push_back(cube);
      }
    }
  }

  // Everything a group claims, for the other groups on its port to avoid.
  auto claimsOf = [&](size_t gi) {
    SmallVector<std::pair<int, int>, 8> claims(statedRules[gi].begin(),
                                               statedRules[gi].end());
    for (int id : derivedIds[gi]) {
      claims.push_back({idMask, id});
    }
    return claims;
  };

  // Realize the routes in MLIR

  // Update tiles map if any new tile op declaration is needed for constructing
  // the flow.
  for (const auto &swMap : mastersets) {
    TileID tileId = swMap.first.first;
    TileOp tileOp = analyzer.getTile(builder, tileId);
    if (llvm::none_of(tiles,
                      [&tileOp](const std::pair<const xilinx::AIE::TileID,
                                                Operation *> &tileMapEntry) {
                        return tileMapEntry.second == tileOp.getOperation();
                      })) {
      tiles[{tileOp.colIndex(), tileOp.rowIndex()}] = tileOp;
    }
  }

  for (auto map : tiles) {
    Operation *tileOp = map.second;
    TileOp tile = cast<TileOp>(map.second);
    TileID tileId = tile.getTileID();
    Location tileLoc = tile.getLoc();

    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox =
        analyzer.getSwitchbox(builder, tile.colIndex(), tile.rowIndex());
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder, tileLoc);
    Block &b = swbox.getConnections().front();
    builder.setInsertionPoint(b.getTerminator());

    for (const Connect &hop : circuitHops[tileId])
      ConnectOp::create(builder, tileLoc, hop.src.bundle, hop.src.channel,
                        hop.dst.bundle, hop.dst.channel);

    std::vector<bool> amselOpNeededVector(numMselsPerArbiter * numArbiters);
    for (const auto &map : mastersets) {
      if (tileId != map.first.first)
        continue;

      for (auto value : map.second) {
        amselOpNeededVector[value] = true;
      }
    }
    // Create all the amsel Ops
    std::map<int, AMSelOp> amselOps;
    for (int i = 0; i < numMselsPerArbiter; i++) {
      for (int a = 0; a < numArbiters; a++) {
        int amselValue = a + i * numArbiters;
        if (amselOpNeededVector[amselValue]) {
          int arbiterID = a;
          int msel = i;
          auto amsel = AMSelOp::create(builder, tileLoc, arbiterID, msel);
          amselOps[amselValue] = amsel;
        }
      }
    }
    // Create all the master set Ops
    // First collect the master sets for this tile.
    SmallVector<Port, 4> tileMasters;
    for (const auto &map : mastersets) {
      if (tileId != map.first.first)
        continue;
      tileMasters.push_back(map.first.second);
    }
    // Sort them so we get a reasonable order
    std::sort(tileMasters.begin(), tileMasters.end());
    for (auto tileMaster : tileMasters) {
      WireBundle bundle = tileMaster.bundle;
      int channel = tileMaster.channel;
      SmallVector<int, 4> msels = mastersets[{tileId, tileMaster}];
      SmallVector<Value, 4> amsels;
      for (auto msel : msels) {
        assert(amselOps.count(msel) == 1);
        amsels.push_back(amselOps[msel]);
      }

      auto msOp = MasterSetOp::create(builder, tileLoc, builder.getIndexType(),
                                      bundle, channel, amsels,
                                      keepPktHeaderAttr[{tileId, tileMaster}]);
      if (ctrlPktOverlayMasterPorts.contains({tileId, tileMaster}))
        msOp->setAttr("is_ctrl_pkt_overlay", builder.getUnitAttr());
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

      WireBundle bundle = port.second.bundle;
      int channel = port.second.channel;
      auto slave = port.second;

      SmallVector<int, 4> matchIds;
      for (auto member : group)
        matchIds.push_back(member.second);

      for (int id : matchIds)
        if (id > static_cast<int>(maxPacketId)) {
          return mlir::emitError(tileLoc)
                 << "packet id " << id << " exceeds the maximum of "
                 << maxPacketId;
        }

      SmallVector<std::pair<int, int>> avoidCubes;
      for (size_t oi = 0; oi < slaveGroups.size(); ++oi) {
        if (oi != gi && slaveGroups[oi].front().first == port) {
          auto claims = claimsOf(oi);
          avoidCubes.append(claims.begin(), claims.end());
        }
      }

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

      // Groups on one slave port carry different destination sets, so two of
      // them must not claim the same id. A mask written on an aie.packet_flow
      // claims every id it matches, including ids no flow in this design
      // mentions, so the overlap does not show in the ids alone.
      for (std::pair<int, int> own : claimsOf(gi)) {
        for (std::pair<int, int> other : avoidCubes) {
          if (cubesIntersect(own, other)) {
            int witness = (own.second & own.first) |
                          (other.second & other.first & ~own.first);
            return mlir::emitError(tileLoc)
                   << "packet flows through " << stringifyWireBundle(bundle)
                   << channel << " claim rule (mask 0x"
                   << llvm::utohexstr(own.first) << ", id 0x"
                   << llvm::utohexstr(own.second) << ") and rule (mask 0x"
                   << llvm::utohexstr(other.first) << ", id 0x"
                   << llvm::utohexstr(other.second)
                   << "), which both match id 0x" << llvm::utohexstr(witness)
                   << "; widen one mask to carry both, or route them apart";
          }
        }
      }

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
          claims.push_back({statedRules[oi], derivedIds[oi]});
        }
        SmallVector<std::pair<int, int>> existing;
        for (PacketRuleOp rule : existingRules.lookup(slave))
          existing.push_back({rule.maskInt(), rule.valueInt()});
        plan.rules = portRules(claims, existing, idBits,
                               device.getTargetModel().getNumSlaveSlots());
        LLVM_DEBUG({
          llvm::dbgs() << "packet rules " << stringifyWireBundle(bundle)
                       << channel << ":";
          for (const PortRule &r : plan.rules)
            llvm::dbgs() << " rule(" << r.mask << ", " << r.value
                         << ") -> group " << plan.groups[r.group];
          llvm::dbgs() << '\n';
        });
      }

      // Every id the group claims takes one of its rules first.
      for ([[maybe_unused]] int id = 0; id <= idMask; ++id) {
        [[maybe_unused]] bool own =
            llvm::is_contained(derivedIds[gi], id) ||
            llvm::any_of(statedRules[gi], [&](std::pair<int, int> c) {
              return (id & c.first) == (c.second & c.first);
            });
        [[maybe_unused]] auto first =
            llvm::find_if(plan.rules, [&](const PortRule &r) {
              return (id & r.mask) == r.value;
            });
        assert((!own || (first != plan.rules.end() &&
                         plan.groups[first->group] == gi)) &&
               "packet rules send a claimed id elsewhere");
      }

      size_t last = plan.emitted;
      for (size_t r = plan.emitted; r < plan.rules.size(); ++r)
        if (plan.groups[plan.rules[r].group] == gi)
          last = r + 1;

      // Check if this group is a ctrl-pkt overlay flow
      bool isCtrlPktGroup = ctrlPacketFlows.count(group.front()) > 0;

      PacketRulesOp packetrules;
      if (slaveRules.count(slave) == 0) {
        packetrules = PacketRulesOp::create(builder, tileLoc, bundle, channel);
        PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                        tileLoc);
        if (isCtrlPktGroup)
          packetrules->setAttr("is_ctrl_pkt_overlay", builder.getUnitAttr());
        slaveRules[slave] = packetrules;
      } else {
        // After the amsels its new rules use.
        packetrules = slaveRules[slave];
        packetrules->moveBefore(b.getTerminator());
      }

      Block &rules = packetrules.getRules().front();

      // A fan-out whose cover exceeds the slave port's packet-rule slots needs
      // channel-level restructuring, not masking.
      uint32_t slotLimit = device.getTargetModel().getNumSlaveSlots();
      uint32_t existingSlots = 0;
      for (auto rule : rules.getOps<PacketRuleOp>()) {
        (void)rule;
        existingSlots++;
      }
      if (existingSlots + (last - plan.emitted) > slotLimit) {
        packetrules->emitOpError("slave port packet rules exceed the ")
            << slotLimit << "-slot limit (" << existingSlots << " + "
            << last - plan.emitted << ").";
        return failure();
      }

      builder.setInsertionPoint(rules.getTerminator());
      for (; plan.emitted < last; ++plan.emitted) {
        const PortRule &r = plan.rules[plan.emitted];
        PacketRuleOp::create(
            builder, tileLoc, r.mask, r.value,
            amselOps[slaveAMSels[slaveGroups[plan.groups[r.group]].front()]]);
      }
    }
  }

  // Add support for shimDMA
  // From shimDMA to BLI: 1) shimDMA 0 --> North 3
  //                      2) shimDMA 1 --> North 7
  // From BLI to shimDMA: 1) North   2 --> shimDMA 0
  //                      2) North   3 --> shimDMA 1

  for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
    auto retVal = switchbox->getOperand(0);
    auto tileOp = retVal.getDefiningOp<TileOp>();

    // Check if it is a shim Tile
    if (!tileOp.isShimNOCTile())
      continue;

    // Check if the switchbox is empty
    if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
      continue;

    Region &r = switchbox.getConnections();
    Block &b = r.front();

    // Find if the corresponding shimmux exsists or not
    int shimExist = 0;
    ShimMuxOp shimOp;
    for (auto shimmux : device.getOps<ShimMuxOp>()) {
      if (shimmux.getTile() == tileOp) {
        shimExist = 1;
        shimOp = shimmux;
        break;
      }
    }

    for (Operation &Op : b.getOperations()) {
      if (auto pktrules = dyn_cast<PacketRulesOp>(Op)) {

        // check if there is MM2S DMA in the switchbox of the 0th row
        if (pktrules.getSourceBundle() == WireBundle::DMA) {

          // If there is, then it should be put into the corresponding shimmux
          // If shimmux not defined then create shimmux
          if (!shimExist) {
            builder.setInsertionPointAfter(tileOp);
            shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          pktrules.setSourceBundle(WireBundle::South);
          if (pktrules.getSourceChannel() == 0) {
            pktrules.setSourceChannel(3);
            getOrCreateConnect(builder, shimOp, tileOp.getLoc(),
                               WireBundle::DMA, 0, WireBundle::North, 3);
          }
          if (pktrules.getSourceChannel() == 1) {
            pktrules.setSourceChannel(7);
            getOrCreateConnect(builder, shimOp, tileOp.getLoc(),
                               WireBundle::DMA, 1, WireBundle::North, 7);
          }
        }
      }

      if (auto mtset = dyn_cast<MasterSetOp>(Op)) {

        // check if there is S2MM DMA in the switchbox of the 0th row
        if (mtset.getDestBundle() == WireBundle::DMA) {

          // If there is, then it should be put into the corresponding shimmux
          // If shimmux not defined then create shimmux
          if (!shimExist) {
            builder.setInsertionPointAfter(tileOp);
            shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          mtset.setDestBundle(WireBundle::South);
          if (mtset.getDestChannel() == 0) {
            mtset.setDestChannel(2);
            getOrCreateConnect(builder, shimOp, tileOp.getLoc(),
                               WireBundle::North, 2, WireBundle::DMA, 0);
          }
          if (mtset.getDestChannel() == 1) {
            mtset.setDestChannel(3);
            getOrCreateConnect(builder, shimOp, tileOp.getLoc(),
                               WireBundle::North, 3, WireBundle::DMA, 1);
          }
        }
      }
    }
  }

  target.addIllegalOp<PacketFlowOp>();
  RewritePatternSet patterns(&getContext());
  patterns.insert<AIEOpRemoval<PacketFlowOp>>(device.getContext());

  if (failed(applyPartialConversion(device, target, std::move(patterns))))
    return failure();

  return success();
}

// The router names a shim DMA by its own port, and the end of
// runOnPacketFlow moves it behind the shim mux onto a South channel. Rules and
// master sets a previous run left there are moved back first, so new flows on
// the same DMA channel share them.
static void unmuxShimDMAPacketPorts(DeviceOp device) {
  for (auto shimMux : device.getOps<ShimMuxOp>()) {
    auto hasConnect = [&](WireBundle srcBundle, int srcCh,
                          WireBundle destBundle, int destCh) {
      return llvm::any_of(shimMux.getConnections().getOps<ConnectOp>(),
                          [&](ConnectOp c) {
                            return c.sourcePort() == Port{srcBundle, srcCh} &&
                                   c.destPort() == Port{destBundle, destCh};
                          });
    };
    for (auto switchbox : device.getOps<SwitchboxOp>()) {
      if (switchbox.getTileOp() != shimMux.getTileOp())
        continue;
      for (auto rules : switchbox.getConnections().getOps<PacketRulesOp>())
        for (int ch : {0, 1})
          if (rules.sourcePort() == Port{WireBundle::South, ch ? 7 : 3} &&
              hasConnect(WireBundle::DMA, ch, WireBundle::North, ch ? 7 : 3)) {
            rules.setSourceBundle(WireBundle::DMA);
            rules.setSourceChannel(ch);
            break;
          }
      for (auto masterSet : switchbox.getConnections().getOps<MasterSetOp>())
        for (int ch : {0, 1})
          if (masterSet.destPort() == Port{WireBundle::South, ch ? 3 : 2} &&
              hasConnect(WireBundle::North, ch ? 3 : 2, WireBundle::DMA, ch)) {
            masterSet.setDestBundle(WireBundle::DMA);
            masterSet.setDestChannel(ch);
            break;
          }
    }
  }
}

void AIEPathfinderPass::runOnOperation() {

  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
  if (clRoutePacket)
    unmuxShimDMAPacketPorts(d);

  // Packet flows that can deadlock must not share an arbiter. Every routing
  // the router finds is checked by planning the arbiters on it, and one that
  // cannot be planned counts as illegal, so routing and allocation agree.
  StreamConflicts conflicts(d);
  if (auto pairs = conflicts.unavoidable(); !pairs.empty()) {
    InFlightDiagnostic warning =
        emitWarning(d.getLoc(), "Flows can deadlock however they are routed: ")
        << conflicts.explain(pairs[0].first, pairs[0].second);
    if (pairs.size() > 1)
      warning << " So can " << pairs.size() - 1 << " other pair"
              << (pairs.size() > 2 ? "s" : "") << " of flows.";
  }
  DynamicTileAnalysis &analyzer = getAnalysis<DynamicTileAnalysis>();
  std::map<PathEndPoint, SmallVector<size_t, 4>> streamsFrom;
  if (clRoutePacket && !d.getOps<PacketFlowOp>().empty()) {
    const AIETargetModel &targetModel = d.getTargetModel();
    if (std::optional<std::string> reason =
            unroutableArbiters(d, conflicts, [&](TileID tile) {
              return !clCircuitSwitchHops ||
                     targetModel.isShimNOCorPLTile(tile.col, tile.row);
            })) {
      d.emitError("Unable to find a legal routing: ") << *reason;
      signalPassFailure();
      return;
    }
    for (auto [i, s] : llvm::enumerate(conflicts.getRequestedStreams()))
      if (s.packetID)
        streamsFrom[{s.src.tile, s.src.port}].push_back(i);
    analyzer.pathfinder->setPacketConflict(
        [&](const PathEndPoint &a, const PathEndPoint &b) {
          auto as = streamsFrom.find(a), bs = streamsFrom.find(b);
          if (as == streamsFrom.end() || bs == streamsFrom.end())
            return false;
          for (size_t s : as->second)
            for (size_t t : bs->second)
              if (conflicts.conflict(s, t))
                return true;
          return false;
        });
    analyzer.pathfinder->setRoutingCheck(
        [&](const std::map<PathEndPoint, SwitchSettings> &solution) {
          RoutingHazards hazards;
          (void)runOnPacketFlow(d, builder, analyzer, solution, conflicts,
                                &hazards);
          LLVM_DEBUG(if (!hazards.connections.empty()) llvm::dbgs()
                     << "Routing rejected: " << hazards.reason << '\n');
          if (!hazards.reason.empty())
            analyzer.routingFailureReason = std::move(hazards.reason);
          return std::move(hazards.connections);
        });
  }
  if (failed(analyzer.runAnalysis(d))) {
    signalPassFailure();
    return;
  }

  if (clRouteCircuit && failed(runOnFlow(d, analyzer))) {
    signalPassFailure();
    return;
  }
  if (clRoutePacket &&
      failed(runOnPacketFlow(d, builder, analyzer, analyzer.flowSolutions,
                             conflicts))) {
    signalPassFailure();
    return;
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
      TileOp tile;
      if (analyzer.coordToTile.count({col, row}))
        tile = analyzer.coordToTile[{col, row}];
      else
        continue;
      SwitchboxOp sw;
      if (analyzer.coordToSwitchbox.count({col, row}))
        sw = analyzer.coordToSwitchbox[{col, row}];
      else
        continue;
      Location loc = tile.getLoc();
      if (col > 0) {
        // connections east-west between stream switches
        if (analyzer.coordToSwitchbox.count({col - 1, row})) {
          auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
          wire(loc, westsw, WireBundle::East, sw, WireBundle::West);
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        wire(loc, tile, WireBundle::Core, sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        wire(loc, tile, WireBundle::DMA, sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          wire(loc, southsw, WireBundle::North, sw, WireBundle::South);
        }
      } else if (row == 0) {
        if (tile.isShimNOCTile()) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            wire(loc, shimsw,
                 WireBundle::North, // Changed to connect into the north
                 sw, WireBundle::South);
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              wire(loc, plio, WireBundle::North, shimsw, WireBundle::South);
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            wire(loc, tile, WireBundle::DMA, shimsw, WireBundle::DMA);
          }
        } else if (tile.isShimPLTile()) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            wire(loc, plio, WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
  }
}

std::unique_ptr<OperationPass<DeviceOp>> createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

} // namespace xilinx::AIE
