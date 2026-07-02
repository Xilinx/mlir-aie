//===- AIEObjectFifoLiveness.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flag under-buffered cyclic objectFIFO dependencies that deadlock the array.
//
// A coupled multicast (broadcast) objectFIFO whose consumers back-pressure the
// producer forms a dependency cycle. When the producer must re-acquire a slot
// on a later trip (replay, T >= 2) while the prior trip's tokens are still
// outstanding, the round-trip slack (2 * depth) can be smaller than the number
// of outstanding tokens demanded by the coupled fan-out (array_fan * T). That
// is a static, structural deadlock: the IR compiles clean and then hangs the
// NPU at runtime.
//
// This pass builds the objectFIFO data + back-pressure dependency graph. The
// graph is UNDIRECTED (a shared tile couples its producer and consumer fifos
// with data and back-pressure edges in both directions), so cycle detection
// degenerates to connected components: any fifo that shares a tile with another
// sits in a back-pressure loop. We therefore group fifos with a union-find and
// treat a component of size > 1 as a cycle. The validated SDF model is then
// applied PER coupled-multicast group so an unrelated cycle elsewhere in the
// device cannot inflate a group's demand:
//
//   demand   = array_fan * T          (array_fan = the group's multicast
//                                        fan-out, summed across the components
//                                        it spans; T = max repeat_count among
//                                        the fifos in those components)
//   slack    = 2 * depth              (depth = min depth in the group)
//   DEADLOCK iff  T >= 2  AND  depth > 0  AND  demand > slack
//
// Coupling (which fifos are halves of ONE logical broadcast tensor) is a
// property the IR does not carry: the halves can live on different, unlinked
// MemTiles, so neither aie.objectfifo.link nor a shared producer tile identifies
// them -- only their shared name base does (see nameBase). Because a name is a
// weak signal, the grouping is gated on structural agreement (identical element
// type, depth and repeat_count, and disjoint consumer tiles); a fifo that only
// shares the base name is analyzed on its own rather than folded in. A durable
// fix would be a first-class coupling attribute on the frontend ops; that is a
// tracked follow-up.
//
// The T >= 2 replay guard avoids false positives on single-trip broadcasts
// (which fan out once and drain monotonically). The analysis is sound for this
// static SDF class and conservative elsewhere (it never errors outside a proven
// cyclic under-buffered multicast); it is not a general deadlock detector.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEOBJECTFIFOLIVENESS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-objectfifo-liveness"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// Strip a trailing "_<digits>" group from a fifo name, mirroring the Phase-0
// Python `re.sub(r'_\d+$', '', name)`. Requires at least one digit after the
// final '_'; otherwise the name is returned unchanged.
static StringRef nameBase(StringRef name) {
  size_t us = name.rfind('_');
  if (us == StringRef::npos || us + 1 >= name.size())
    return name;
  StringRef suffix = name.substr(us + 1);
  for (char c : suffix)
    if (c < '0' || c > '9')
      return name;
  return name.substr(0, us);
}

struct AIEObjectFifoLivenessPass
    : xilinx::AIE::impl::AIEObjectFifoLivenessBase<AIEObjectFifoLivenessPass> {
  void runOnOperation() override {
    DeviceOp mod = getOperation();

    struct F {
      Operation *op;
      long depth;
      mlir::Value prod;
      SmallVector<mlir::Value> cons;
      long repeat;
      StringRef name;
      mlir::Type elemTy; // coupling signature (see nameBase gating below)
    };
    SmallVector<F> fifos;

    mod.walk([&](xilinx::AIE::ObjectFifoCreateOp ofo) {
      // Mirror the Phase-0 parser: it only matches a single-integer depth
      // (`}, N : i32`); array-form per-tile depths are skipped entirely, so
      // skip them here too to keep the dependency graph identical.
      auto depthInt = dyn_cast<IntegerAttr>(ofo.getElemNumber());
      if (!depthInt)
        return;
      F f;
      f.op = ofo;
      f.name = ofo.getSymName();
      f.depth = depthInt.getInt();
      f.prod = ofo.getProducerTile();
      for (mlir::Value c : ofo.getConsumerTiles())
        f.cons.push_back(c);
      f.repeat = 1;
      if (auto rc = ofo.getRepeatCount())
        f.repeat = *rc;
      f.elemTy = ofo.getElemType();
      fifos.push_back(f);
    });

    unsigned n = fifos.size();

    // build_dep_graph: group fifos by shared tile Value; for each tile, ins =
    // fifos it consumes, outs = fifos it produces; add edges o<->i (data +
    // back-pressure). Use Value identity as the tile key.
    mlir::DenseMap<mlir::Value, SmallVector<unsigned>> consumes, produces;
    for (unsigned i = 0; i < n; ++i) {
      produces[fifos[i].prod].push_back(i);
      for (mlir::Value c : fifos[i].cons)
        consumes[c].push_back(i);
    }
    std::vector<std::set<unsigned>> adj(n);
    mlir::DenseSet<mlir::Value> tiles;
    for (auto &kv : consumes)
      tiles.insert(kv.first);
    for (auto &kv : produces)
      tiles.insert(kv.first);
    for (mlir::Value tile : tiles) {
      auto &ins = consumes[tile];
      auto &outs = produces[tile];
      for (unsigned o : outs)
        for (unsigned i : ins) {
          adj[o].insert(i);
          adj[i].insert(o);
        }
    }

    // find_cycles: the graph is undirected, so a directed SCC search would just
    // recover connected components (every shared-tile edge is a 2-cycle). Use a
    // union-find and treat a component of size > 1 as a cycle; a lone fifo that
    // shares no tile is acyclic. sccId is the component root for cyclic nodes and
    // -1 otherwise (the downstream group logic only keys off sccId identity, so a
    // root index is as good as a dense SCC id).
    SmallVector<int> uf(n);
    for (unsigned i = 0; i < n; ++i)
      uf[i] = i;
    std::function<int(int)> find = [&](int x) {
      while (uf[x] != x) {
        uf[x] = uf[uf[x]]; // path halving
        x = uf[x];
      }
      return x;
    };
    for (unsigned v = 0; v < n; ++v)
      for (unsigned w : adj[v])
        uf[find(v)] = find(w);
    SmallVector<int> compSize(n, 0);
    for (unsigned i = 0; i < n; ++i)
      ++compSize[find(i)];
    SmallVector<int> sccId(n, -1);
    for (unsigned i = 0; i < n; ++i)
      if (compSize[find(i)] > 1)
        sccId[i] = find(i);

    // SDF check, scoped PER coupled-multicast group -- NOT globally. A coupled
    // broadcast is a group of cyclic multicast objectFIFOs that are one logical
    // broadcast tensor (e.g. memW1_0 + memW1_1 = one weight broadcast, split
    // across MemTiles); the halves can land in different components, so the group
    // -- not the component -- is the unit of coupling, and array_fan SUMS the
    // group's fan-out across the components it spans. The trip-count T is taken
    // ONLY from the component(s) this group lives in, so an unrelated
    // high-repeat_count (or a wide broadcast) elsewhere in the device cannot
    // inflate this group's demand and false-positive.
    //
    // The IR carries no signal for "these fifos are one logical tensor", so the
    // grouping keys off the shared name base (nameBase). Because a name is a weak
    // signal, it is GATED on structural agreement before two fifos are coupled:
    // identical element type, depth and repeat_count, and DISJOINT consumer tiles
    // (two halves of a broadcast reach different cores). A fifo that shares the
    // base name but fails any gate is analyzed as its own standalone group rather
    // than folded in -- so a stray name collision cannot fabricate a coupling,
    // and the summation is never applied to fifos that are not actually halves.
    struct Group {
      long fan = 0, depth = 0;
      unsigned rep = 0;
      std::string name;           // display name (nameBase of the representative)
      mlir::Type elemTy;          // coupling signature: element type ...
      long sigDepth = 0;          // ... depth ...
      long sigRepeat = 0;         // ... and repeat_count must all match to couple
      llvm::DenseSet<mlir::Value> cons; // consumer tiles (must stay disjoint)
      std::set<int> sccs;
    };
    SmallVector<Group> groupList;
    llvm::StringMap<unsigned> groupIdx; // name base -> owning group in groupList
    for (unsigned i = 0; i < n; ++i) {
      if (sccId[i] < 0 || fifos[i].cons.size() <= 1)
        continue; // only a cyclic multicast couples a broadcast
      StringRef base = nameBase(fifos[i].name);
      int slot = -1;
      auto it = groupIdx.find(base);
      if (it != groupIdx.end()) {
        Group &cand = groupList[it->second];
        bool consistent = cand.elemTy == fifos[i].elemTy &&
                          cand.sigDepth == fifos[i].depth &&
                          cand.sigRepeat == fifos[i].repeat;
        for (mlir::Value c : fifos[i].cons)
          if (consistent && cand.cons.contains(c))
            consistent = false;
        if (consistent)
          slot = it->second;
      }
      if (slot < 0) {
        // First fifo for this base, or a same-name fifo that failed a gate: open
        // a fresh group. Only the first owns the name-base map slot; a later
        // mismatching fifo becomes an unindexed standalone group.
        Group g;
        g.rep = i;
        g.name = base.str();
        g.elemTy = fifos[i].elemTy;
        g.sigDepth = fifos[i].depth;
        g.sigRepeat = fifos[i].repeat;
        g.depth = fifos[i].depth;
        groupList.push_back(std::move(g));
        slot = (int)groupList.size() - 1;
        if (it == groupIdx.end())
          groupIdx[base] = slot;
      } else {
        groupList[slot].depth =
            std::min(groupList[slot].depth, fifos[i].depth);
      }
      Group &g = groupList[slot];
      g.fan += (long)fifos[i].cons.size();
      for (mlir::Value c : fifos[i].cons)
        g.cons.insert(c);
      g.sccs.insert(sccId[i]);
    }

    // Replay guard (mirrors sdf_checker.py): the multicast back-pressure cycle
    // only forms when the broadcast producer must RE-acquire a slot (trip t+1)
    // while the prior trip's tokens are still outstanding -- i.e. T >= 2. A
    // single-trip (T == 1) broadcast fans out once and drains monotonically, so
    // it never deadlocks regardless of fan-out (confirmed: bundled whole_array
    // matmul examples broadcast to the whole array at depth 2, T == 1, and RUN
    // on device).
    for (Group &g : groupList) {
      long T = 1; // trip-count from the component(s) coupled to THIS broadcast
      for (unsigned i = 0; i < n; ++i)
        if (sccId[i] >= 0 && g.sccs.count(sccId[i]))
          T = std::max(T, fifos[i].repeat);
      long demand = g.fan * T;
      long slack = 2 * g.depth;
      if (!(g.depth > 0 && T >= 2 && demand > slack))
        continue;
      long reqDepth = (demand + 1) / 2; // ceil(demand / 2)
      fifos[g.rep].op->emitError()
          << "objectFIFO @" << g.name
          << " in a cyclic dependency requires depth >= " << reqDepth
          << " for deadlock-free execution; allocated depth = " << g.depth
          << ". (Coupled broadcast fan-out " << g.fan << " x trip-count " << T
          << " = " << demand
          << " outstanding tokens exceeds round-trip slack 2*depth = " << slack
          << "; raise depth or reduce coupled consumers / output tiles.)";
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoLivenessPass() {
  return std::make_unique<AIEObjectFifoLivenessPass>();
}
