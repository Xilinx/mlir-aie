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
// This pass builds the objectFIFO data + back-pressure dependency graph, finds
// cyclic strongly-connected components (Tarjan), and applies the validated SDF
// model PER coupled-multicast group (grouped by name base) so an unrelated
// cycle elsewhere in the device cannot inflate a group's demand:
//
//   demand   = array_fan * T          (array_fan = the group's multicast
//                                        fan-out, summed across the SCCs it
//                                        spans; T = max repeat_count among the
//                                        fifos in those SCCs)
//   slack    = 2 * depth              (depth = min depth in the group)
//   DEADLOCK iff  T >= 2  AND  depth > 0  AND  demand > slack
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
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <functional>
#include <map>
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

    // find_cycles: Tarjan SCC; assign an SCC id to every node in a cycle
    // (component of size > 1). Nodes not in any cycle keep sccId = -1.
    SmallVector<int> idx(n, -1), low(n, 0);
    SmallVector<char> onStack(n, 0);
    SmallVector<unsigned> stk;
    int counter = 0;
    SmallVector<int> sccId(n, -1);
    int sccCount = 0;
    std::function<void(unsigned)> dfs = [&](unsigned v) {
      idx[v] = low[v] = counter++;
      stk.push_back(v);
      onStack[v] = 1;
      for (unsigned w : adj[v]) {
        if (idx[w] == -1) {
          dfs(w);
          low[v] = std::min(low[v], low[w]);
        } else if (onStack[w]) {
          low[v] = std::min(low[v], idx[w]);
        }
      }
      if (low[v] == idx[v]) {
        SmallVector<unsigned> comp;
        while (true) {
          unsigned w = stk.back();
          stk.pop_back();
          onStack[w] = 0;
          comp.push_back(w);
          if (w == v)
            break;
        }
        if (comp.size() > 1) {
          for (unsigned w : comp)
            sccId[w] = sccCount;
          ++sccCount;
        }
      }
    };
    for (unsigned v = 0; v < n; ++v)
      if (idx[v] == -1)
        dfs(v);

    // SDF check, scoped PER coupled-multicast group -- NOT globally. A coupled
    // broadcast is a name-base group of cyclic multicast objectFIFOs (e.g.
    // memW1_0 + memW1_1 = one weight broadcast, split across MemTiles); the
    // halves can land in different SCCs, so the group -- not the SCC -- is the
    // unit of coupling, and array_fan SUMS the group's fan-out across the SCCs
    // it spans. The trip-count T is taken ONLY from the SCC(s) this group lives
    // in, so an unrelated high-repeat_count (or a wide broadcast) elsewhere in
    // the device cannot inflate this group's demand and false-positive.
    struct Group {
      long fan = 0, depth = 0;
      bool haveDepth = false, haveRep = false;
      unsigned rep = 0;
      std::set<int> sccs;
    };
    std::map<std::string, Group> groups;
    for (unsigned i = 0; i < n; ++i) {
      if (sccId[i] < 0 || fifos[i].cons.size() <= 1)
        continue; // only a cyclic multicast couples a broadcast
      Group &g = groups[nameBase(fifos[i].name).str()];
      g.fan += (long)fifos[i].cons.size();
      g.depth =
          g.haveDepth ? std::min(g.depth, fifos[i].depth) : fifos[i].depth;
      g.haveDepth = true;
      g.sccs.insert(sccId[i]);
      if (!g.haveRep) {
        g.rep = i;
        g.haveRep = true;
      }
    }

    // Replay guard (mirrors sdf_checker.py): the multicast back-pressure cycle
    // only forms when the broadcast producer must RE-acquire a slot (trip t+1)
    // while the prior trip's tokens are still outstanding -- i.e. T >= 2. A
    // single-trip (T == 1) broadcast fans out once and drains monotonically, so
    // it never deadlocks regardless of fan-out (confirmed: bundled whole_array
    // matmul examples broadcast to the whole array at depth 2, T == 1, and RUN
    // on device).
    for (auto &kv : groups) {
      Group &g = kv.second;
      long T = 1; // trip-count from the SCC(s) coupled to THIS broadcast only
      for (unsigned i = 0; i < n; ++i)
        if (sccId[i] >= 0 && g.sccs.count(sccId[i]))
          T = std::max(T, fifos[i].repeat);
      long demand = g.fan * T;
      long slack = 2 * g.depth;
      if (!(g.depth > 0 && T >= 2 && demand > slack))
        continue;
      long reqDepth = (demand + 1) / 2; // ceil(demand / 2)
      fifos[g.rep].op->emitError()
          << "objectFIFO @" << kv.first
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
