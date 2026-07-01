//===- AIEObjectFifoLiveness.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices Inc.
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
// model:
//
//   demand   = array_fan * T          (array_fan = name-base-summed multicast
//                                        fan-out in a cycle == n_cols;
//                                        T = max objectFIFO repeat_count)
//   slack    = 2 * depth
//   DEADLOCK iff  T >= 2  AND  depth > 0  AND  demand > slack
//
// The T >= 2 replay guard avoids false positives on single-trip broadcasts
// (which fan out once and drain monotonically). The analysis is sound for this
// static SDF class and conservative elsewhere (it never errors outside a proven
// cyclic under-buffered multicast).
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

    // find_cycles: Tarjan SCC; keep components of size > 1.
    SmallVector<int> idx(n, -1), low(n, 0);
    SmallVector<char> onStack(n, 0);
    SmallVector<unsigned> stk;
    int counter = 0;
    SmallVector<char> inCycle(n, 0);
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
        if (comp.size() > 1)
          for (unsigned w : comp)
            inCycle[w] = 1;
      }
    };
    for (unsigned v = 0; v < n; ++v)
      if (idx[v] == -1)
        dfs(v);

    // T = max repeat_count over all objectFIFOs (default 1).
    long T = 1;
    for (auto &f : fifos)
      T = std::max(T, f.repeat);

    // array_fan / depth: group cyclic multicasts (cons.size() > 1) by name
    // base, SUM their consumer-counts; array_fan = max group sum, depth = min
    // depth among those cyclic multicasts.
    std::map<std::string, long> groups;
    long depth = 0;
    bool haveDepth = false;
    for (unsigned i = 0; i < n; ++i) {
      if (!inCycle[i])
        continue;
      long Fan = fifos[i].cons.size();
      if (Fan <= 1)
        continue;
      std::string base = nameBase(fifos[i].name).str();
      groups[base] += Fan;
      if (!haveDepth) {
        depth = fifos[i].depth;
        haveDepth = true;
      } else {
        depth = std::min(depth, fifos[i].depth);
      }
    }
    long array_fan = 0;
    for (auto &kv : groups)
      array_fan = std::max(array_fan, kv.second);

    long demand = array_fan * T;
    long slack = 2 * depth;
    // Replay guard (mirrors sdf_checker.py): the multicast back-pressure cycle
    // only forms when the broadcast producer must RE-acquire a slot (trip t+1)
    // while the prior trip's tokens are still outstanding -- i.e. T >= 2. A
    // single-trip (T == 1) broadcast fans out once and drains monotonically, so
    // it never deadlocks regardless of fan-out (confirmed: bundled whole_array
    // matmul examples broadcast to the whole array at depth 2, T == 1, and RUN
    // on device).
    bool deadlock = depth > 0 && T >= 2 && demand > slack;

    if (deadlock) {
      long reqDepth = (demand + 1) / 2; // ceil(demand / 2)
      // Representative: first fifo in walk order whose base is a widest group
      // (fan == array_fan) and which is itself a multicast.
      for (unsigned i = 0; i < n; ++i) {
        if (!inCycle[i] || fifos[i].cons.size() <= 1)
          continue;
        std::string base = nameBase(fifos[i].name).str();
        if (groups[base] != array_fan)
          continue;
        fifos[i].op->emitError()
            << "objectFIFO @" << base
            << " in a cyclic dependency requires depth >= " << reqDepth
            << " for deadlock-free execution; allocated depth = "
            << fifos[i].depth << ". (Coupled broadcast fan-out " << array_fan
            << " x trip-count " << T << " = " << demand
            << " outstanding tokens exceeds round-trip slack 2*depth = " << slack
            << "; raise depth or reduce coupled consumers / output tiles.)";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoLivenessPass() {
  return std::make_unique<AIEObjectFifoLivenessPass>();
}
