//===- ExecutionEngine.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_EXECUTIONENGINE_H
#define AIECC_EXECUTIONENGINE_H

#include "Graph.h"
#include "Utils.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xilinx::aiecc {

// Backward-reachable set of edges from the requested `outputs`
inline llvm::DenseSet<EdgeBase *>
reachableEdges(const std::vector<EdgeBase *> &outputs) {
  llvm::DenseSet<EdgeBase *> reachable;
  std::vector<EdgeBase *> stack(outputs.begin(), outputs.end());
  while (!stack.empty()) {
    EdgeBase *e = stack.back();
    stack.pop_back();
    if (!e || !reachable.insert(e).second)
      continue;
    for (NodeBase *n : e->inputNodes())
      if (n && n->producer)
        stack.push_back(n->producer);
  }
  return reachable;
}

// Resolve an output name to its unique live edge. A few names are shared by two
// edges by design (chess vs. peano "elfs_{0}.elf", per-core vs. unified
// "objects_{0}.o"); exactly one is live in a given build, so disambiguate by
// membership in `live`. Returns the edge, or an Error describing why the name
// is unknown or stays ambiguous.
inline llvm::Expected<EdgeBase *>
resolveLiveEdge(const Graph &g, llvm::StringRef name,
                const llvm::DenseSet<EdgeBase *> &live) {
  auto err = [](llvm::Twine msg) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
  };
  llvm::SmallVector<EdgeBase *, 2> matches = g.edgesByName(name);
  if (matches.empty())
    return err(llvm::Twine("unknown output '") + name + "'");
  if (matches.size() == 1)
    return matches.front();
  EdgeBase *chosen = nullptr;
  for (EdgeBase *e : matches) {
    if (!live.count(e))
      continue;
    if (chosen)
      return err(llvm::Twine("ambiguous output '") + name +
                 "' matches multiple live edges");
    chosen = e;
  }
  if (!chosen)
    return err(llvm::Twine("output '") + name + "' is not part of this build");
  return chosen;
}

// Resolve a list of output names (as accepted by --get / --cut) to their live
// edges via resolveLiveEdge. Returns the resolved edges in order, or the first
// name's Error; the caller owns how that error is reported (see aiecc.cpp).
inline llvm::Expected<std::vector<EdgeBase *>>
resolveLiveEdges(const Graph &g, llvm::ArrayRef<std::string> names,
                 const llvm::DenseSet<EdgeBase *> &live) {
  std::vector<EdgeBase *> resolved;
  resolved.reserve(names.size());
  for (llvm::StringRef name : names) {
    llvm::Expected<EdgeBase *> e = resolveLiveEdge(g, name, live);
    if (!e)
      return e.takeError();
    resolved.push_back(*e);
  }
  return resolved;
}

// Emit a GraphViz `dot` description of the sub-graph reachable from the
// requested `outputs` (pipe through `dot`). Each node is a compilation-graph
// edge labeled by its output-name template; an arrow runs from a producing
// edge to each edge that consumes its output. Requested outputs are
// highlighted.
//
// When `cutEdges` is non-empty (the --checkpoint/--get frontier), the graph
// also shows where a checkpoint would cut it.
inline void writeDotGraph(const Graph &g,
                          const std::vector<EdgeBase *> &outputs,
                          llvm::raw_ostream &os,
                          llvm::ArrayRef<EdgeBase *> cutEdges = {}) {
  llvm::DenseSet<EdgeBase *> reachable = reachableEdges(outputs);
  llvm::DenseSet<EdgeBase *> outputSet(outputs.begin(), outputs.end());

  // The checkpoint cut: the frontier edges plus the prefix (everything needed
  // to produce them). An arrow from a prefix edge to a non-prefix consumer
  // crosses the cut.
  llvm::DenseSet<EdgeBase *> cutSet(cutEdges.begin(), cutEdges.end());
  llvm::DenseSet<EdgeBase *> prefix;
  if (!cutEdges.empty())
    prefix = reachableEdges(
        std::vector<EdgeBase *>(cutEdges.begin(), cutEdges.end()));

  // Stable integer ids for the reachable edges (in declaration order).
  llvm::DenseMap<EdgeBase *, unsigned> id;
  unsigned next = 0;
  for (auto &e : g.edges)
    if (reachable.count(e.get()))
      id[e.get()] = next++;

  auto escape = [](llvm::StringRef s) {
    std::string out;
    for (char c : s) {
      if (c == '"' || c == '\\')
        out.push_back('\\');
      out.push_back(c);
    }
    return out;
  };

  os << "digraph aiecc {\n";
  os << "  rankdir=LR;\n";
  os << "  node [shape=box, fontname=\"monospace\"];\n";
  for (auto &e : g.edges) {
    if (!reachable.count(e.get()))
      continue;
    os << "  n" << id[e.get()] << " [label=\"" << escape(e->name) << "\"";
    if (cutSet.count(e.get()))
      os << ", style=filled, fillcolor=\"#8fd694\"";
    else if (prefix.count(e.get()))
      os << ", style=filled, fillcolor=\"#cdeccd\"";
    else if (outputSet.count(e.get()))
      os << ", style=filled, fillcolor=lightblue";
    os << "];\n";
  }
  for (auto &e : g.edges) {
    if (!reachable.count(e.get()))
      continue;
    for (NodeBase *n : e->inputNodes()) {
      if (!n || !n->producer || !reachable.count(n->producer))
        continue;
      os << "  n" << id[n->producer] << " -> n" << id[e.get()];
      // The dependency crosses the checkpoint cut when its producer is in the
      // saved prefix but its consumer is not
      if (!cutEdges.empty() && prefix.count(n->producer) &&
          !prefix.count(e.get()))
        os << " [color=red, style=dashed, penwidth=2, "
              "label=\"cut\", fontcolor=red]";
      os << ";\n";
    }
  }
  os << "}\n";
}

// A checkpoint node to restore in place of executing an edge: the JSON
// descriptor captureNode produced plus the directory its artifacts live in.
struct RestoredNode {
  llvm::json::Value descriptor = nullptr;
  std::string dir;
};

struct Engine {
  struct Options {
    std::string outputDir; // where persisted outputs go
    std::string workDir;   // where intermediates go
    bool verbose = false;
    bool progress = false;
    bool keepIntermediates = false;
    unsigned numThreads = 1; // 0 = auto-detect; 1 = sequential
    bool profile = false; // print a per-edge execution-time summary at the end
  };

  Options opts;

  // Set to the edge whose execution failed (first failure wins under
  // parallelism); consulted by the caller to dump an on-failure reproducer.
  EdgeBase *failedEdge = nullptr;

  explicit Engine(Options o) : opts(std::move(o)) {}

  // Fine-grained (edge, key) task scheduler for the reachable sub-graph.
  //
  // A fan-out edge (Map/BundleForEach) is `open()`ed to create its per-key
  // output slots and then runs one task per key ("item"); a structural edge
  // (split / join / filter / rekey / ...) runs as a single whole-edge
  // `execute()`. A task becomes runnable as soon as the specific inputs it
  // consumes are ready -- per key for fan-out edges -- so a core's downstream
  // work (e.g. linking its ELF) can begin while other cores are still
  // compiling, rather than waiting on a whole-edge barrier.
  //
  // A task is "exclusive" when its edge is not marked threadSafe: such edges
  // may touch shared, non-thread-safe state (today, the shared MLIRContext), so
  // at most one exclusive task runs at a time -- `exclusiveBusy` is that
  // mutual-exclusion slot. threadSafe tasks (context-free external-tool
  // invocations) run in parallel across the whole pool and alongside the one
  // in-flight exclusive task.
  struct Scheduler {
    // Per-edge scheduling state.
    struct EdgeState {
      EdgeBase *edge = nullptr;
      bool fan = false;         // fan-out (Map/BundleForEach) vs structural
      bool threadSafe = false;  // may run concurrently with any other task
      bool opened = false;      // fan: slots created (keys final); else ==done
      bool finished = false;    // all of this edge's work is done
      bool wholeQueued = false; // whole-edge task already enqueued
      unsigned running = 0;     // in-flight tasks of this edge
      bool started = false;
      std::chrono::steady_clock::time_point startTime;
      size_t itemsTotal = 0;
      size_t itemsDone = 0;
      std::vector<uint8_t> itemState; // 0 pending, 1 queued/running, 2 done
      llvm::StringMap<size_t> keyIndex;
    };

    // A unit of work: `item < 0` is a whole-edge task, `item >= 0` a per-key
    // item task.
    struct Task {
      unsigned edge;
      int item;
    };

    const Options &opts;
    std::vector<EdgeBase *> edges; // reachable edges, in topological order
    llvm::DenseMap<EdgeBase *, unsigned> edgeIndex;
    const llvm::DenseMap<EdgeBase *, RestoredNode> &satisfied;
    const DeserializeContext &deserCtx; // forwarded to restoreNode on --resume
    std::vector<EdgeState> st;
    // Per-edge wall time, in finish order; summarized slowest-first under
    // --profile.
    std::vector<std::pair<std::string, int64_t>> edgeTimings;

    std::mutex mtx;
    std::condition_variable cv;
    std::vector<Task> ready;
    llvm::StringSet<> seenPaths; // duplicate-output-path guard
    unsigned numEdges = 0;
    unsigned finishedEdges = 0;
    unsigned inFlight = 0;
    bool exclusiveBusy = false; // an exclusive (non-threadSafe) task is running
    bool failed = false;
    bool stalled = false;
    size_t progressPrevLen = 0;
    EdgeBase *failedEdge = nullptr;

    Scheduler(const Options &o, std::vector<EdgeBase *> reachableEdges,
              const llvm::DenseMap<EdgeBase *, RestoredNode> &sat,
              const DeserializeContext &dc)
        : opts(o), edges(std::move(reachableEdges)), satisfied(sat),
          deserCtx(dc) {
      numEdges = edges.size();
      st.resize(numEdges);
      for (unsigned u = 0; u < numEdges; ++u) {
        edgeIndex[edges[u]] = u;
        st[u].edge = edges[u];
        st[u].fan = edges[u]->isFanOut();
        st[u].threadSafe = edges[u]->isThreadSafe;
      }
    }

    // An edge's `name` doubles as its output-path template and may embed
    // directories; show just the file name for progress/logging.
    static llvm::StringRef displayName(EdgeBase *e) {
      return llvm::sys::path::filename(e->name);
    }

    // Scheduling state of the edge that produced node `n` (null if `n` has no
    // producer in the reachable set).
    EdgeState *stateOf(NodeBase *n) {
      if (!n || !n->producer)
        return nullptr;
      auto it = edgeIndex.find(n->producer);
      return it == edgeIndex.end() ? nullptr : &st[it->second];
    }
    // A node's key set is final once its producer is opened (fan-out) or
    // finished (structural edges open and finish together).
    bool nodeFinalKeys(NodeBase *n) {
      EdgeState *s = stateOf(n);
      return !s || s->opened;
    }
    // A node is complete once every one of its items has been produced.
    bool nodeComplete(NodeBase *n) {
      EdgeState *s = stateOf(n);
      return !s || s->finished;
    }
    // Has the item keyed `key` in node `n` been produced yet? A key absent from
    // an opened producer is treated as "done" so the consuming action can
    // surface the incompatible-keys error itself.
    bool nodeItemDone(NodeBase *n, llvm::StringRef key) {
      EdgeState *s = stateOf(n);
      if (!s || s->finished)
        return true;
      if (s->fan && s->opened) {
        auto it = s->keyIndex.find(key);
        if (it == s->keyIndex.end())
          return true;
        return s->itemState[it->second] == 2;
      }
      return false;
    }

    // Record an edge's output paths in the duplicate-path guard; returns false
    // (after a diagnostic) on a collision. Called under `mtx`.
    bool recordPaths(unsigned u) {
      EdgeBase *e = st[u].edge;
      if (!e->producesFiles || !e->outputNode())
        return true;
      for (const ItemBase *item : e->outputNode()->itemRefs()) {
        if (item->filePath.empty())
          continue;
        if (!seenPaths.insert(item->filePath).second) {
          if (opts.progress)
            llvm::errs() << '\n';
          llvm::errs() << "aiecc: edge '" << displayName(e)
                       << "' produced duplicate output path '" << item->filePath
                       << "'\n";
          return false;
        }
      }
      return true;
    }

    // Create a fan-out edge's output slots (its key set becomes final). Empty
    // fan-outs finalize immediately. Called under `mtx`.
    void openEdge(unsigned u) {
      EdgeState &s = st[u];
      if (mlir::failed(s.edge->open())) {
        failed = true;
        failedEdge = s.edge;
        return;
      }
      s.opened = true;
      s.itemsTotal = s.edge->numItems();
      s.itemState.assign(s.itemsTotal, 0);
      s.keyIndex.clear();
      for (size_t i = 0; i < s.itemsTotal; ++i)
        s.keyIndex[s.edge->itemKey(i)] = i;
      if (s.itemsTotal == 0) {
        s.finished = true;
        ++finishedEdges;
        if (!recordPaths(u))
          failed = true;
      }
    }

    // Single-line progress: "(done/total) [i/n] edge_1, [j/m] edge_2, ...".
    // The list is the edges with an in-flight task; i/n is that edge's per-item
    // fan-out progress (0/1 for whole-edge tasks). Called under `mtx`.
    void renderProgress() {
      if (!opts.progress)
        return;
      std::string line = "(" + std::to_string(finishedEdges) + "/" +
                         std::to_string(numEdges) + ")";
      bool first = true;
      for (unsigned u = 0; u < numEdges; ++u) {
        if (st[u].running == 0)
          continue;
        size_t total = st[u].fan ? st[u].itemsTotal : 1;
        size_t done = st[u].fan ? st[u].itemsDone : 0;
        line += (first ? " " : ", ");
        first = false;
        line += "[" + std::to_string(done) + "/" + std::to_string(total) +
                "] " + displayName(st[u].edge).str();
      }
      size_t pad =
          progressPrevLen > line.size() ? progressPrevLen - line.size() : 0;
      std::lock_guard<std::mutex> log(logMutex());
      llvm::errs() << '\r' << line << std::string(pad, ' ');
      llvm::errs().flush();
      progressPrevLen = line.size();
    }

    // Re-scan the reachable edges and enqueue every task that is now runnable.
    // Opening a fan-out edge can unblock its downstream edges, so iterate to a
    // fixpoint. Called under `mtx`.
    void rescan() {
      bool changed = true;
      while (changed && !failed) {
        changed = false;
        for (unsigned u = 0; u < numEdges && !failed; ++u) {
          EdgeState &s = st[u];
          if (s.finished)
            continue;
          std::vector<NodeBase *> ins = s.edge->inputNodes();
          if (!s.fan) {
            if (s.wholeQueued || s.running)
              continue;
            bool depsMet = true;
            for (NodeBase *n : ins)
              if (!nodeComplete(n)) {
                depsMet = false;
                break;
              }
            if (depsMet) {
              ready.push_back({u, -1});
              s.wholeQueued = true;
              changed = true;
            }
            continue;
          }
          if (!s.opened) {
            bool keysReady = true;
            for (NodeBase *n : ins)
              if (!nodeFinalKeys(n)) {
                keysReady = false;
                break;
              }
            if (keysReady) {
              openEdge(u);
              changed = true;
              continue;
            }
          }
          if (s.opened) {
            for (size_t i = 0; i < s.itemsTotal; ++i) {
              if (s.itemState[i] != 0)
                continue;
              llvm::StringRef key = s.edge->itemKey(i);
              bool depsMet = true;
              for (NodeBase *n : ins)
                if (!nodeItemDone(n, key)) {
                  depsMet = false;
                  break;
                }
              if (depsMet) {
                ready.push_back({u, static_cast<int>(i)});
                s.itemState[i] = 1;
                changed = true;
              }
            }
          }
        }
      }
    }

    // Pick the first dispatchable ready task: threadSafe tasks always qualify;
    // an exclusive (non-threadSafe) task only when the exclusive slot is free.
    // Removes and returns it, claiming the slot if needed. Called under `mtx`.
    bool pickTask(Task &out) {
      for (size_t i = 0; i < ready.size(); ++i) {
        bool ts = st[ready[i].edge].threadSafe;
        if (ts || !exclusiveBusy) {
          out = ready[i];
          ready.erase(ready.begin() + i);
          if (!ts)
            exclusiveBusy = true;
          return true;
        }
      }
      return false;
    }

    // Verbose "(x/y) name (a inputs) took N ms" line for a just-finished edge.
    // Called under `mtx`.
    void logEdgeFinished(const EdgeState &s) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - s.startTime)
                    .count();
      size_t inItems = 0;
      for (NodeBase *n : s.edge->inputNodes())
        if (n)
          inItems += n->itemRefs().size();
      std::string msg = "aiecc: (" + std::to_string(finishedEdges) + "/" +
                        std::to_string(numEdges) + ") " +
                        displayName(s.edge).str() + " (" +
                        std::to_string(inItems) + " inputs) took " +
                        std::to_string(ms) + " ms\n";
      std::lock_guard<std::mutex> log(logMutex());
      llvm::errs() << msg;
      llvm::errs().flush();
    }

    // Worker loop: pull a dispatchable task, run it without the lock, then fold
    // its completion back into the schedule and wake peers.
    void worker() {
      std::unique_lock<std::mutex> lock(mtx);
      for (;;) {
        Task task{0, -1};
        while (!failed && finishedEdges < numEdges && !pickTask(task)) {
          if (ready.empty() && inFlight == 0) {
            // Nothing runnable and nothing in flight, yet work remains: a
            // dependency stall (a scheduler bug). Fail rather than hang.
            stalled = true;
            failed = true;
            cv.notify_all();
            break;
          }
          cv.wait(lock);
        }
        if (failed || finishedEdges >= numEdges)
          break;

        EdgeState &s = st[task.edge];
        if (!s.started) {
          s.started = true;
          s.startTime = std::chrono::steady_clock::now();
        }
        ++s.running;
        ++inFlight;
        renderProgress();
        lock.unlock();

        // Run the task without the scheduler lock.
        mlir::LogicalResult r = task.item < 0
                                    ? s.edge->execute()
                                    : s.edge->runItem((size_t)task.item);

        lock.lock();
        --s.running;
        --inFlight;
        if (!s.threadSafe)
          exclusiveBusy = false;

        if (mlir::failed(r)) {
          if (!failed) {
            failed = true;
            failedEdge = s.edge;
          }
          cv.notify_all();
          continue;
        }

        bool edgeJustFinished = false;
        if (task.item < 0) {
          s.opened = true;
          s.finished = true;
          edgeJustFinished = true;
        } else {
          s.itemState[task.item] = 2;
          ++s.itemsDone;
          if (s.itemsDone == s.itemsTotal) {
            s.finished = true;
            edgeJustFinished = true;
          }
        }
        if (edgeJustFinished) {
          ++finishedEdges;
          if (opts.verbose)
            logEdgeFinished(s);
          if (opts.profile) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - s.startTime)
                          .count();
            edgeTimings.emplace_back(displayName(s.edge).str(), ms);
          }
          if (!recordPaths(task.edge))
            failed = true;
        }
        if (!failed)
          rescan();
        renderProgress();
        cv.notify_all();
        if (finishedEdges >= numEdges)
          break;
      }
      cv.notify_all();
    }

    // Preload resumed leaves, run the pool to completion, and emit the final
    // failure diagnostic. Returns failure if any task failed.
    mlir::LogicalResult run() {
      // Pre-load resumed (--resume) edges as already-finished leaves so their
      // downstream dependencies resolve without re-executing them.
      for (unsigned u = 0; u < numEdges; ++u) {
        auto sat = satisfied.find(st[u].edge);
        if (sat == satisfied.end())
          continue;
        if (mlir::failed(st[u].edge->restoreNode(
                sat->second.descriptor, sat->second.dir, deserCtx))) {
          failedEdge = st[u].edge;
          failed = true;
          llvm::errs() << "aiecc: edge '" << displayName(st[u].edge)
                       << "' failed to load from checkpoint\n";
          return mlir::failure();
        }
        st[u].opened = true;
        st[u].finished = true;
        ++finishedEdges;
        if (!recordPaths(u)) {
          failed = true;
          return mlir::failure();
        }
      }

      unsigned effThreads = opts.numThreads;
      if (effThreads == 0) {
        effThreads = std::thread::hardware_concurrency();
        if (effThreads == 0)
          effThreads = 1;
      }

      // Seed the ready queue, then run the pool to completion.
      {
        std::lock_guard<std::mutex> lock(mtx);
        rescan();
      }
      if (!failed) {
        std::vector<std::thread> pool;
        pool.reserve(effThreads);
        for (unsigned w = 0; w < effThreads; ++w)
          pool.emplace_back([this] { worker(); });
        for (std::thread &t : pool)
          t.join();
      }

      // Terminate the single progress line before any further output.
      if (opts.progress)
        llvm::errs() << '\n';

      if (failed) {
        if (stalled)
          llvm::errs() << "aiecc: scheduler stalled with unsatisfied "
                          "dependencies\n";
        else if (failedEdge) {
          llvm::errs() << "aiecc: edge '" << displayName(failedEdge) << "'";
          if (!failedEdge->failedKey.empty())
            llvm::errs() << " (key '" << failedEdge->failedKey << "')";
          llvm::errs() << " failed\n";
        }
        return mlir::failure();
      }
      return mlir::success();
    }
  };

  // Run the graph. The full graph is declared up front; only the edges
  // backward-reachable from the requested `outputs` are executed (in
  // insertion order, a valid topo order by construction). Materialization is
  // lazy: intermediates only hit disk when a downstream action calls
  // asFile(); declared outputs (and all reachable edges under
  // --keep-intermediates) are force-written at the end.
  //
  // `deserCtx` is forwarded verbatim to any resumed edge's restoreNode (see
  // --resume / `satisfied`); the engine never inspects it -- it is an opaque
  // capability bag owned by the caller (see Items.h DeserializeContext).
  //
  // `buildAlso` edges are built too (added as reachability roots) but are NOT
  // treated as outputs: they stay in the work dir as intermediates rather than
  // being relocated to the output dir. Used for `--cut` checkpoint frontiers,
  // which must be produced (and captured) without moving them out from under
  // the downstream consumers that reference them by path (e.g. core ELFs the
  // CDO step loads).
  mlir::LogicalResult
  run(Graph &g, const std::vector<EdgeBase *> &outputs,
      const llvm::DenseMap<EdgeBase *, RestoredNode> &satisfied =
          llvm::DenseMap<EdgeBase *, RestoredNode>{},
      const DeserializeContext &deserCtx = {},
      const std::vector<EdgeBase *> &buildAlso = {}) {
    llvm::sys::fs::create_directories(opts.workDir);
    if (!opts.outputDir.empty())
      llvm::sys::fs::create_directories(opts.outputDir);

    // Walk dependencies backwards from the requested outputs (and any extra
    // `buildAlso` roots) to find the set of edges that actually need to run.
    // Anything not reached is pruned.
    llvm::DenseSet<EdgeBase *> reachable;
    {
      std::vector<EdgeBase *> stack(outputs.begin(), outputs.end());
      stack.insert(stack.end(), buildAlso.begin(), buildAlso.end());
      while (!stack.empty()) {
        EdgeBase *e = stack.back();
        stack.pop_back();
        if (!e || !reachable.insert(e).second)
          continue;
        if (satisfied.count(e))
          continue; // resumed from disk: treat as a leaf, prune its inputs
        for (NodeBase *n : e->inputNodes())
          if (n && n->producer)
            stack.push_back(n->producer);
      }
    }

    auto isOutput = [&](EdgeBase *e) {
      return std::find(outputs.begin(), outputs.end(), e) != outputs.end();
    };

    for (auto &e : g.edges) {
      bool out = isOutput(e.get());
      e->outputDir = out ? opts.outputDir : opts.workDir;
    }
    llvm::scope_exit clearWiring([&]() {
      for (auto &e : g.edges)
        e->outputDir.clear();
    });

    // An edge's `name` doubles as its output-path template and may embed
    // directories; show just the file name for progress/logging.
    auto displayName = [](EdgeBase *e) {
      return llvm::sys::path::filename(e->name);
    };

    // Execute the reachable sub-graph as a pool of fine-grained (edge, key)
    // tasks; see Scheduler.
    std::vector<EdgeBase *> reachableEdges;
    for (auto &e : g.edges)
      if (reachable.count(e.get()))
        reachableEdges.push_back(e.get());

    Scheduler scheduler(opts, std::move(reachableEdges), satisfied, deserCtx);
    mlir::LogicalResult r = scheduler.run();
    failedEdge = scheduler.failedEdge;
    if (mlir::failed(r))
      return mlir::failure();
    auto &edgeTimings = scheduler.edgeTimings;

    // Force declared outputs (and, under --keep-intermediates, every reachable
    // edge) to disk.
    for (auto &e : g.edges) {
      if (!reachable.count(e.get()) || !e->producesFiles || !e->outputNode() ||
          (!isOutput(e.get()) && !opts.keepIntermediates))
        continue;
      e->writeOutput();
      if (opts.verbose)
        llvm::errs() << "aiecc: wrote edge '" << displayName(e.get()) << "'\n";
    }

    // --profile: report each edge's execution time, slowest first, plus total.
    if (opts.profile && !edgeTimings.empty()) {
      std::stable_sort(
          edgeTimings.begin(), edgeTimings.end(),
          [](const auto &a, const auto &b) { return a.second > b.second; });
      int64_t total = 0;
      for (const auto &[name, ms] : edgeTimings)
        total += ms;
      llvm::errs() << "aiecc: profile (per-edge execution time):\n";
      for (const auto &[name, ms] : edgeTimings)
        llvm::errs() << llvm::formatv("  {0,8} ms  {1}\n", ms, name);
      llvm::errs() << llvm::formatv("  {0,8} ms  total\n", total);
    }
    return mlir::success();
  }
};

// Write a checkpoint of a graph cut: each cut edge captures its whole output
// node (via captureNode) into its own subdir of `dir`, and a `manifest.json`
// records {argv, frontier} where each frontier entry is {name, dir, descriptor}
// -- the per-node restore descriptor that captureNode returned. A later
// `aiecc --resume=<dir>/manifest.json` rebuilds each node via restoreNode and
// continues.
inline void writeCheckpoint(llvm::ArrayRef<EdgeBase *> cutEdges,
                            llvm::StringRef dir,
                            llvm::ArrayRef<std::string> argv) {
  if (llvm::sys::fs::create_directories(dir)) {
    llvm::errs() << "aiecc: could not create checkpoint dir '" << dir << "'\n";
    return;
  }
  llvm::json::Array frontier;
  llvm::StringSet<> usedDirs;
  for (EdgeBase *e : cutEdges) {
    if (!e || !e->outputNode())
      continue;
    // Each node captures its own group of artifacts into a subdir so shared
    // names (e.g. OpInModule's module.mlir) can't collide across cut edges.
    std::string base = llvm::sys::path::filename(e->name).str();
    std::string sub;
    for (char c : base) {
      bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
      sub += ok ? c : '_';
    }
    if (sub.empty())
      sub = "node";
    std::string uniq = sub;
    for (unsigned i = 1; !usedDirs.insert(uniq).second; ++i)
      uniq = sub + "_" + std::to_string(i);
    llvm::SmallString<256> subAbs(dir);
    llvm::sys::path::append(subAbs, uniq);
    if (llvm::sys::fs::create_directories(subAbs))
      continue;
    llvm::json::Value descriptor = e->captureNode(subAbs);
    frontier.push_back(
        llvm::json::Object{{"name", e->name},
                           {"dir", uniq},
                           {"descriptor", std::move(descriptor)}});
  }
  llvm::json::Array argvArr;
  for (const std::string &a : argv)
    argvArr.push_back(a);
  llvm::json::Object manifest{{"argv", std::move(argvArr)},
                              {"frontier", std::move(frontier)}};
  llvm::SmallString<256> manifestPath(dir);
  llvm::sys::path::append(manifestPath, "manifest.json");
  std::error_code ec;
  llvm::raw_fd_ostream os(manifestPath, ec);
  if (ec)
    return;
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));
  llvm::errs() << "aiecc: wrote checkpoint to " << manifestPath << "\n";
}

} // namespace xilinx::aiecc

#endif // AIECC_EXECUTIONENGINE_H
