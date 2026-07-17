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

    // Per-edge progress reporting (--verbose / --progress);
    // --verbose logs one line per edge; --progress overwrites a single line
    // with a leading '\r'.
    const unsigned totalSteps = reachable.size();
    unsigned step = 0;
    size_t progressPrevLen = 0;
    std::mutex progressMutex; // serializes concurrent per-item updates
    std::string progressBase; // base line for the current edge
    // Overwrite the previous status; pad to clear any leftover characters.
    auto writeProgressLine = [&](const std::string &line) {
      size_t pad =
          progressPrevLen > line.size() ? progressPrevLen - line.size() : 0;
      llvm::errs() << '\r' << line << std::string(pad, ' ');
      progressPrevLen = line.size();
    };
    auto reportEdge = [&](EdgeBase *e) {
      std::string line = "(" + std::to_string(step) + "/" +
                         std::to_string(totalSteps) + ") " +
                         displayName(e).str();
      if (opts.progress) {
        std::lock_guard<std::mutex> lock(progressMutex);
        progressBase = line;
        writeProgressLine(line);
      } else {
        llvm::errs() << "aiecc: " << line << "\n";
      }
    };

    // Per-item fan-out progress
    if (opts.progress) {
      itemProgressHook() = [&](size_t done, size_t total) {
        if (total <= 1)
          return;
        std::lock_guard<std::mutex> lock(progressMutex);
        writeProgressLine(progressBase + ", " + std::to_string(done) + "/" +
                          std::to_string(total));
      };
    }
    llvm::scope_exit clearItemHook([&]() { itemProgressHook() = nullptr; });

    // Effective worker count: 0 auto-detects the hardware concurrency; 1 runs
    // fully sequentially. Thread-safe edges (external-tool invocations) fan
    // their per-core item loops out across a pool of this size; see
    // parallelThreadCount() / MapEdge / BundleForEachEdge.
    unsigned effThreads = opts.numThreads;
    if (effThreads == 0) {
      effThreads = std::thread::hardware_concurrency();
      if (effThreads == 0)
        effThreads = 1;
    }
    parallelThreadCount() = effThreads;
    llvm::scope_exit resetThreads([&]() { parallelThreadCount() = 1; });

    llvm::StringSet<> seenPaths;
    std::vector<std::pair<std::string, int64_t>> edgeTimings;
    for (auto &e : g.edges) {
      if (!reachable.count(e.get()))
        continue;
      ++step;
      if (opts.verbose || opts.progress)
        reportEdge(e.get());
      auto sat = satisfied.find(e.get());
      auto edgeStart = std::chrono::steady_clock::now();
      mlir::LogicalResult r = sat != satisfied.end()
                                  ? e->restoreNode(sat->second.descriptor,
                                                   sat->second.dir, deserCtx)
                                  : e->execute();
      if (opts.verbose || opts.profile) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - edgeStart)
                      .count();
        if (opts.profile)
          edgeTimings.emplace_back(displayName(e.get()).str(), ms);
        if (opts.verbose)
          llvm::errs() << "aiecc: edge '" << displayName(e.get()) << "' took "
                       << ms << " ms\n";
      }
      if (mlir::failed(r)) {
        failedEdge = e.get();
        if (opts.progress)
          llvm::errs() << '\n';
        llvm::errs() << "aiecc: edge '" << displayName(e.get()) << "'";
        if (!e->failedKey.empty())
          llvm::errs() << " (key '" << e->failedKey << "')";
        llvm::errs() << " failed\n";
        return mlir::failure();
      }
      if (!e->producesFiles)
        continue;
      if (NodeBase *out = e->outputNode()) {
        for (const ItemBase *item : out->itemRefs()) {
          if (item->filePath.empty())
            continue;
          if (!seenPaths.insert(item->filePath).second) {
            if (opts.progress)
              llvm::errs() << '\n';
            llvm::errs() << "aiecc: edge '" << displayName(e.get())
                         << "' produced duplicate output path '"
                         << item->filePath << "'\n";
            return mlir::failure();
          }
        }
      }
    }
    // Terminate the single progress line before any further output.
    if (opts.progress)
      llvm::errs() << '\n';

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
