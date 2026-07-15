//===- ExecutionEngine.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_EXECUTIONENGINE_H
#define AIECC_EXECUTIONENGINE_H

#include "Graph.h"

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

// Backward-reachable set of edges from the requested `outputs` (the same
// pruning the engine uses to decide what to run).
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
// is unknown or stays ambiguous. Shared by `--get` output selection and
// `--resume` checkpoint resolution so the two identify edges the same way.
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
// also shows where a checkpoint would cut it: every edge backward-reachable
// from the cut is the "prefix" a checkpoint runs (green, the frontier edges
// darker); and every dependency arrow that crosses from the prefix into the
// downstream "suffix" — the part a --resume runs (for the first time, after
// reloading the frontier from disk) — is drawn as a dashed red "cut" edge.
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
    // Cut-aware styling (only when a cut is present): the frontier edges get a
    // darker green fill, the rest of the saved prefix a lighter green, and
    // requested outputs on the suffix side stay blue.
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
      // saved prefix but its consumer is not: this is an edge a --resume severs
      // (loading the frontier from disk) and runs downstream of (for the first
      // time — nothing in the prefix is executed again).
      if (!cutEdges.empty() && prefix.count(n->producer) &&
          !prefix.count(e.get()))
        os << " [color=red, style=dashed, penwidth=2, "
              "label=\"cut\", fontcolor=red]";
      os << ";\n";
    }
  }
  os << "}\n";
}

struct Engine {
  struct Options {
    std::string outputDir; // where persisted outputs go
    std::string workDir;   // where intermediates go
    bool verbose = false;
    bool progress = false;
    bool keepIntermediates = false;
    unsigned numThreads = 1; // 0 = auto-detect; 1 = sequential
    // When non-empty, only output items whose key is listed are written to the
    // output directory (the --get-key filter); empty writes every key.
    std::vector<std::string> outputKeyFilter;
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
  mlir::LogicalResult
  run(Graph &g, const std::vector<EdgeBase *> &outputs,
      const llvm::DenseMap<EdgeBase *, llvm::StringMap<std::string>>
          &satisfied = llvm::DenseMap<EdgeBase *, llvm::StringMap<std::string>>{
              0}) {
    llvm::sys::fs::create_directories(opts.workDir);
    if (!opts.outputDir.empty())
      llvm::sys::fs::create_directories(opts.outputDir);

    // Walk dependencies backwards from the requested outputs to find the set
    // of edges that actually need to run. Anything not reached is pruned.
    llvm::DenseSet<EdgeBase *> reachable;
    {
      std::vector<EdgeBase *> stack(outputs.begin(), outputs.end());
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
      // --get-key: keep only the listed keys of the requested outputs in the
      // output dir; the rest spill to the work dir. Execution is unaffected —
      // this only routes where each produced artifact lands.
      if (out && !opts.outputKeyFilter.empty()) {
        e->outputKeys = opts.outputKeyFilter;
        e->spilloverDir = opts.workDir;
      }
    }
    llvm::scope_exit clearWiring([&]() {
      for (auto &e : g.edges) {
        e->outputDir.clear();
        e->outputKeys.clear();
        e->spilloverDir.clear();
      }
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
      size_t inItems = 0;
      for (NodeBase *n : e->inputNodes())
        if (n)
          inItems += n->itemRefs().size();
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
    for (auto &e : g.edges) {
      if (!reachable.count(e.get()))
        continue;
      ++step;
      if (opts.verbose || opts.progress)
        reportEdge(e.get());
      auto sat = satisfied.find(e.get());
      auto edgeStart = std::chrono::steady_clock::now();
      mlir::LogicalResult r =
          sat != satisfied.end() ? e->loadFromDisk(sat->second) : e->execute();
      if (opts.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - edgeStart)
                      .count();
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
    return mlir::success();
  }
};

// Write a checkpoint of a graph cut: copy each cut edge's produced artifacts
// into `dir` alongside a `manifest.json` recording {argv, frontier} where each
// frontier entry is {name, key, path}. A later
// `aiecc --resume=<dir>/manifest.json` reloads these as graph leaves and
// continues (optionally narrowing the suffix with `--get`). Every item of each
// cut edge is captured — the checkpoint is a complete snapshot of the cut.
inline void writeCheckpoint(llvm::ArrayRef<EdgeBase *> cutEdges,
                            llvm::StringRef dir,
                            llvm::ArrayRef<std::string> argv) {
  if (llvm::sys::fs::create_directories(dir)) {
    llvm::errs() << "aiecc: could not create checkpoint dir '" << dir << "'\n";
    return;
  }
  llvm::json::Array frontier;
  llvm::StringSet<> used;
  for (EdgeBase *e : cutEdges) {
    NodeBase *out = e ? e->outputNode() : nullptr;
    if (!out)
      continue;
    for (const ItemBase *it : out->itemRefs()) {
      const std::string &src = it->asFile();
      // Unique destination name to avoid collisions across cut edges / keys.
      std::string base = llvm::sys::path::filename(it->filePath).str();
      std::string destName = base;
      for (unsigned i = 1; !used.insert(destName).second; ++i)
        destName = std::to_string(i) + "_" + base;
      llvm::SmallString<256> dest(dir);
      llvm::sys::path::append(dest, destName);
      if (std::error_code ec = llvm::sys::fs::copy_file(src, dest)) {
        llvm::errs() << "aiecc: checkpoint failed to copy '" << src << "' to '"
                     << dest << "': " << ec.message() << "\n";
        continue;
      }
      frontier.push_back(llvm::json::Object{
          {"name", e->name}, {"key", it->key}, {"path", destName}});
    }
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
