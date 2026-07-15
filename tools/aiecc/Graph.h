//===- Graph.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarative compilation graph DSL: payload types, Items (artifacts) carried
// by typed Nodes, Edges (transformations between Nodes), and the Graph that
// owns them. Lazy materialization: payloads only hit disk when downstream
// asks for asFile().
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_GRAPH_H
#define AIECC_GRAPH_H

#include "Items.h"
#include "Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace xilinx::aiecc {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Serializes the lazy "write payload to disk" step in Item::asFile() so that
// concurrent edges (see ExecutionEngine's parallel scheduler) can safely
// materialize intermediates. Each item is normally written by a single
// downstream consumer; this guard makes the rare shared case race-free.
inline std::mutex &fileWriteMutex() {
  static std::mutex m;
  return m;
}

// Serializes diagnostic/progress writes to the shared stdout/stderr so that
// concurrent worker threads (and the tool-invocation echo) don't interleave
// their lines. Any code that prints a full log line from a worker thread should
// hold this across the write (and flush before releasing).
inline std::mutex &logMutex() {
  static std::mutex m;
  return m;
}

//===----------------------------------------------------------------------===//
// Items
//===----------------------------------------------------------------------===//

struct ItemBase {
  std::string key;
  std::string filePath;

  virtual ~ItemBase() = default;

  // Materialize the item to disk if needed and return its path. Cached.
  virtual const std::string &asFile() const = 0;

  // Render the item's value as a string. Only valid for string-like payloads.
  virtual std::string asString() const = 0;

  // Expand the item into a list of argv strings. List-of-string payloads
  // expand to one argv per element (and never materialize a file); everything
  // else yields a single entry. Used by ShellCommand's variadic `inputs()`.
  virtual std::vector<std::string> asArgList() const { return {asString()}; }
};

template <typename T>
struct Item : ItemBase {
  std::optional<T> value;
  mutable bool fileWritten = false;
  // Non-null on items produced by FilterEdge: accessors forward to the source
  // item so payloads aren't cloned. The source outlives this item (edges live
  // for the program's lifetime).
  const Item<T> *aliasSource = nullptr;

  const T &get() const {
    if (aliasSource)
      return aliasSource->get();
    assert(value && "Item has no value (producing action did not populate it)");
    return *value;
  }

  std::vector<std::string> asArgList() const override {
    if (aliasSource)
      return aliasSource->asArgList();
    if constexpr (std::is_same_v<T, std::vector<std::string>>)
      return get();
    else
      return {asString()};
  }

  std::string asString() const override {
    if (aliasSource)
      return aliasSource->asString();
    if constexpr (IsFileLikeV<T>)
      llvm_unreachable("asString() not valid for file-like payloads");
    else {
      assert(value && "Item has no value to render");
      std::string out;
      llvm::raw_string_ostream os(out);
      Serializer<T>::write(*value, os);
      return out;
    }
  }

  const std::string &asFile() const override {
    if (aliasSource)
      return aliasSource->asFile();
    assert(!filePath.empty() && "Item has no assigned filePath");
    if constexpr (IsFileLikeV<T>)
      return filePath;
    else {
      if (fileWritten)
        return filePath;
      std::lock_guard<std::mutex> lock(fileWriteMutex());
      if (fileWritten)
        return filePath;
      assert(value && "Item has no value to write");
      std::error_code ec;
      llvm::raw_fd_ostream os(filePath, ec);
      if (ec) {
        llvm::errs() << "aiecc: could not open '" << filePath
                     << "' for writing: " << ec.message() << "\n";
        return filePath;
      }
      Serializer<T>::write(*value, os);
      fileWritten = true;
      return filePath;
    }
  }
};

// Map action that lifts a File produced by an external tool into a typed
// payload via its Deserializer. Use as the action of a
// `.map<T>(name, deserializeFile<T>())` edge so downstream edges receive the
// already-parsed value.
template <typename T>
auto deserializeFile() {
  return [](const Item<File> &in, Item<T> &out) -> mlir::LogicalResult {
    auto value = Deserializer<T>::read(in.asFile(), DeserializeContext{});
    if (mlir::failed(value))
      return mlir::failure();
    out.value = std::move(*value);
    return mlir::success();
  };
}

//===----------------------------------------------------------------------===//
// Per-Node serialization: capture/restore a whole node's items as a group
//===----------------------------------------------------------------------===//
//
// NodeSerializer::write lays a node's items out under a directory however it
// likes and returns a JSON descriptor; NodeDeserializer::read consumes that
// descriptor to rebuild the items. The default treats each item as an
// independent artifact (delegating to the per-item Serializer/Deserializer).
// Payloads whose items share // structure (OpInModule: one module, many focus
// ops) or carry a companion directory (Directory) specialize this.

// Position of `target` among `KeyOp`s in `module`'s pre-order walk, or -1.
template <typename KeyOp>
inline int64_t opWalkIndex(mlir::ModuleOp module, KeyOp target) {
  int64_t idx = 0, found = -1;
  module.walk([&](KeyOp op) -> mlir::WalkResult {
    if (op == target) {
      found = idx;
      return mlir::WalkResult::interrupt();
    }
    ++idx;
    return mlir::WalkResult::advance();
  });
  return found;
}

// The `KeyOp` at pre-order walk position `target` in `module`, or null.
template <typename KeyOp>
inline KeyOp opAtWalkIndex(mlir::ModuleOp module, int64_t target) {
  int64_t idx = 0;
  KeyOp result;
  module.walk([&](KeyOp op) -> mlir::WalkResult {
    if (idx++ == target) {
      result = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

template <typename T>
struct NodeSerializer {
  // Default: each item is an independent artifact. Materialize each item's file
  // into `dir` and record {key, path}. Requires a per-item Serializer (via
  // asFile()); in-memory-only payloads specialize this instead.
  static llvm::json::Value write(llvm::ArrayRef<Item<T>> items,
                                 llvm::StringRef dir) {
    llvm::json::Array entries;
    llvm::StringSet<> used;
    for (const Item<T> &it : items) {
      llvm::StringRef src = it.asFile();
      std::string base = llvm::sys::path::filename(it.filePath).str();
      std::string name = base;
      for (unsigned i = 1; !used.insert(name).second; ++i)
        name = std::to_string(i) + "_" + base;
      llvm::SmallString<256> dest(dir);
      llvm::sys::path::append(dest, name);
      (void)llvm::sys::fs::copy_file(src, dest);
      entries.push_back(llvm::json::Object{{"key", it.key}, {"path", name}});
    }
    return llvm::json::Object{{"items", std::move(entries)}};
  }
};

template <typename T>
struct NodeDeserializer {
  // Default: rebuild each item independently via Deserializer<T>. A node is
  // resumable iff its payload is (HasDeserializerV).
  static mlir::FailureOr<std::vector<Item<T>>>
  read(const llvm::json::Value &desc, llvm::StringRef dir,
       const DeserializeContext &dc) {
    if constexpr (!HasDeserializerV<T>) {
      llvm::errs() << "aiecc: cannot resume: payload is not deserializable\n";
      return mlir::failure();
    } else {
      const llvm::json::Object *o = desc.getAsObject();
      const llvm::json::Array *entries = o ? o->getArray("items") : nullptr;
      std::vector<Item<T>> items;
      if (!entries)
        return items;
      for (const llvm::json::Value &e : *entries) {
        const llvm::json::Object *eo = e.getAsObject();
        llvm::SmallString<256> p(dir);
        llvm::sys::path::append(p, eo->getString("path").value_or(""));
        auto v = Deserializer<T>::read(p, dc);
        if (mlir::failed(v))
          return mlir::failure();
        Item<T> it;
        it.key = eo->getString("key").value_or("").str();
        it.filePath = std::string(p.str());
        it.fileWritten = true;
        it.value = std::move(*v);
        items.push_back(std::move(it));
      }
      return items;
    }
  }
};

// OpInModule: all items share the same module content and differ only in which
// op is focused. Write the module once and record each item's focus op as its
// pre-order walk index; on restore, parse the module once and rebind.
template <typename KeyOp>
struct NodeSerializer<OpInModule<KeyOp>> {
  static llvm::json::Value write(llvm::ArrayRef<Item<OpInModule<KeyOp>>> items,
                                 llvm::StringRef dir) {
    std::string moduleName = "module.mlir";
    if (!items.empty()) {
      llvm::SmallString<256> dest(dir);
      llvm::sys::path::append(dest, moduleName);
      std::error_code ec;
      llvm::raw_fd_ostream os(dest, ec);
      if (!ec)
        printModuleWithDebugInfo(items.front().get().module.get(), os);
    }
    llvm::json::Array entries;
    for (const Item<OpInModule<KeyOp>> &it : items)
      entries.push_back(llvm::json::Object{
          {"key", it.key},
          {"walkIdx", opWalkIndex<KeyOp>(it.get().module.get(), it.get().op)}});
    return llvm::json::Object{{"module", moduleName},
                              {"items", std::move(entries)}};
  }
};

template <typename KeyOp>
struct NodeDeserializer<OpInModule<KeyOp>> {
  static mlir::FailureOr<std::vector<Item<OpInModule<KeyOp>>>>
  read(const llvm::json::Value &desc, llvm::StringRef dir,
       const DeserializeContext &dc) {
    if (!dc.mlirContext) {
      llvm::errs() << "aiecc: cannot resume: no MLIRContext to parse module\n";
      return mlir::failure();
    }
    const llvm::json::Object *o = desc.getAsObject();
    llvm::SmallString<256> modPath(dir);
    llvm::sys::path::append(modPath, o->getString("module").value_or(""));
    mlir::OwningOpRef<mlir::ModuleOp> parsed =
        parseModuleFromFile(modPath, dc.mlirContext);
    if (!parsed) {
      llvm::errs() << "aiecc: cannot resume: failed to parse '" << modPath
                   << "'\n";
      return mlir::failure();
    }
    const llvm::json::Array *entries = o->getArray("items");
    std::vector<Item<OpInModule<KeyOp>>> items;
    if (!entries)
      return items;
    for (const llvm::json::Value &e : *entries) {
      const llvm::json::Object *eo = e.getAsObject();
      int64_t walkIdx = eo->getInteger("walkIdx").value_or(-1);
      // Each item owns its own module (matching the split); clone per item.
      mlir::OwningOpRef<mlir::ModuleOp> clone(parsed.get().clone());
      KeyOp op = opAtWalkIndex<KeyOp>(clone.get(), walkIdx);
      if (!op) {
        llvm::errs() << "aiecc: cannot resume: focus op index " << walkIdx
                     << " out of range\n";
        return mlir::failure();
      }
      Item<OpInModule<KeyOp>> it;
      it.key = eo->getString("key").value_or("").str();
      it.value = OpInModule<KeyOp>{std::move(clone), op};
      items.push_back(std::move(it));
    }
    return items;
  }
};

// Directory: each item carries a companion folder that must travel whole.
// Copy each item's bundle into its own subdir and record the item's primary
// artifact as a path relative to that subdir; on restore, point the item back
// at the copied folder.
template <>
struct NodeSerializer<Directory> {
  static llvm::json::Value write(llvm::ArrayRef<Item<Directory>> items,
                                 llvm::StringRef dir) {
    llvm::json::Array entries;
    llvm::StringSet<> used;
    for (const Item<Directory> &it : items) {
      const std::string &bundle = it.get().dir;
      std::string base = llvm::sys::path::filename(bundle).str();
      std::string name = base;
      for (unsigned i = 1; !used.insert(name).second; ++i)
        name = std::to_string(i) + "_" + base;
      llvm::SmallString<256> dest(dir);
      llvm::sys::path::append(dest, name);
      if (std::error_code ec = copyDirectoryRecursively(bundle, dest)) {
        llvm::errs() << "aiecc: checkpoint failed to copy bundle '" << bundle
                     << "': " << ec.message() << "\n";
      }
      // Primary artifact's path within the bundle (empty when the item's path
      // *is* the directory, e.g. a CDO).
      llvm::StringRef rel = llvm::StringRef(it.filePath).substr(bundle.size());
      while (rel.starts_with("/"))
        rel = rel.drop_front();
      entries.push_back(llvm::json::Object{
          {"key", it.key}, {"path", name}, {"file", rel.str()}});
    }
    return llvm::json::Object{{"items", std::move(entries)}};
  }
};

template <>
struct NodeDeserializer<Directory> {
  static mlir::FailureOr<std::vector<Item<Directory>>>
  read(const llvm::json::Value &desc, llvm::StringRef dir,
       const DeserializeContext & /*dc*/) {
    const llvm::json::Object *o = desc.getAsObject();
    const llvm::json::Array *entries = o ? o->getArray("items") : nullptr;
    std::vector<Item<Directory>> items;
    if (!entries)
      return items;
    for (const llvm::json::Value &e : *entries) {
      const llvm::json::Object *eo = e.getAsObject();
      llvm::SmallString<256> folder(dir);
      llvm::sys::path::append(folder, eo->getString("path").value_or(""));
      llvm::StringRef rel = eo->getString("file").value_or("");
      llvm::SmallString<256> primary(folder);
      if (!rel.empty())
        llvm::sys::path::append(primary, rel);
      Item<Directory> it;
      it.key = eo->getString("key").value_or("").str();
      it.filePath = std::string(primary.str());
      it.fileWritten = true;
      it.value = Directory{std::string(folder.str())};
      items.push_back(std::move(it));
    }
    return items;
  }
};

struct Graph;
struct EdgeBase;

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

// A Node is a collection of Items (artifacts) produced by an Edge.
// Type-erased base; Node<T> holds the typed items.
struct NodeBase {
  Graph &graph;
  // The edge that produces this node. Lets the engine walk dependencies
  // backwards from the requested outputs and execute only the edges needed.
  EdgeBase *producer = nullptr;

  explicit NodeBase(Graph &g) : graph(g) {}
  virtual ~NodeBase() = default;

  virtual std::vector<const ItemBase *> itemRefs() const = 0;
};

template <typename T>
struct Node : NodeBase {
  std::vector<Item<T>> items;

  using NodeBase::NodeBase;

  const T &get() const {
    assert(items.size() == 1 && "Node::get() requires a singleton");
    return items.front().get();
  }

  std::vector<const ItemBase *> itemRefs() const override {
    std::vector<const ItemBase *> r;
    r.reserve(items.size());
    for (const auto &item : items)
      r.push_back(&item);
    return r;
  }
};

//===----------------------------------------------------------------------===//
// Edges
//===----------------------------------------------------------------------===//

// An Edge transforms one Node to another, either by performing an action (e.g.
// IR -> Binary) or by reorganizing the graph (e.g. splitting IR into
// constituent Ops).
struct EdgeBase {
  Graph &graph;
  std::string name;
  std::string outputDir; // engine sets this before execute()
  bool producesFiles = true;

  // True if execute() is free of shared, mutable state — in particular it does
  // not touch the shared MLIRContext (e.g. external-tool/subprocess edges).
  // The parallel scheduler may run such edges concurrently with any other
  // edge; edges left false are serialized against each other. Set fluently via
  // EdgeWithTypedOutput::threadSafe().
  bool isThreadSafe = false;

  EdgeBase(Graph &g, std::string n) : graph(g), name(std::move(n)) {}
  virtual ~EdgeBase() = default;

  virtual NodeBase *outputNode() = 0;
  virtual mlir::LogicalResult execute() = 0;

  // Fan-out interface. Fan-out edges (Map/BundleForEach) produce one output
  // item per key and are executed by the scheduler as independent (edge, key)
  // tasks: `open()` creates all output slots (making the key set final) without
  // running any per-key work, and `runItem(i)` runs the action for slot `i`.
  // Structural edges leave isFanOut() false and run as a single `execute()`.
  // `execute()` remains valid on fan-out edges (open + sequential runItem) for
  // non-scheduled callers.
  virtual bool isFanOut() const { return false; }
  virtual mlir::LogicalResult open() { return mlir::success(); }
  virtual size_t numItems() const { return 0; }
  virtual llvm::StringRef itemKey(size_t) const { return {}; }
  virtual mlir::LogicalResult runItem(size_t) { return mlir::failure(); }

  // Key of the item whose per-item action failed (first failure wins under
  // parallelism); empty for whole-edge failures. Reported in the failure
  // diagnostic to identify the offending instance.
  std::atomic<bool> failedKeySet{false};
  std::string failedKey;
  void recordFailedKey(llvm::StringRef k) {
    bool expected = false;
    if (failedKeySet.compare_exchange_strong(expected, true))
      failedKey = k.str();
  }

  // Capture this edge's output node into `dir` for a checkpoint, returning a
  // JSON descriptor the resume path feeds back to restoreNode. Default: nothing
  // (edges with no typed output).
  virtual llvm::json::Value captureNode(llvm::StringRef dir) {
    return llvm::json::Value(nullptr);
  }

  // Rebuild this edge's output node from a checkpoint descriptor (see
  // captureNode) instead of executing. Default: unsupported.
  virtual mlir::LogicalResult restoreNode(const llvm::json::Value &desc,
                                          llvm::StringRef dir,
                                          const DeserializeContext &dc) {
    return mlir::failure();
  }

  // The input nodes this edge consumes. The engine follows these (via each
  // node's `producer`) to compute which edges are reachable from the requested
  // outputs. Source edges (e.g. file inputs) have none.
  virtual std::vector<NodeBase *> inputNodes() { return {}; }

  // Force every output item to disk; no-op for edges with no typed output.
  virtual void writeOutput() {}

  // Render the edge's filename template (with `{0}` → key) into a path. The
  // name may embed subdirectories, created on materialization (see
  // prepareItem). An absolute name is honored verbatim; a relative name is
  // placed under the edge's outputDir.
  std::string makeOutputPath(llvm::StringRef key) const {
    std::string fileName = name;
    auto pos = fileName.find("{0}");
    if (pos != std::string::npos)
      fileName.replace(pos, 3, key.str());
    if (llvm::sys::path::is_absolute(fileName))
      return fileName;
    llvm::SmallString<256> path(outputDir);
    llvm::sys::path::append(path, fileName);
    return std::string(path.str());
  }
};

// Forward declarations of concrete edge types (used by chaining helpers).
struct FileInputEdge;
template <typename In, typename Out, typename MapFn>
struct MapEdge;
template <typename T, typename U, typename JoinFn>
struct JoinEdge;
template <typename In, typename Out, typename SplitFn>
struct SplitEdge;
template <typename T, typename FilterFn>
struct FilterEdge;
template <typename P, typename S, typename KeyFn>
struct RekeyEdge;
template <typename U, typename BundleMapFn, typename... Ts>
struct BundleForEachEdge;
template <typename U, typename BundleJoinFn, typename... Ts>
struct BundleJoinEdge;

// The Graph owns edges; nodes are owned by the edges that produce them.
struct Graph {
  std::vector<std::unique_ptr<EdgeBase>> edges;

  template <typename EdgeT, typename... Args>
  EdgeT &addEdge(Args &&...args) {
    auto e = std::make_unique<EdgeT>(std::forward<Args>(args)...);
    EdgeT &ref = *e;
    edges.push_back(std::move(e));
    return ref;
  }

  // Seed the graph with an existing on-disk file; item.filePath == path.
  FileInputEdge &fileInput(std::string path, std::string name);

  // Every edge registered under exactly `name` (its output-path template). A
  // handful of names are shared by two edges on purpose — the toolchain /
  // strategy variants that emit the same artifact (chess vs peano
  // "elfs_{0}.elf", per-core vs unified "objects_{0}.o") — so this may return
  // more than one; callers disambiguate via reachability.
  llvm::SmallVector<EdgeBase *, 2> edgesByName(llvm::StringRef name) const {
    llvm::SmallVector<EdgeBase *, 2> matches;
    for (const auto &e : edges)
      if (e->name == name)
        matches.push_back(e.get());
    return matches;
  }
};

// Base for edges that produce a typed output Node — provides fluent chaining.
template <typename Out>
struct EdgeWithTypedOutput : EdgeBase {
  Node<Out> out;

  EdgeWithTypedOutput(Graph &g, std::string name)
      : EdgeBase(g, std::move(name)), out(g) {
    out.producer = this;
  }

  NodeBase *outputNode() final { return &out; }

  // Fluent marker: declare this edge free of shared mutable state so the
  // scheduler may run its per-item actions in parallel. Returns the typed edge
  // for chaining onto a map/split/filter/join call.
  EdgeWithTypedOutput &threadSafe() {
    isThreadSafe = true;
    return *this;
  }

  void writeOutput() final {
    for (auto &item : out.items)
      (void)item.asFile();
  }

  // Rehydrate output items from artifacts on disk (see EdgeBase::restoreNode).
  llvm::json::Value captureNode(llvm::StringRef dir) final {
    return NodeSerializer<Out>::write(out.items, dir);
  }

  mlir::LogicalResult restoreNode(const llvm::json::Value &desc,
                                  llvm::StringRef dir,
                                  const DeserializeContext &dc) final {
    auto items = NodeDeserializer<Out>::read(desc, dir, dc);
    if (mlir::failed(items))
      return mlir::failure();
    out.items = std::move(*items);
    return mlir::success();
  }

  // Two callable shapes are accepted:
  //  - Pure: `U fn(const Out&)`. Infallible — the action just sets out.value.
  //  - Full: `LogicalResult fn(const Item<Out>&, Item<U>&)`. Required when the
  //    action is fallible, mutates out.filePath, or needs Item accessors.
  template <typename U, typename Fn>
  auto &map(std::string name, Fn fn) {
    if constexpr (std::is_invocable_r_v<U, Fn, const Out &>) {
      auto wrapped = [fn = std::move(fn)](const Item<Out> &in,
                                          Item<U> &out) -> mlir::LogicalResult {
        out.value = fn(in.get());
        return mlir::success();
      };
      return graph.template addEdge<MapEdge<Out, U, decltype(wrapped)>>(
          out, std::move(name), std::move(wrapped));
    } else {
      return graph.template addEdge<MapEdge<Out, U, Fn>>(out, std::move(name),
                                                         std::move(fn));
    }
  }

  template <typename U, typename JoinFn>
  JoinEdge<Out, U, JoinFn> &join(std::string name, JoinFn fn) {
    return graph.template addEdge<JoinEdge<Out, U, JoinFn>>(
        out, std::move(name), std::move(fn));
  }

  template <typename U, typename SplitFn>
  SplitEdge<Out, U, SplitFn> &split(std::string name, SplitFn fn) {
    return graph.template addEdge<SplitEdge<Out, U, SplitFn>>(
        out, std::move(name), std::move(fn));
  }

  template <typename FilterFn>
  FilterEdge<Out, FilterFn> &filter(std::string name, FilterFn fn) {
    return graph.template addEdge<FilterEdge<Out, FilterFn>>(
        out, std::move(name), std::move(fn));
  }

  // Re-key a `secondary` Node onto this node's keys: one output item per item
  // here, aliasing the secondary item whose key is `keyFn(payload)`. Makes a
  // broadcast/keyed-join (e.g. each core picking its device's shared object) an
  // explicit edge.
  template <typename S, typename KeyFn>
  RekeyEdge<Out, S, KeyFn> &rekeyFrom(std::string name, Node<S> &secondary,
                                      KeyFn fn) {
    return graph.template addEdge<RekeyEdge<Out, S, KeyFn>>(
        out, secondary, std::move(name), std::move(fn));
  }

protected:
  Item<Out> prepareItem(llvm::StringRef key) {
    Item<Out> item;
    item.key = key.str();
    item.filePath = this->makeOutputPath(key);
    // The name may embed subdirectories; ensure the parent exists before
    // anything materializes the item to disk.
    llvm::StringRef parent = llvm::sys::path::parent_path(item.filePath);
    if (!parent.empty())
      llvm::sys::fs::create_directories(parent);
    return item;
  }
};

// Single-input edge.
template <typename In, typename Out>
struct Edge : EdgeWithTypedOutput<Out> {
  Node<In> &in;

  Edge(Node<In> &src, std::string name)
      : EdgeWithTypedOutput<Out>(src.graph, std::move(name)), in(src) {}

  std::vector<NodeBase *> inputNodes() override { return {&in}; }
};

// FileInputEdge — seeds the graph with an existing on-disk file.
struct FileInputEdge : EdgeWithTypedOutput<File> {
  std::string path;

  FileInputEdge(Graph &g, std::string path, std::string name)
      : EdgeWithTypedOutput<File>(g, std::move(name)), path(std::move(path)) {
    producesFiles = false;
  }

  mlir::LogicalResult execute() final {
    Item<File> item;
    item.key = this->name;
    item.filePath = path;
    item.value = File{};
    this->out.items.push_back(std::move(item));
    return mlir::success();
  }
};

inline FileInputEdge &Graph::fileInput(std::string path, std::string name) {
  return addEdge<FileInputEdge>(*this, std::move(path), std::move(name));
}

// MapEdge — apply `fn` to each input item, one output item per input.
template <typename In, typename Out, typename MapFn>
struct MapEdge : Edge<In, Out> {
  MapFn fn;

  MapEdge(Node<In> &src, std::string name, MapFn f)
      : Edge<In, Out>(src, std::move(name)), fn(std::move(f)) {}

  bool isFanOut() const override { return true; }

  // Prepare all output slots (renders paths, creates parent dirs) without
  // running any per-item work; the scheduler runs the items as separate tasks.
  mlir::LogicalResult open() override {
    const size_t n = this->in.items.size();
    this->out.items.clear();
    this->out.items.reserve(n);
    for (const auto &inItem : this->in.items)
      this->out.items.push_back(this->prepareItem(inItem.key));
    return mlir::success();
  }

  size_t numItems() const override { return this->out.items.size(); }

  llvm::StringRef itemKey(size_t i) const override {
    return this->out.items[i].key;
  }

  mlir::LogicalResult runItem(size_t i) override {
    if (mlir::failed(fn(this->in.items[i], this->out.items[i]))) {
      this->recordFailedKey(this->in.items[i].key);
      return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult execute() override {
    if (mlir::failed(open()))
      return mlir::failure();
    for (size_t i = 0, n = this->out.items.size(); i < n; ++i)
      if (mlir::failed(runItem(i)))
        return mlir::failure();
    return mlir::success();
  }
};

// JoinEdge — fold a Node<T> into a singleton U.
template <typename T, typename U, typename JoinFn>
struct JoinEdge : Edge<T, U> {
  JoinFn fn;

  JoinEdge(Node<T> &src, std::string name, JoinFn f)
      : Edge<T, U>(src, std::move(name)), fn(std::move(f)) {}

  mlir::LogicalResult execute() override {
    Item<U> outItem = this->prepareItem(this->name);
    if (mlir::failed(fn(this->in, outItem)))
      return mlir::failure();
    this->out.items.clear();
    this->out.items.push_back(std::move(outItem));
    return mlir::success();
  }
};

// SplitEdge — split a singleton input into multiple output items. `fn` returns
// bare (key, value) pairs; the edge wraps each into an Item<Out> with a
// rendered path. If >1 item, the name template must contain `{0}`.
template <typename In, typename Out, typename SplitFn>
struct SplitEdge : Edge<In, Out> {
  SplitFn fn;

  SplitEdge(Node<In> &src, std::string name, SplitFn f)
      : Edge<In, Out>(src, std::move(name)), fn(std::move(f)) {}

  mlir::LogicalResult execute() override {
    assert(this->in.items.size() == 1 &&
           "SplitEdge requires a singleton input node");
    auto produced = fn(this->in.get());
    if (mlir::failed(produced))
      return mlir::failure();
    this->out.items.clear();
    this->out.items.reserve(produced->size());
    for (auto &[key, value] : *produced) {
      Item<Out> outItem = this->prepareItem(key);
      outItem.value = std::move(value);
      this->out.items.push_back(std::move(outItem));
    }
    return mlir::success();
  }
};

// FilterEdge — produce a view of a Node<T> containing only items whose payload
// satisfies `fn`. Output items alias the source items (Item::aliasSource)
// rather than owning a payload, so no cloning happens.
template <typename T, typename FilterFn>
struct FilterEdge : Edge<T, T> {
  FilterFn fn;

  FilterEdge(Node<T> &src, std::string name, FilterFn f)
      : Edge<T, T>(src, std::move(name)), fn(std::move(f)) {
    this->producesFiles = false;
  }

  mlir::LogicalResult execute() override {
    this->out.items.clear();
    for (const auto &src : this->in.items)
      if (fn(src.get())) {
        Item<T> view;
        view.key = src.key;
        view.aliasSource = &src;
        this->out.items.push_back(std::move(view));
      }
    return mlir::success();
  }
};

// RekeyEdge — re-key a `secondary` Node onto a `primary` Node's keys. Produces
// one output item per *primary* item, aliasing (no copy) the *secondary* item
// whose key equals `keyFn(primaryPayload)`.
template <typename P, typename S, typename KeyFn>
struct RekeyEdge : Edge<P, S> {
  Node<S> &secondary;
  KeyFn keyFn;

  RekeyEdge(Node<P> &primary, Node<S> &secondary, std::string name, KeyFn fn)
      : Edge<P, S>(primary, std::move(name)), secondary(secondary),
        keyFn(std::move(fn)) {
    this->producesFiles = false; // aliases secondary items; no new payloads
  }

  std::vector<NodeBase *> inputNodes() override {
    return {&this->in, &secondary};
  }

  mlir::LogicalResult execute() override {
    this->out.items.clear();
    this->out.items.reserve(this->in.items.size());
    for (const auto &primaryItem : this->in.items) {
      std::string sk = keyFn(primaryItem.get());
      const Item<S> *src = nullptr;
      for (const auto &cand : secondary.items)
        if (cand.key == sk) {
          src = &cand;
          break;
        }
      if (!src) {
        llvm::errs() << "aiecc: rekey edge '" << this->name
                     << "': no source item for key '" << sk << "'\n";
        return mlir::failure();
      }
      Item<S> view;
      view.key = primaryItem.key;
      view.aliasSource = src;
      this->out.items.push_back(std::move(view));
    }
    return mlir::success();
  }
};

// BundleForEachEdge — zip N source Nodes by the first source's keys, invoke
// `fn` once per key with per-source Item refs.
template <typename U, typename BundleMapFn, typename... Ts>
struct BundleForEachEdge : EdgeWithTypedOutput<U> {
  std::tuple<Node<Ts> &...> in;
  BundleMapFn fn;

  BundleForEachEdge(Graph &g, std::string name, BundleMapFn f,
                    std::tuple<Node<Ts> &...> srcs)
      : EdgeWithTypedOutput<U>(g, std::move(name)), in(srcs), fn(std::move(f)) {
  }

  std::vector<NodeBase *> inputNodes() override {
    std::vector<NodeBase *> r;
    std::apply([&](Node<Ts> &...nodes) { (r.push_back(&nodes), ...); }, in);
    return r;
  }

  bool isFanOut() const override { return true; }

  // Prepare one output slot per key of the first source node; the scheduler
  // runs the per-key actions as separate tasks.
  mlir::LogicalResult open() override {
    auto &firstNode = std::get<0>(in);
    const size_t n = firstNode.items.size();
    this->out.items.clear();
    this->out.items.reserve(n);
    for (const auto &firstItem : firstNode.items)
      this->out.items.push_back(this->prepareItem(firstItem.key));
    return mlir::success();
  }

  size_t numItems() const override { return this->out.items.size(); }

  llvm::StringRef itemKey(size_t i) const override {
    return this->out.items[i].key;
  }

  // Zip the source nodes by the first source's key `i` and run `fn` on the
  // matched items.
  mlir::LogicalResult runItem(size_t i) override {
    const std::string &key = this->out.items[i].key;
    Item<U> &outItem = this->out.items[i];
    mlir::LogicalResult r = std::apply(
        [&](Node<Ts> &...nodes) -> mlir::LogicalResult {
          auto findItem = [&key](auto &node)
              -> const std::decay_t<decltype(node.items.front())> * {
            for (const auto &item : node.items)
              if (item.key == key)
                return &item;
            return nullptr;
          };
          bool anyMissing = false;
          auto require = [&anyMissing](auto *p) {
            if (!p)
              anyMissing = true;
            return p;
          };
          auto found = std::make_tuple(require(findItem(nodes))...);
          if (anyMissing) {
            llvm::errs() << "aiecc: bundle edge '" << this->name
                         << "': a bundled source node has no item for "
                            "key '"
                         << key
                         << "'; the bundled nodes have incompatible "
                            "keys\n";
            return mlir::failure();
          }
          return std::apply(
              [&](auto *...items) { return fn(*items..., outItem); }, found);
        },
        in);
    if (mlir::failed(r))
      this->recordFailedKey(key);
    return r;
  }

  mlir::LogicalResult execute() override {
    if (mlir::failed(open()))
      return mlir::failure();
    for (size_t i = 0, n = this->out.items.size(); i < n; ++i)
      if (mlir::failed(runItem(i)))
        return mlir::failure();
    return mlir::success();
  }
};

// BundleJoinEdge — fold N source Nodes into a singleton output U.
template <typename U, typename BundleJoinFn, typename... Ts>
struct BundleJoinEdge : EdgeWithTypedOutput<U> {
  std::tuple<Node<Ts> &...> in;
  BundleJoinFn fn;

  BundleJoinEdge(Graph &g, std::string name, BundleJoinFn f,
                 std::tuple<Node<Ts> &...> srcs)
      : EdgeWithTypedOutput<U>(g, std::move(name)), in(srcs), fn(std::move(f)) {
  }

  std::vector<NodeBase *> inputNodes() override {
    std::vector<NodeBase *> r;
    std::apply([&](Node<Ts> &...nodes) { (r.push_back(&nodes), ...); }, in);
    return r;
  }

  mlir::LogicalResult execute() override {
    Item<U> outItem = this->prepareItem(this->name);
    if (mlir::failed(std::apply(
            [&](const Node<Ts> &...nodes) { return fn(nodes..., outItem); },
            in)))
      return mlir::failure();
    this->out.items.clear();
    this->out.items.push_back(std::move(outItem));
    return mlir::success();
  }
};

// BundleHelper — fluent entry to BundleForEach / BundleJoin. Bundles zip
// multiple nodes by key and run an action on the matched items together.
template <typename... Ts>
struct BundleHelper {
  Graph &graph;
  std::tuple<Node<Ts> &...> srcs;

  BundleHelper(Graph &g, Node<Ts> &...s) : graph(g), srcs(s...) {}

  template <typename U, typename BundleMapFn>
  BundleForEachEdge<U, BundleMapFn, Ts...> &map(std::string name,
                                                BundleMapFn fn) {
    return graph.template addEdge<BundleForEachEdge<U, BundleMapFn, Ts...>>(
        graph, std::move(name), std::move(fn), srcs);
  }

  template <typename U, typename BundleJoinFn>
  BundleJoinEdge<U, BundleJoinFn, Ts...> &join(std::string name,
                                               BundleJoinFn fn) {
    return graph.template addEdge<BundleJoinEdge<U, BundleJoinFn, Ts...>>(
        graph, std::move(name), std::move(fn), srcs);
  }
};

template <typename T0, typename... Ts>
BundleHelper<T0, Ts...> bundle(Node<T0> &first, Node<Ts> &...rest) {
  return BundleHelper<T0, Ts...>(first.graph, first, rest...);
}

} // namespace xilinx::aiecc

#endif // AIECC_GRAPH_H
