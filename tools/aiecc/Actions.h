//===- Actions.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reusable action classes: an "action" is the transformation callable an edge
// invokes to turn its input Item(s) into an output Item -- i.e. what you pass
// as the `fn` argument to `.map()` / `.join()` when building the graph (a
// MapEdge applies it per input item, a BundleForEachEdge applies it per zipped
// bundle). Plain lambdas are also valid actions and live at their use sites;
// the named classes here (ShellCommand, PassPipeline, ...) are the ones reused
// across many edges.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_ACTIONS_H
#define AIECC_ACTIONS_H

#include "Graph.h"
#include "Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace xilinx::aiecc {

// Lift an IR-to-instruction-stream translator into a graph map action emitting
// raw bytes. `fill` populates a `std::vector<uint32_t>` from the input item;
// this wrapper handles failure propagation and the word->byte conversion.
template <typename In, typename Fill>
auto emitBinary(Fill fill) {
  return [fill = std::move(fill)](
             const Item<In> &item,
             Item<std::vector<char>> &out) -> mlir::LogicalResult {
    std::vector<uint32_t> words;
    if (mlir::failed(fill(item, words)))
      return mlir::failure();
    out.value = wordsToBytes(words);
    return mlir::success();
  };
}

// Materialize an MLIR module from an item whose payload carries IR. Overloaded
// per module-bearing payload type: ModRef and OpInModule clone their in-memory
// module; File re-parses its .mlir text into `ctx`.
inline mlir::OwningOpRef<mlir::ModuleOp>
asModule(const Item<mlir::OwningOpRef<mlir::ModuleOp>> &in,
         mlir::MLIRContext * /*ctx*/) {
  return mlir::OwningOpRef<mlir::ModuleOp>(in.get().get().clone());
}

template <typename KeyOp>
inline mlir::OwningOpRef<mlir::ModuleOp>
asModule(const Item<OpInModule<KeyOp>> &in, mlir::MLIRContext * /*ctx*/) {
  return mlir::OwningOpRef<mlir::ModuleOp>(in.get().module.get().clone());
}

inline mlir::OwningOpRef<mlir::ModuleOp> asModule(const Item<File> &in,
                                                  mlir::MLIRContext *ctx) {
  return parseModuleFromFile(in.asFile(), ctx);
}

// PassPipeline — execute an MLIR pass-pipeline on a clone of the input.
// Accepts any payload the asModule overloads accept (ModRef / OpInModule /
// File); any other payload type fails to compile.
//
// Two construction modes:
//  - Pre-built form: pass an already-configured PassManager. Use when the
//    pipeline is fixed and independent of the input IR.
//  - Builder form: pass a callback `(MLIRContext*, ModuleOp) -> unique_ptr<PM>`
//    invoked per input. Use when pipeline configuration depends on inspecting
//    the input module (e.g. target-arch detection). Return nullptr to fail.
struct PassPipeline {
  using Builder = std::function<std::unique_ptr<mlir::PassManager>(
      mlir::MLIRContext *, mlir::ModuleOp)>;
  mlir::MLIRContext *ctx;
  Builder build;
  std::unique_ptr<mlir::PassManager> prebuilt;

  PassPipeline(mlir::MLIRContext *ctx, Builder b)
      : ctx(ctx), build(std::move(b)) {}

  explicit PassPipeline(std::unique_ptr<mlir::PassManager> pm)
      : ctx(pm->getContext()), prebuilt(std::move(pm)) {}

  template <typename T>
  mlir::LogicalResult
  operator()(const Item<T> &in,
             Item<mlir::OwningOpRef<mlir::ModuleOp>> &out) const {
    auto mod = asModule(in, ctx);
    if (!mod) {
      llvm::errs() << "aiecc: PassPipeline could not obtain input module\n";
      return mlir::failure();
    }
    mlir::PassManager *pm = prebuilt.get();
    std::unique_ptr<mlir::PassManager> built;
    if (!pm) {
      built = build(ctx, *mod);
      if (!built) {
        llvm::errs() << "aiecc: PassPipeline builder returned null\n";
        return mlir::failure();
      }
      pm = built.get();
    }
    if (mlir::failed(pm->run(*mod)))
      return mlir::failure();
    out.value = std::move(mod);
    return mlir::success();
  }
};

// SplitIRAction — walks a ModuleOp for KeyOp instances; clones the module
// once per match. Use `.filter` downstream to skip matches.
template <typename KeyOp>
struct SplitIRAction {
  using KeyFn = std::function<std::string(KeyOp)>;
  KeyFn keyFn;

  SplitIRAction(KeyFn fn) : keyFn(std::move(fn)) {}

  mlir::FailureOr<std::vector<std::pair<std::string, OpInModule<KeyOp>>>>
  operator()(const mlir::OwningOpRef<mlir::ModuleOp> &in) const {
    auto srcModule = in.get();
    std::vector<std::pair<std::string, size_t>> matches;
    size_t idx = 0;
    srcModule.walk([&](KeyOp op) {
      matches.emplace_back(keyFn(op), idx);
      ++idx;
    });

    std::vector<std::pair<std::string, OpInModule<KeyOp>>> out;
    out.reserve(matches.size());
    for (auto &[key, target] : matches) {
      mlir::OwningOpRef<mlir::ModuleOp> clone = srcModule.clone();
      KeyOp clonedOp;
      size_t cur = 0;
      clone->walk([&](KeyOp op) -> mlir::WalkResult {
        if (cur++ != target)
          return mlir::WalkResult::advance();
        clonedOp = op;
        return mlir::WalkResult::interrupt();
      });
      out.emplace_back(std::move(key),
                       OpInModule<KeyOp>{std::move(clone), clonedOp});
    }
    return out;
  }
};

// ShellCommand — declarative external-tool invocation, built fluently:
//   ShellCommand{"llc"}.flag("--march", arch).arg("--filetype=obj")
//                      .input().output("-o")
struct ShellCommand {
  struct Part {
    enum Kind { Literal, Slot, SlotList, Output, OutputDir } kind;
    enum Mode { Path, Value } mode = Path;
    std::string text;
    std::string suffix;

    static Part literal(std::string s) {
      return {Literal, Path, std::move(s), {}};
    }
    static Part mkSlot(Mode m, std::string prefix, std::string suffix) {
      return {Slot, m, std::move(prefix), std::move(suffix)};
    }
    static Part mkSlotList(std::string prefix, std::string suffix) {
      return {SlotList, Value, std::move(prefix), std::move(suffix)};
    }
    static Part mkOutput(std::string prefix, bool concat = false) {
      return {Output, concat ? Value : Path, std::move(prefix), {}};
    }
    static Part mkOutputDir(std::string prefix) {
      return {OutputDir, Path, std::move(prefix), {}};
    }
  };

  std::string tool;
  std::vector<Part> parts;

  inline static std::vector<std::string> searchPaths;
  inline static std::map<std::string, std::string> toolPathCache;

  // Guards toolPathCache against concurrent resolveTool() calls
  inline static std::mutex toolCacheMutex;

  // Serializes the --verbose command echo so concurrent invocations don't
  // interleave their lines on stdout.
  inline static std::mutex echoMutex;

  inline static bool verbose = false;

  // When true, an empty placeholder output is created so the engine's path
  // bookkeeping resolves; downstream edges that parse a tool's output won't
  // have real data.
  inline static bool dryRun = false;

  // Prepend a directory to the search list; most-recently-added wins.
  static void addSearchPath(std::string prefix) {
    searchPaths.push_back(std::move(prefix));
    toolPathCache.clear();
  }

  // Register `<dir>/bin` as a search path. Tries (in order): overrideDir,
  // $<NAME>_INSTALL_DIR, <exe-prefix>/<name>, <exe-grandparent>/<name>.
  static void addInstallPrefix(llvm::StringRef name,
                               llvm::StringRef overrideDir = "") {
    auto tryAdd = [](llvm::StringRef dir) -> bool {
      if (dir.empty() || !llvm::sys::fs::is_directory(dir))
        return false;
      llvm::SmallString<256> binDir(dir);
      llvm::sys::path::append(binDir, "bin");
      addSearchPath(std::string(binDir));
      return true;
    };
    if (tryAdd(overrideDir))
      return;
    std::string envName = name.upper() + "_INSTALL_DIR";
    if (const char *env = std::getenv(envName.c_str()))
      if (tryAdd(env))
        return;
    std::string mainExe = llvm::sys::fs::getMainExecutable(
        nullptr, reinterpret_cast<void *>(&addInstallPrefix));
    llvm::StringRef prefix =
        llvm::sys::path::parent_path(llvm::sys::path::parent_path(mainExe));
    for (auto base : {prefix, llvm::sys::path::parent_path(prefix)}) {
      llvm::SmallString<256> p(base);
      llvm::sys::path::append(p, name);
      if (tryAdd(p))
        return;
    }
  }

  // Absolute paths pass through; basenames look up most-recent prefix
  // first, then PATH. Results cached for process lifetime.
  static std::string resolveTool(llvm::StringRef name) {
    if (llvm::sys::path::is_absolute(name))
      return name.str();
    std::lock_guard<std::mutex> lock(toolCacheMutex);
    auto it = toolPathCache.find(name.str());
    if (it != toolPathCache.end())
      return it->second;
    std::string result;
    for (auto p = searchPaths.rbegin(); p != searchPaths.rend(); ++p) {
      llvm::SmallString<256> candidate(*p);
      llvm::sys::path::append(candidate, name);
      if (llvm::sys::fs::can_execute(candidate)) {
        result = std::string(candidate);
        break;
      }
    }
    if (result.empty())
      if (auto r = llvm::sys::findProgramByName(name))
        result = *r;
    return toolPathCache.emplace(name.str(), std::move(result)).first->second;
  }

  explicit ShellCommand(std::string t) : tool(std::move(t)) {}

  ShellCommand &arg(std::string a) {
    parts.push_back(Part::literal(std::move(a)));
    return *this;
  }

  // Insert next source's on-disk path; `prefix`/`suffix` bracket into one argv.
  ShellCommand &input(std::string prefix = "", std::string suffix = "") {
    parts.push_back(
        Part::mkSlot(Part::Path, std::move(prefix), std::move(suffix)));
    return *this;
  }

  // Insert next source's string value (std::string-payload sources only).
  ShellCommand &value(std::string prefix = "", std::string suffix = "") {
    parts.push_back(
        Part::mkSlot(Part::Value, std::move(prefix), std::move(suffix)));
    return *this;
  }

  // Expand next source into zero or more argv entries (one per list element,
  // each bracketed by prefix/suffix). The source must carry a list payload;
  // entries are used verbatim and never materialized to disk.
  ShellCommand &inputs(std::string prefix = "", std::string suffix = "") {
    parts.push_back(Part::mkSlotList(std::move(prefix), std::move(suffix)));
    return *this;
  }

  ShellCommand &output(std::string flagPrefix = "") {
    parts.push_back(Part::mkOutput(std::move(flagPrefix)));
    return *this;
  }

  // Like output(), but concatenates `prefix` and the output path into a single
  // argv entry (e.g. xclbinutil's `--dump-section SECTION:JSON:<path>`).
  ShellCommand &outputConcat(std::string prefix) {
    parts.push_back(Part::mkOutput(std::move(prefix), /*concat=*/true));
    return *this;
  }

  // Insert the output *directory* path. Only meaningful when the edge's output
  // is an Item<Directory>: that is the folder ShellCommand creates and runs the
  // tool inside, so a tool that takes its own scratch/work dir (e.g. chess's
  // `+w <dir>`) can point at it.
  ShellCommand &outputDir(std::string prefix = "") {
    parts.push_back(Part::mkOutputDir(std::move(prefix)));
    return *this;
  }

  // Uniform entry: takes any number of input Items (in bundle declaration
  // order) followed by the Item<File> output. Each input/value part consumes
  // the next source in order.
  template <typename... Args>
  mlir::LogicalResult operator()(Args &&...args) const {
    static_assert(sizeof...(Args) >= 1,
                  "ShellCommand requires at least the output Item (File or "
                  "Directory) as its last argument");
    return invokeImpl(std::forward_as_tuple(std::forward<Args>(args)...),
                      std::make_index_sequence<sizeof...(Args) - 1>{});
  }

private:
  template <typename Tup, std::size_t... I>
  mlir::LogicalResult invokeImpl(Tup &&tup, std::index_sequence<I...>) const {
    std::array<const ItemBase *, sizeof...(I)> items = {&std::get<I>(tup)...};
    auto &out = std::get<sizeof...(I)>(tup);
    return run(items, out);
  }

  mlir::LogicalResult run(llvm::ArrayRef<const ItemBase *> sources,
                          Item<Directory> &out) const {
    std::string resolved = resolveTool(tool);
    if (resolved.empty()) {
      if (!dryRun) {
        llvm::errs() << "aiecc: ShellCommand: tool '" << tool
                     << "' not found in search paths or PATH\n";
        return mlir::failure();
      }
      resolved = tool;
    }
    // The output is a directory holding the `-o` file and the sidecar files the
    // tool writes beside it (e.g. chess's `<elf>.map`). Run the tool in place
    // there -- nothing is copied or moved. Each item's key gives it its own
    // directory (named after the requested output's stem), so parallel per-core
    // invocations never share one. A tool scratch dir (`+w`) can point at it
    // via .outputDir().
    llvm::SmallString<256> dir(llvm::sys::path::parent_path(out.filePath));
    llvm::sys::path::append(
        dir, llvm::sys::path::stem(llvm::sys::path::filename(out.filePath)));
    if (llvm::sys::fs::create_directories(dir)) {
      llvm::errs() << "aiecc: ShellCommand '" << tool
                   << "': cannot create output dir '" << dir << "'\n";
      return mlir::failure();
    }
    std::string outputFile =
        (llvm::Twine(dir) + "/" + llvm::sys::path::filename(out.filePath))
            .str();
    if (mlir::failed(
            runResolved(std::move(resolved), sources, outputFile, dir)))
      return mlir::failure();
    out.filePath = outputFile;
    out.value = Directory{std::string(dir)};
    return mlir::success();
  }

  mlir::LogicalResult run(llvm::ArrayRef<const ItemBase *> sources,
                          Item<File> &out) const {
    std::string resolved = resolveTool(tool);
    if (resolved.empty()) {
      // A dry run never executes the tool, so a missing tool must not fail the
      // pipeline: fall back to the bare tool name for the echoed command line.
      // This keeps metadata-only dry-run flows (e.g. inspecting the generated
      // kernels JSON) working on machines without XRT/aietools on PATH.
      if (!dryRun) {
        llvm::errs() << "aiecc: ShellCommand: tool '" << tool
                     << "' not found in search paths or PATH\n";
        return mlir::failure();
      }
      resolved = tool;
    }
    if (mlir::failed(runResolved(std::move(resolved), sources, out.filePath,
                                 /*outputDir=*/"")))
      return mlir::failure();
    out.value = File{};
    return mlir::success();
  }

  mlir::LogicalResult runResolved(std::string resolved,
                                  llvm::ArrayRef<const ItemBase *> sources,
                                  llvm::StringRef outputFile,
                                  llvm::StringRef outputDirPath) const {
    std::vector<std::string> cmd{std::move(resolved)};
    cmd.reserve(parts.size() + 2);
    size_t cursor = 0;
    for (const auto &p : parts) {
      switch (p.kind) {
      case Part::Literal:
        cmd.push_back(p.text);
        break;
      case Part::Slot:
        if (cursor >= sources.size()) {
          llvm::errs() << "aiecc: ShellCommand '" << tool
                       << "': not enough sources for input/value parts\n";
          return mlir::failure();
        }
        cmd.push_back(p.text +
                      (p.mode == Part::Path ? sources[cursor]->asFile()
                                            : sources[cursor]->asString()) +
                      p.suffix);
        ++cursor;
        break;
      case Part::SlotList:
        if (cursor >= sources.size()) {
          llvm::errs() << "aiecc: ShellCommand '" << tool
                       << "': not enough sources for inputs parts\n";
          return mlir::failure();
        }
        for (const auto &entry : sources[cursor]->asArgList())
          cmd.push_back(p.text + entry + p.suffix);
        ++cursor;
        break;
      case Part::Output:
        if (p.mode == Part::Value) {
          cmd.push_back(p.text + outputFile.str());
        } else {
          if (!p.text.empty())
            cmd.push_back(p.text);
          cmd.push_back(outputFile.str());
        }
        break;
      case Part::OutputDir:
        cmd.push_back(p.text + outputDirPath.str());
        break;
      }
    }
    if (verbose) {
      // Echo the command on stdout so callers that capture stdout can see it.
      std::lock_guard<std::mutex> lock(echoMutex);
      llvm::outs() << "aiecc: exec:";
      for (const auto &a : cmd)
        llvm::outs() << ' ' << a;
      llvm::outs() << '\n';
      llvm::outs().flush();
    }
    if (dryRun)
      return mlir::success();
    llvm::SmallVector<llvm::StringRef> argv(cmd.begin(), cmd.end());
    std::string errMsg;
    // Capture the tool's stdout+stderr into a temp file so routine chatter
    // stays hidden; replayed to stderr only on failure. Under --verbose the
    // tool inherits stdout/stderr and prints normally.
    // Redirects are [stdin, stdout, stderr]; std::nullopt inherits, and
    // pointing stdout and stderr at the same path merges them.
    llvm::SmallString<128> logPath;
    int logFd = -1;
    std::optional<llvm::StringRef> capture;
    if (!verbose && !llvm::sys::fs::createTemporaryFile("aiecc-tool", "log",
                                                        logFd, logPath)) {
      // ExecuteAndWait opens the path itself, so close our handle.
      llvm::sys::Process::SafelyCloseFileDescriptor(logFd);
      capture = llvm::StringRef(logPath);
    }
    std::array<std::optional<llvm::StringRef>, 3> redirects = {
        std::nullopt, capture, capture};
    llvm::ArrayRef<std::optional<llvm::StringRef>> redirectRef;
    if (capture)
      redirectRef = redirects;
    int rc = llvm::sys::ExecuteAndWait(cmd[0], argv, std::nullopt, redirectRef,
                                       0, 0, &errMsg);
    if (capture) {
      if (rc != 0)
        if (auto buf = llvm::MemoryBuffer::getFile(logPath))
          llvm::errs() << (*buf)->getBuffer();
      llvm::sys::fs::remove(logPath);
    }
    if (rc != 0) {
      llvm::errs() << "aiecc: '" << cmd[0] << "' failed: " << errMsg << "\n";
      return mlir::failure();
    }
    return mlir::success();
  }
};

} // namespace xilinx::aiecc

#endif // AIECC_ACTIONS_H
