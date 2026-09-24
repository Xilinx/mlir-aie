//===- DeviceCache.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `--device-cache=<dir>` reuses an earlier build's buffer-placed and patched
// device IR plus each linked core ELF, avoiding recompilation.
//
// Entries are keyed by routed device IR, other top-level ops, linked-file
// contents, command line, and aiecc/Peano tool identities. Locations are
// excluded, so reused devices retain the storing build's locations.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_DEVICECACHE_H
#define AIECC_DEVICECACHE_H

#include "IRTransforms.h"
#include "Items.h"
#include "Utils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace xilinx::aiecc {

class DeviceCache {
public:
  DeviceCache(std::string dir, std::string configKey, bool verbose)
      : dir(absolutePath(dir)), configKey(std::move(configKey)),
        verbose(verbose) {}

  // Whether this build reads and writes the cache. Decided once the requested
  // outputs are known: a build that emits anything from inside a core's
  // compilation needs every core compiled.
  bool active = false;

  // What the command line and toolchain contribute to every key. Options that
  // only name, select or report outputs are left out, as are positionals: the
  // input's content is keyed per device.
  static std::string configurationKey(llvm::ArrayRef<std::string> argv,
                                      llvm::ArrayRef<std::string> tools) {
    static const llvm::StringSet<> ignored = {
        "o",
        "output-dir",
        "tmpdir",
        "verbose",
        "v",
        "progress",
        "no-progress",
        "profile",
        "verify-each",
        "nthreads",
        "get",
        "g",
        "cut",
        "checkpoint",
        "resume",
        "enable-repeater-scripts",
        "disable-repeater-scripts",
        "repeater-output-dir",
        "emit-dot",
        "dump-intermediates",
        "device-cache",
        "device-name",
        "npu-cpp-name",
        "npu-insts-name",
        "elf-name",
        "pdi-name",
        "txn-name",
        "ctrlpkt-name",
        "ctrlpkt-dma-seq-name",
        "ctrlpkt-elf-name",
        "xclbin-name",
        "full-elf-name",
        "xclbin-kernel-name",
        "xclbin-instance-name",
        "xclbin-kernel-id",
        "xclbin-input",
        "xclbinutil-path",
    };
    auto &registered = llvm::cl::getRegisteredOptions();
    llvm::SHA256 hash;
    hash.update("aiecc-device-cache-v1");
    std::string exe = llvm::sys::fs::getMainExecutable(
        nullptr, reinterpret_cast<void *>(&configurationKey));
    addToolIdentity(hash, exe);
    for (const std::string &tool : tools) {
      addToolIdentity(hash, tool);
    }
    for (size_t i = 1; i < argv.size(); ++i) {
      llvm::StringRef arg = argv[i];
      if (arg == "--") {
        break;
      }
      if (!arg.starts_with("-")) {
        continue;
      }
      llvm::StringRef name = arg.ltrim('-');
      bool hasValue = name.contains('=');
      name = name.take_until([](char c) { return c == '='; });
      llvm::cl::Option *opt = registered.lookup(name);
      bool takesNext = !hasValue && opt &&
                       opt->getValueExpectedFlag() == llvm::cl::ValueRequired &&
                       i + 1 < argv.size();
      bool isIgnored =
          ignored.contains(name) ||
          (name.consume_front("j") && llvm::all_of(name, llvm::isDigit));
      if (!isIgnored) {
        addField(hash, arg);
        if (takesNext) {
          addField(hash, argv[i + 1]);
        }
      }
      if (takesNext) {
        ++i;
      }
    }
    return llvm::toHex(hash.result(), /*LowerCase=*/true);
  }

  // Find each cacheable device's entry. `resolvePath` locates a file a core
  // links, the way the link itself would.
  void lookup(mlir::ModuleOp module,
              llvm::function_ref<bool(xilinx::AIE::DeviceOp)> matches,
              llvm::function_ref<std::string(llvm::StringRef)> resolvePath) {
    if (!active) {
      return;
    }
    mlir::OwningOpRef<mlir::ModuleOp> shared =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(module.getContext()));
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      if (!mlir::isa<xilinx::AIE::DeviceOp>(op)) {
        shared->getBody()->push_back(op.clone());
      }
    }
    std::string sharedText = printWithoutLocations(shared.get());
    for (auto device : module.getOps<xilinx::AIE::DeviceOp>()) {
      if (!matches(device) || !isCacheable(device)) {
        continue;
      }
      std::optional<std::string> key =
          deviceKey(device, sharedText, resolvePath);
      if (!key) {
        continue;
      }
      Entry &entry = devices[device.getSymName()];
      llvm::SmallString<256> entryDir(dir);
      llvm::sys::path::append(entryDir, *key);
      entry.dir = std::string(entryDir);
      if (llvm::sys::fs::exists(entry.dir)) {
        entry.hit = load(entry, device.getSymName(), module.getContext());
        entry.stale = !entry.hit;
      }
      if (verbose) {
        llvm::errs() << "aiecc: device cache " << (entry.hit ? "hit" : "miss")
                     << ": " << device.getSymName() << "\n";
      }
    }
  }

  bool isHit(xilinx::AIE::DeviceOp device) const {
    auto it = devices.find(device.getSymName());
    return it != devices.end() && it->second.hit;
  }

  // Put each hit device in as its entry placed it, then run buffer placement
  // (`placeRest`), which must leave hit devices alone. Each missed device's
  // placement is kept for `link` to store.
  mlir::LogicalResult
  place(mlir::ModuleOp module,
        llvm::function_ref<mlir::LogicalResult()> placeRest) {
    if (!active) {
      return placeRest();
    }
    replaceHits(module, &Entry::placedDevice);
    if (mlir::failed(placeRest())) {
      return mlir::failure();
    }
    for (auto device : module.getOps<xilinx::AIE::DeviceOp>()) {
      auto it = devices.find(device.getSymName());
      if (it != devices.end() && !it->second.hit) {
        it->second.placedText = printDevice(device);
      }
    }
    return mlir::success();
  }

  // Replace each hit device with its entry's linked form, and store each
  // missed device whose cores all compiled into `elfByKey`.
  void link(mlir::ModuleOp module,
            const llvm::StringMap<std::string> &elfByKey) {
    if (!active) {
      return;
    }
    replaceHits(module, &Entry::linkedDevice);
    for (auto device : module.getOps<xilinx::AIE::DeviceOp>()) {
      auto it = devices.find(device.getSymName());
      if (it != devices.end() && !it->second.hit &&
          !it->second.placedText.empty()) {
        store(it->second, device, elfByKey);
      }
    }
  }

private:
  struct Entry {
    std::string dir;
    bool hit = false;
    bool stale = false;
    mlir::OwningOpRef<mlir::ModuleOp> placed;
    mlir::OwningOpRef<mlir::ModuleOp> linked;
    std::string placedText;

    mlir::Operation *placedDevice() const {
      return &placed.get().getBody()->front();
    }
    mlir::Operation *linkedDevice() const {
      return &linked.get().getBody()->front();
    }
  };

  static constexpr llvm::StringLiteral placedFile = "placed.mlir";
  static constexpr llvm::StringLiteral linkedFile = "linked.mlir";

  std::string dir;
  std::string configKey;
  bool verbose;
  llvm::StringMap<Entry> devices;

  void replaceHits(mlir::ModuleOp module,
                   mlir::Operation *(Entry::*form)() const) {
    for (auto device :
         llvm::make_early_inc_range(module.getOps<xilinx::AIE::DeviceOp>())) {
      auto it = devices.find(device.getSymName());
      if (it != devices.end() && it->second.hit) {
        mlir::OpBuilder(device).clone(*(it->second.*form)());
        device.erase();
      }
    }
  }

  static void addField(llvm::SHA256 &hash, llvm::StringRef field) {
    hash.update(std::to_string(field.size()));
    hash.update(":");
    hash.update(field);
  }

  static void addToolIdentity(llvm::SHA256 &hash, llvm::StringRef path) {
    addField(hash, path);
    llvm::sys::fs::file_status status;
    if (path.empty() || llvm::sys::fs::status(path, status)) {
      addField(hash, "missing");
      return;
    }
    addField(hash,
             std::to_string(
                 status.getLastModificationTime().time_since_epoch().count()));
    addField(hash, std::to_string(status.getSize()));
  }

  // Cores whose ELF the build is handed rather than compiles are left to the
  // build: their path, not just their content, reaches the output.
  static bool isCacheable(xilinx::AIE::DeviceOp device) {
    bool anyCore = false;
    bool anyPrebuilt = false;
    device.walk([&](xilinx::AIE::CoreOp core) {
      anyCore = true;
      anyPrebuilt |= static_cast<bool>(core.getElfFileAttr());
    });
    return anyCore && !anyPrebuilt;
  }

  static std::string printWithoutLocations(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(/*enable=*/false);
    module.print(os, flags);
    return text;
  }

  static mlir::OwningOpRef<mlir::ModuleOp>
  wrapDevice(xilinx::AIE::DeviceOp device) {
    mlir::OwningOpRef<mlir::ModuleOp> wrapper =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(device.getContext()));
    wrapper->getBody()->push_back(device->clone());
    return wrapper;
  }

  static std::string printDevice(xilinx::AIE::DeviceOp device) {
    std::string text;
    llvm::raw_string_ostream os(text);
    printModuleWithDebugInfo(wrapDevice(device).get(), os);
    return text;
  }

  std::optional<std::string>
  deviceKey(xilinx::AIE::DeviceOp device, llvm::StringRef sharedText,
            llvm::function_ref<std::string(llvm::StringRef)> resolvePath) {
    llvm::SHA256 hash;
    addField(hash, configKey);
    addField(hash, sharedText);
    addField(hash, printWithoutLocations(wrapDevice(device).get()));
    bool readable = true;
    auto addFile = [&](llvm::StringRef name) {
      auto buf = llvm::MemoryBuffer::getFile(resolvePath(name));
      if (!buf) {
        readable = false;
        return;
      }
      addField(hash, name);
      addField(hash, (*buf)->getBuffer());
    };
    device.walk([&](mlir::Operation *op) {
      for (llvm::StringRef attrName :
           {"link_with", "link_files", "link_merge_files"}) {
        mlir::Attribute attr = op->getAttr(attrName);
        if (auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(attr)) {
          addFile(name.getValue());
        } else if (auto names = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
          for (auto name : names.getAsRange<mlir::StringAttr>()) {
            addFile(name.getValue());
          }
        }
      }
    });
    if (!readable) {
      return std::nullopt;
    }
    return llvm::toHex(hash.result(), /*LowerCase=*/true);
  }

  // Parse the entry's two forms of `name` and point the linked cores at the
  // entry's ELFs. Anything short of a complete entry is a miss.
  bool load(Entry &entry, llvm::StringRef name, mlir::MLIRContext *context) {
    mlir::ScopedDiagnosticHandler quiet(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    mlir::ParserConfig config(context, /*verifyAfterParse=*/false);
    auto parse = [&](llvm::StringRef file) {
      llvm::SmallString<256> path(entry.dir);
      llvm::sys::path::append(path, file);
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::parseSourceFile<mlir::ModuleOp>(path, config);
      bool oneDevice =
          module && llvm::hasSingleElement(module->getBody()->getOperations());
      auto device = oneDevice ? mlir::dyn_cast<xilinx::AIE::DeviceOp>(
                                    module->getBody()->front())
                              : xilinx::AIE::DeviceOp();
      if (!device || device.getSymName() != name) {
        return mlir::OwningOpRef<mlir::ModuleOp>();
      }
      return module;
    };
    entry.placed = parse(placedFile);
    entry.linked = parse(linkedFile);
    if (!entry.placed || !entry.linked) {
      return false;
    }
    bool complete = true;
    entry.linked->walk([&](xilinx::AIE::CoreOp core) {
      mlir::StringAttr elf = core.getElfFileAttr();
      llvm::SmallString<256> path(entry.dir);
      if (elf) {
        llvm::sys::path::append(path, elf.getValue());
      }
      if (!elf || llvm::sys::path::has_parent_path(elf.getValue()) ||
          !llvm::sys::fs::exists(path)) {
        complete = false;
        return;
      }
      core.setElfFileAttr(mlir::StringAttr::get(context, path));
    });
    return complete;
  }

  // Write the entry beside its final place, then rename it there, so a reader
  // sees either the whole entry or none of it. Failing to store only costs a
  // later build the reuse.
  void store(const Entry &entry, xilinx::AIE::DeviceOp device,
             const llvm::StringMap<std::string> &elfByKey) {
    mlir::OwningOpRef<mlir::ModuleOp> linked = wrapDevice(device);
    std::vector<std::pair<std::string, std::string>> elfs;
    bool allCompiled = true;
    linked->walk([&](xilinx::AIE::CoreOp core) {
      std::string key = coreKey(core);
      auto it = elfByKey.find(key);
      if (it == elfByKey.end()) {
        allCompiled = false;
        return;
      }
      std::string file = key + ".elf";
      elfs.push_back({it->second, file});
      core.setElfFileAttr(mlir::StringAttr::get(core.getContext(), file));
    });
    if (!allCompiled) {
      return;
    }
    auto warn = [&](llvm::StringRef what, std::error_code ec) {
      llvm::errs() << "aiecc: warning: device cache: cannot " << what << " for "
                   << device.getSymName() << ": " << ec.message() << "\n";
    };
    if (std::error_code ec = llvm::sys::fs::create_directories(dir)) {
      warn("create the cache directory", ec);
      return;
    }
    llvm::SmallString<256> tmp;
    if (std::error_code ec =
            llvm::sys::fs::createUniqueDirectory(entry.dir + ".tmp", tmp)) {
      warn("create an entry", ec);
      return;
    }
    auto writeText = [&](llvm::StringRef file, llvm::StringRef text) {
      llvm::SmallString<256> path(tmp);
      llvm::sys::path::append(path, file);
      std::error_code ec;
      llvm::raw_fd_ostream os(path, ec);
      if (!ec) {
        os << text;
        os.close();
        ec = os.error();
      }
      return ec;
    };
    std::error_code ec = writeText(placedFile, entry.placedText);
    if (!ec) {
      std::string text;
      llvm::raw_string_ostream os(text);
      printModuleWithDebugInfo(linked.get(), os);
      ec = writeText(linkedFile, text);
    }
    for (const auto &[from, file] : elfs) {
      if (ec) {
        break;
      }
      llvm::SmallString<256> to(tmp);
      llvm::sys::path::append(to, file);
      ec = llvm::sys::fs::copy_file(from, to);
    }
    if (!ec && entry.stale) {
      ec = llvm::sys::fs::remove_directories(entry.dir);
    }
    if (!ec) {
      ec = llvm::sys::fs::rename(tmp, entry.dir);
      // Another build may have stored the same entry first.
      if (ec && llvm::sys::fs::exists(entry.dir)) {
        llvm::sys::fs::remove_directories(tmp);
        return;
      }
    }
    if (ec) {
      warn("store an entry", ec);
      llvm::sys::fs::remove_directories(tmp);
      return;
    }
    if (verbose) {
      llvm::errs() << "aiecc: device cache store: " << device.getSymName()
                   << "\n";
    }
  }
};

} // namespace xilinx::aiecc

#endif // AIECC_DEVICECACHE_H
