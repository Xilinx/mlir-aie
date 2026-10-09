//===- device_cache.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandLineOptions.h"
#include "DeviceCache.h"

#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace xilinx::AIE;
using namespace xilinx::aiecc;

int main() {
  MLIRContext context;
  context.getOrLoadDialect<AIEDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      aie.device(npu2) @device {
        %t02 = aie.tile(0, 2)
        %t12 = aie.tile(1, 2)
        %c02 = aie.core(%t02) { aie.end }
        %c12 = aie.core(%t12) { aie.end }
      }
    }
  )mlir",
                                            &context);
  if (!module)
    return 1;

  llvm::SmallString<256> root;
  if (llvm::sys::fs::createUniqueDirectory("aie-device-cache-test", root))
    return 1;
  auto cleanup =
      llvm::scope_exit([&] { llvm::sys::fs::remove_directories(root); });
  llvm::SmallString<256> cacheDir(root);
  llvm::sys::path::append(cacheDir, "cache");

  for (llvm::StringRef name :
       {"../escape", "nested/device", R"(..\escape)", "/absolute", "plain"}) {
    auto device = *module->getOps<DeviceOp>().begin();
    device.setSymName(name);
    DeviceCache cache(cacheDir.str().str(), "test", false);
    cache.active = true;
    auto matches = [](DeviceOp) { return true; };
    auto resolve = [](llvm::StringRef path) { return path.str(); };
    cache.lookup(*module, matches, resolve);
    if (cache.isHit(device) ||
        failed(cache.place(*module, [] { return success(); })))
      return 1;

    llvm::StringMap<std::string> elfs;
    for (auto core : device.getOps<CoreOp>()) {
      auto tile = cast<TileOp>(core.getTile().getDefiningOp());
      std::string col = std::to_string(tile.getCol());
      llvm::SmallString<256> source(root);
      llvm::sys::path::append(source, "input_" + col + ".elf");
      std::error_code ec;
      llvm::raw_fd_ostream os(source, ec);
      if (ec)
        return 1;
      os << name << ":" << col;
      os.close();
      if (os.has_error())
        return 1;
      elfs[coreKey(core)] = source.str().str();
    }
    cache.link(*module, elfs);

    DeviceCache hit(cacheDir.str().str(), "test", false);
    hit.active = true;
    hit.lookup(*module, matches, resolve);
    if (!hit.isHit(device)) {
      llvm::errs() << "cache did not round-trip device " << name << "\n";
      return 1;
    }
    hit.link(*module, {});
    device = *module->getOps<DeviceOp>().begin();
    for (auto core : device.getOps<CoreOp>()) {
      auto tile = cast<TileOp>(core.getTile().getDefiningOp());
      std::string col = std::to_string(tile.getCol());
      llvm::StringRef path = core.getElfFileAttr().getValue();
      auto bytes = llvm::MemoryBuffer::getFile(path);
      if (llvm::sys::path::filename(path) != "core_" + col + "_2.elf" ||
          llvm::sys::path::parent_path(llvm::sys::path::parent_path(path)) !=
              cacheDir ||
          !bytes || (*bytes)->getBuffer() != name.str() + ":" + col)
        return 1;
      core->removeAttr("elf_file");
    }
  }
  // No ELF may escape an entry and land alongside the entry directories.
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(cacheDir, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (llvm::sys::path::extension(it->path()) == ".elf")
      return 1;
  }
  return ec ? 1 : 0;
}
