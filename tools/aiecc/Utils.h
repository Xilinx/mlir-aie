//===- Utils.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Domain-agnostic helpers used across aiecc — currently filesystem-path
// mangling. Nothing in here should know about MLIR/AIE types.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_UTILS_H
#define AIECC_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace xilinx::aiecc {

// Reinterpret a u32 instruction stream as its raw little-endian bytes. Every
// AIE binary translator emits `std::vector<uint32_t>`, but artifacts are
// materialized as bytes, so this conversion is shared by all of them.
inline std::vector<char> wordsToBytes(llvm::ArrayRef<uint32_t> words) {
  const auto *p = reinterpret_cast<const char *>(words.data());
  return std::vector<char>(p, p + words.size() * sizeof(uint32_t));
}

// Absolutize a path against the current working directory.
inline std::string absolutePath(llvm::StringRef p) {
  llvm::SmallString<256> abs(p);
  llvm::sys::fs::make_absolute(abs);
  return std::string(abs);
}

// Resolve a (possibly relative) path against, in order: cwd, workDir, and the
// directory containing inputFile. The first candidate that exists wins; if
// none exist, the inputFile-relative path is returned as a best-effort guess.
// Absolute inputs pass through unchanged.
inline std::string resolveExternalPath(llvm::StringRef rel,
                                       llvm::StringRef inputFile,
                                       llvm::StringRef workDir) {
  if (llvm::sys::path::is_absolute(rel))
    return rel.str();

  llvm::SmallString<256> cwd;
  llvm::sys::fs::current_path(cwd);
  llvm::sys::path::append(cwd, rel);
  if (llvm::sys::fs::exists(cwd))
    return std::string(cwd);

  if (!workDir.empty()) {
    llvm::SmallString<256> w(workDir);
    llvm::sys::path::append(w, rel);
    if (llvm::sys::fs::exists(w))
      return std::string(w);
  }

  if (!inputFile.empty()) {
    llvm::SmallString<256> inDir = llvm::sys::path::parent_path(inputFile);
    if (inDir.empty())
      llvm::sys::fs::current_path(inDir);
    llvm::sys::path::append(inDir, rel);
    llvm::sys::path::remove_dots(inDir, /*remove_dot_dot=*/true);
    return std::string(inDir);
  }
  return rel.str();
}

// Extract the object-file paths referenced by a BCF's `_include _file <path>`
// directives. The chess linker needs these objects passed on its command line
// (the BCF directive alone does not pull them into the link), so the xbridge
// link step resolves and forwards each one.
inline std::vector<std::string> parseBcfIncludeFiles(llvm::StringRef bcf) {
  std::vector<std::string> files;
  llvm::SmallVector<llvm::StringRef> lines;
  bcf.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.consume_front("_include _file ")) {
      llvm::StringRef path = line.trim();
      if (!path.empty())
        files.push_back(path.str());
    }
  }
  return files;
}

// Discover the aietools (Vitis AIE / Chess) install directory. Search order:
//   1. an explicit override (e.g. the --aietools flag),
//   2. the $AIETOOLS_ROOT environment variable,
//   3. derive it from `xchesscc` on PATH — it lives at <aietools>/bin/xchesscc
//      or <aietools>/bin/unwrapped/lnx64.o/xchesscc.
// Returns "" if none of these resolve to a directory.
inline std::string discoverAietoolsDir(llvm::StringRef overrideDir = "") {
  if (!overrideDir.empty() && llvm::sys::fs::is_directory(overrideDir))
    return overrideDir.str();
  if (const char *env = std::getenv("AIETOOLS_ROOT"))
    if (llvm::sys::fs::is_directory(env))
      return std::string(env);
  auto xchesscc = llvm::sys::findProgramByName("xchesscc");
  if (xchesscc) {
    llvm::StringRef binDir = llvm::sys::path::parent_path(*xchesscc);
    // Unwrap <aietools>/bin/unwrapped/lnx64.o/xchesscc back to <aietools>/bin.
    if (llvm::sys::path::filename(binDir) == "lnx64.o")
      binDir = llvm::sys::path::parent_path(
          llvm::sys::path::parent_path(binDir)); // up from lnx64.o/unwrapped
    return std::string(llvm::sys::path::parent_path(binDir)); // up from bin
  }
  return "";
}

// Install prefix of the running executable (parent of bin/), used to locate
// aie_runtime_lib. Falls back to a sibling install/ dir, then
// $MLIR_AIE_INSTALL_DIR, then the computed prefix.
inline std::string getInstallDir() {
  std::string mainExe = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getInstallDir));
  llvm::SmallString<256> prefix(
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(mainExe)));
  auto hasRuntimeLib = [](llvm::StringRef dir) {
    llvm::SmallString<256> p(dir);
    llvm::sys::path::append(p, "aie_runtime_lib");
    return llvm::sys::fs::is_directory(p);
  };
  if (hasRuntimeLib(prefix))
    return std::string(prefix);
  llvm::SmallString<256> sibling(llvm::sys::path::parent_path(prefix));
  llvm::sys::path::append(sibling, "install");
  if (hasRuntimeLib(sibling))
    return std::string(sibling);
  if (const char *env = std::getenv("MLIR_AIE_INSTALL_DIR"))
    if (llvm::sys::fs::is_directory(env))
      return std::string(env);
  return std::string(prefix);
}

} // namespace xilinx::aiecc

#endif // AIECC_UTILS_H
