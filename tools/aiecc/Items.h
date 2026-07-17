//===- Items.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Payload types carried by the compilation graph's Items, plus their
// serialization traits: `Serializer<T>` writes a payload to disk;
// `Deserializer<T>` lifts a tool-produced file back into a typed payload. Add
// new payload types and their traits here.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_ITEMS_H
#define AIECC_ITEMS_H

#include "aie/Targets/AIETargets.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace xilinx::aiecc {

//===----------------------------------------------------------------------===//
// Payload types
//===----------------------------------------------------------------------===//

// `File` is an opaque tag: the on-disk path lives on ItemBase::filePath.
struct File {};

// A `Directory` groups files that must travel together on disk: the CDO's set
// of `.bin` files, or a compiled core's ELF plus chess's `.map`/`.lst`/...
// sidecars. Like `File` the payload is opaque and ItemBase::filePath holds the
// item's primary on-disk path -- for a CDO that path is the directory itself,
// for a core it is the ELF/object inside the bundle -- so asFile() (and hence
// link inputs / `elf_file` patching) keeps yielding the right thing. `dir`
// names the folder whose entire contents travel with the item (recursively
// copied when a checkpoint captures it).
struct Directory {
  std::string dir;
};

// File-like payloads carry their data as on-disk path(s) rather than an
// in-memory value: asFile() returns filePath verbatim and asString() is
// invalid. `Directory` additionally tracks a companion folder that travels
// with the item.
template <typename T>
constexpr bool IsFileLikeV =
    std::is_same_v<T, File> || std::is_same_v<T, Directory>;

// A whole-module clone paired with a pointer to one op inside it. Lambdas
// see the full surrounding context while `op` identifies the focus.
template <typename KeyOp>
struct OpInModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  KeyOp op;
};

// One runtime sequence's NPU program: the transaction instruction binary and
// its source-location map. Both fall out of a single AIETranslateNpuToBinary
// call, so they travel together and translation is never repeated.
struct NpuProgram {
  std::vector<char> insts;
  std::vector<xilinx::AIE::TxnLocEntry> locmap;
  std::string deviceName;
};

//===----------------------------------------------------------------------===//
// Serializer: how a materialized payload is written to disk
//===----------------------------------------------------------------------===//

// Per-payload write trait. Specialize for any payload that needs custom
// serialization; the primary template handles anything with operator<<.
template <typename T>
struct Serializer {
  static void write(const T &value, llvm::raw_ostream &os) { os << value; }
};

// Print a module with source-location info: materialized MLIR is only read by
// humans, so keep the `loc(...)` that traces an op back to its origin.
inline void printModuleWithDebugInfo(mlir::ModuleOp m, llvm::raw_ostream &os) {
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true);
  m.print(os, flags);
}

template <>
struct Serializer<mlir::OwningOpRef<mlir::ModuleOp>> {
  static void write(const mlir::OwningOpRef<mlir::ModuleOp> &value,
                    llvm::raw_ostream &os) {
    printModuleWithDebugInfo(value.get(), os);
  }
};

template <typename KeyOp>
struct Serializer<OpInModule<KeyOp>> {
  static void write(const OpInModule<KeyOp> &value, llvm::raw_ostream &os) {
    printModuleWithDebugInfo(value.module.get(), os);
  }
};

template <>
struct Serializer<std::vector<char>> {
  static void write(const std::vector<char> &value, llvm::raw_ostream &os) {
    os.write(value.data(), value.size());
  }
};

// List of argument strings, written one per line when materialized.
template <>
struct Serializer<std::vector<std::string>> {
  static void write(const std::vector<std::string> &value,
                    llvm::raw_ostream &os) {
    for (const auto &s : value)
      os << s << "\n";
  }
};

template <>
struct Serializer<llvm::json::Value> {
  static void write(const llvm::json::Value &value, llvm::raw_ostream &os) {
    os << llvm::formatv("{0:2}", value);
  }
};

// Materializing an NpuProgram writes the instruction binary; the locmap is
// emitted separately as a JSON sidecar (see the --keep-loc edge).
template <>
struct Serializer<NpuProgram> {
  static void write(const NpuProgram &value, llvm::raw_ostream &os) {
    os.write(value.insts.data(), value.insts.size());
  }
};

//===----------------------------------------------------------------------===//
// Deserializer: how a tool-produced file is lifted back into a payload
//===----------------------------------------------------------------------===//

// Opaque bag of ambient resources a Deserializer may need to lift a payload
// from disk on --resume. The graph and execution engine forward this through
// without inspecting it.
struct DeserializeContext {
  // The context module-valued payloads are re-parsed into on --resume. Null
  // when no MLIR payload can appear at a resumable cut.
  mlir::MLIRContext *mlirContext = nullptr;
};

// Parse a .mlir file back into a module in `ctx`.
inline mlir::OwningOpRef<mlir::ModuleOp>
parseModuleFromFile(llvm::StringRef path, mlir::MLIRContext *ctx) {
  assert(ctx && "parseModuleFromFile requires an MLIRContext");
  std::string err;
  auto buf = mlir::openInputFile(path, &err);
  if (!buf) {
    llvm::errs() << "aiecc: " << err << "\n";
    return nullptr;
  }
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler dh(sm, ctx);
  return mlir::parseSourceFile<mlir::ModuleOp>(sm, ctx);
}

// Per-payload read trait: the inverse of Serializer. Specialize for payloads an
// external tool emits as a file, to lift them back into a typed Item so
// downstream edges see the parsed value instead of raw bytes. Undefined by
// default; opt in per type. `dc` carries any ambient resource the read needs
// (e.g. an MLIRContext); specializations that don't need one ignore it.
template <typename T>
struct Deserializer;

// A File payload is just its on-disk path (held on the Item), so lifting it is
// a no-op.
template <>
struct Deserializer<File> {
  static mlir::FailureOr<File> read(llvm::StringRef /*path*/,
                                    const DeserializeContext & /*dc*/) {
    return File{};
  }
};

// An IR frontier is re-parsed from its saved .mlir text into the shared
// MLIRContext carried by `dc`.
template <>
struct Deserializer<mlir::OwningOpRef<mlir::ModuleOp>> {
  static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
  read(llvm::StringRef path, const DeserializeContext &dc) {
    if (!dc.mlirContext) {
      llvm::errs() << "aiecc: cannot parse module '" << path
                   << "': no MLIRContext available\n";
      return mlir::failure();
    }
    mlir::OwningOpRef<mlir::ModuleOp> mod =
        parseModuleFromFile(path, dc.mlirContext);
    if (!mod) {
      llvm::errs() << "aiecc: failed to parse '" << path << "'\n";
      return mlir::failure();
    }
    return std::move(mod);
  }
};

template <>
struct Deserializer<llvm::json::Value> {
  static mlir::FailureOr<llvm::json::Value>
  read(llvm::StringRef path, const DeserializeContext & /*dc*/) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
      llvm::errs() << "aiecc: cannot read JSON file '" << path
                   << "': " << buf.getError().message() << "\n";
      return mlir::failure();
    }
    auto parsed = llvm::json::parse((*buf)->getBuffer());
    if (!parsed) {
      llvm::errs() << "aiecc: invalid JSON in '" << path
                   << "': " << llvm::toString(parsed.takeError()) << "\n";
      return mlir::failure();
    }
    return std::move(*parsed);
  }
};

template <>
struct Deserializer<std::vector<char>> {
  static mlir::FailureOr<std::vector<char>>
  read(llvm::StringRef path, const DeserializeContext & /*dc*/) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
      llvm::errs() << "aiecc: cannot read file '" << path
                   << "': " << buf.getError().message() << "\n";
      return mlir::failure();
    }
    llvm::StringRef data = (*buf)->getBuffer();
    return std::vector<char>(data.begin(), data.end());
  }
};

template <>
struct Deserializer<std::string> {
  static mlir::FailureOr<std::string> read(llvm::StringRef path,
                                           const DeserializeContext & /*dc*/) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
      llvm::errs() << "aiecc: cannot read file '" << path
                   << "': " << buf.getError().message() << "\n";
      return mlir::failure();
    }
    return (*buf)->getBuffer().str();
  }
};

// Trait: does `T` have a `Deserializer<T>::read` (i.e. can a cut be resumed at
// a node carrying `T`)? Used to gate loading pre-generated artifacts from disk.
template <typename T, typename = void>
struct HasDeserializer : std::false_type {};
template <typename T>
struct HasDeserializer<T, std::void_t<decltype(Deserializer<T>::read(
                              std::declval<llvm::StringRef>(),
                              std::declval<const DeserializeContext &>()))>>
    : std::true_type {};
template <typename T>
inline constexpr bool HasDeserializerV = HasDeserializer<T>::value;

} // namespace xilinx::aiecc

#endif // AIECC_ITEMS_H
