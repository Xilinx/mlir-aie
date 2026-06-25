//===- AIETargetNPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Runtime/TxnEncoding.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

// Example:
// - instructions = {3,4,5}
// - tailSize = 2
// instructions becomes {3,4,5,0,0} and
// a mutable reference to the tail {0,0} is returned.
llvm::MutableArrayRef<uint32_t>
reserveAndGetTail(std::vector<uint32_t> &instructions, uint64_t tailSize) {
  auto oldSize = instructions.size();
  auto newSize = oldSize + tailSize;
  instructions.resize(newSize, 0);
  return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                         tailSize);
}

// Thin wrappers that extract MLIR attributes and delegate to TxnEncoding.h,
// the single source of truth for the TXN word layout.

void appendSync(std::vector<uint32_t> &instructions, NpuSyncOp op) {
  aie_runtime::txn_append_sync(instructions, op.getColumn(), op.getRow(),
                               static_cast<uint32_t>(op.getDirection()),
                               op.getChannel(), op.getColumnNum(),
                               op.getRowNum());
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {
  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }
  aie_runtime::txn_append_write32(instructions, *op.getAbsoluteAddress(),
                                  op.getValue());
}

void appendMaskWrite32(std::vector<uint32_t> &instructions,
                       NpuMaskWrite32Op op) {
  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }
  aie_runtime::txn_append_maskwrite32(instructions, *op.getAbsoluteAddress(),
                                      op.getValue(), op.getMask());
}

void appendLoadPdi(std::vector<uint32_t> &instructions, NpuLoadPdiOp op) {
  aie_runtime::txn_append_loadpdi(instructions, op.getId(), op.getSize(),
                                  op.getAddress());
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {
  aie_runtime::txn_append_address_patch(instructions, op.getAddr(),
                                        op.getArgIdx(), op.getArgPlus());
}

// Cached resolution of NpuBlockWriteOp's data words.
// NpuBlockWriteOp::getDataWords() resolves the data memref.global via
// device.lookupSymbol (a LINEAR symbol scan over the device's symbols) +
// global.getInitialValue() on EVERY call — perf shows this
// (SymbolTable::lookupSymbolIn + memref::GlobalOp::getInherentAttr) is ~47% of
// AIETranslateNpuToBinary on a B=128 runtime sequence (~10^5 block-writes),
// i.e. O(block-writes x globals). Resolve via a prebuilt per-device SymbolTable
// (O(1)) and memoize per global symbol (the B-unroll's streams reuse the same
// data globals), so the returned attribute is byte-identical but the work
// collapses. Falls back to the canonical method on any non-global memref /
// lookup miss (identical behavior).
static DenseIntElementsAttr cachedBlockWriteData(
    NpuBlockWriteOp op, mlir::SymbolTable &symTab,
    llvm::DenseMap<mlir::StringAttr, DenseIntElementsAttr> &cache) {
  auto getGlobal = op.getData().getDefiningOp<mlir::memref::GetGlobalOp>();
  if (!getGlobal)
    return op.getDataWords();
  // Honor getDataWords()'s 32-bit-element contract exactly: a non-32-bit memref
  // must take the canonical path (which emits "Only 32-bit data type is
  // supported" + returns nullptr), not the cached fast path.
  mlir::DataLayout dataLayout = mlir::DataLayout::closest(op);
  if (dataLayout.getTypeSizeInBits(
          mlir::cast<mlir::MemRefType>(op.getData().getType())
              .getElementType()) != 32)
    return op.getDataWords();
  mlir::StringAttr key = getGlobal.getNameAttr().getRootReference();
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  DenseIntElementsAttr data;
  if (auto global =
          dyn_cast_if_present<mlir::memref::GlobalOp>(symTab.lookup(key)))
    if (auto initVal = global.getInitialValue())
      data = dyn_cast<DenseIntElementsAttr>(*initVal);
  if (!data)
    data = op.getDataWords(); // preserve original error/edge behavior exactly
  cache[key] = data;
  return data;
}

void appendBlockWrite(
    std::vector<uint32_t> &instructions, NpuBlockWriteOp op,
    mlir::SymbolTable &symTab,
    llvm::DenseMap<mlir::StringAttr, DenseIntElementsAttr> &dataCache) {
  std::optional<uint32_t> address = op.getAbsoluteAddress();
  DenseIntElementsAttr data = cachedBlockWriteData(op, symTab, dataCache);

  // Resolve the payload words via the (cached) data attribute, then hand off to
  // the encoder which owns the word layout.
  std::vector<uint32_t> payload;
  payload.reserve(data.size());
  for (auto d : data)
    payload.push_back(d.getZExtValue());

  // The col/row word is only populated when BOTH are present (matching the
  // historical behavior); otherwise it stays 0 (a flat address).
  auto col = op.getColumn();
  auto row = op.getRow();
  uint32_t colVal = 0, rowVal = 0;
  if (col && row) {
    colVal = *col;
    rowVal = *row;
  }
  aie_runtime::txn_append_blockwrite(instructions, *address, payload.data(),
                                     payload.size(), colVal, rowVal);
}

void appendPreempt(std::vector<uint32_t> &instructions, NpuPreemptOp op) {
  aie_runtime::txn_append_preempt(instructions, op.getLevel());
}

void appendCreateScratchpad(std::vector<uint32_t> &instructions,
                            NpuCreateScratchpadOp op) {
  // TXN_OPC_CREATE_SCRATCHPAD encoding (4 words = 16 bytes):
  // Byte 0: Opcode (10)
  // Byte 1: Usage Type
  // Bytes 2-3: padding
  // Bytes 4-7: Size
  // Bytes 8-15: DDR Address (patched at runtime by XRT)
  auto words = reserveAndGetTail(instructions, 4);

  words[0] = aie_runtime::TXN_OPC_CREATE_SCRATCHPAD;
  words[0] |= (static_cast<uint32_t>(op.getUsageType()) << 8);
  words[1] = op.getSize();
  // DDR address words[2] and words[3] are left as 0;
  // they will be patched at runtime by XRT/aiebu based on the
  // .ctrl.scratchpad section.
  words[2] = 0;
  words[3] = 0;
}

void appendUpdateRegFromScratchpad(std::vector<uint32_t> &instructions,
                                   NpuUpdateFromScratchpadOp op) {
  // TXN_OPC_UPDATE_REG encoding (3 words = 12 bytes):
  // Byte 0: Opcode (12)
  // Byte 1: StateTableIdx
  // Byte 2: Func
  // Byte 3: padding
  // Bytes 4-7: FuncArg
  // Bytes 8-11: RegOff (absolute offset from AIE array base to register pair)
  auto words = reserveAndGetTail(instructions, 3);

  words[0] = aie_runtime::TXN_OPC_UPDATE_REG;
  words[0] |= (static_cast<uint32_t>(op.getStateTableIdx()) << 8);
  words[0] |= (static_cast<uint32_t>(op.getFunc()) << 16);
  words[1] = op.getFuncArg();
  words[2] = *op.getAbsoluteAddress();
}

} // namespace

namespace {

// Lazy-loaded regdb for register-name decoration. Loaded once per process,
// returns nullptr if the JSON files cannot be located (e.g. install dir not
// set). Decoration is best-effort.
static const xilinx::AIE::RegisterDatabase *getRegDB() {
  static std::unique_ptr<xilinx::AIE::RegisterDatabase> db = []() {
    return xilinx::AIE::RegisterDatabase::loadAIE2();
  }();
  return db.get();
}

// Decompose an absolute device address into (col, row, offset) using the
// target model's column/row shifts, then derive the regdb module name from
// the row's tile type. Looks the (module, offset) up in regdb. Returns the
// register name on success and writes the module name to *outModule. On any
// failure returns empty StringRef.
static llvm::StringRef
lookupRegisterByAddress(uint64_t address, const AIETargetModel &tm,
                        const xilinx::AIE::RegisterDatabase *regdb,
                        std::string &outModule) {
  outModule.clear();
  if (!regdb)
    return {};
  uint32_t colShift = tm.getColumnShift();
  uint32_t rowShift = tm.getRowShift();
  uint8_t col = static_cast<uint8_t>((address >> colShift) & 0xFF);
  uint8_t row = static_cast<uint8_t>((address >> rowShift) & 0xFF);
  uint64_t baseMask = (uint64_t{1} << rowShift) - 1;
  uint32_t offset = static_cast<uint32_t>(address & baseMask);

  // Determine module from tile type. Module names must match the JSON keys in
  // aie_registers_aie2.json: "core", "memory", "memory_tile", "shim".
  StringRef moduleName;
  if (tm.isShimNOCTile(col, row) || tm.isShimPLTile(col, row)) {
    moduleName = "shim";
  } else if (tm.isMemTile(col, row)) {
    moduleName = "memory_tile";
  } else {
    // For aie tiles, regdb has both core and memory modules. The offset alone
    // disambiguates. Try core first then memory.
    if (auto *r = regdb->lookupRegisterByOffset(offset, "core")) {
      outModule = "core";
      return r->name;
    }
    if (auto *r = regdb->lookupRegisterByOffset(offset, "memory")) {
      outModule = "memory";
      return r->name;
    }
    return {};
  }
  if (auto *r = regdb->lookupRegisterByOffset(offset, moduleName)) {
    outModule = moduleName.str();
    return r->name;
  }
  return {};
}

static void pushLocEntry(std::vector<TxnLocEntry> *locmap,
                         uint32_t byteOffsetBefore, uint32_t byteOffsetAfter,
                         StringRef opcodeName, StringRef sourceOpName,
                         std::optional<uint64_t> address, mlir::Operation *op,
                         const AIETargetModel &tm) {
  if (!locmap)
    return;
  TxnLocEntry e;
  e.byteOffset = byteOffsetBefore;
  e.byteSize = byteOffsetAfter - byteOffsetBefore;
  e.opcodeName = opcodeName.str();
  e.sourceOpName = sourceOpName.str();
  e.address = address;
  e.loc = op->getLoc();
  if (address) {
    if (auto regName =
            lookupRegisterByAddress(*address, tm, getRegDB(), e.registerModule);
        !regName.empty()) {
      e.registerName = regName.str();
    }
  }
  locmap->push_back(std::move(e));
}

} // namespace

LogicalResult xilinx::AIE::AIETranslateNpuToBinary(
    mlir::ModuleOp moduleOp, std::vector<uint32_t> &instructions,
    StringRef deviceName, StringRef sequenceName,
    std::vector<TxnLocEntry> *locmap) {

  DeviceOp deviceOp =
      DeviceOp::getForSymbolInModuleOrError(moduleOp, deviceName);
  if (!deviceOp) {
    return failure();
  }

  const AIETargetModel &tm = deviceOp.getTargetModel();

  // Reserve the 4-word header up front; finalized in-place by
  // txn_prepend_header once all instructions are appended.
  aie_runtime::txn_init(instructions);

  aie_runtime::TxnDeviceInfo devInfo;
  devInfo.devGen = llvm::isa<AIE::BaseNPU2TargetModel>(tm) ? 4 : 3;
  devInfo.numRows = tm.rows();
  devInfo.numCols = tm.columns();
  devInfo.numMemTileRows = tm.getNumMemTileRows();
  uint32_t count = 0;

  AIE::RuntimeSequenceOp seq =
      AIE::RuntimeSequenceOp::getForSymbolInDeviceOrError(deviceOp,
                                                          sequenceName);
  if (!seq) {
    return failure();
  }

  auto byteOffset = [&]() -> uint32_t {
    return static_cast<uint32_t>(instructions.size() * sizeof(uint32_t));
  };

  // Build the device symbol table ONCE + a per-global data cache, so
  // block-write data resolution is O(1)+memoized instead of a per-op linear
  // symbol scan (cachedBlockWriteData). ~47% of this function on a B=128
  // sequence.
  mlir::SymbolTable symTab(deviceOp.getOperation());
  llvm::DenseMap<mlir::StringAttr, DenseIntElementsAttr> blockWriteDataCache;

  for (Block &block : seq.getBody()) {
    for (Operation &o : block) {
      llvm::TypeSwitch<Operation *>(&o)
          .Case<NpuSyncOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendSync(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "TCT",
                         op->getName().getStringRef(), std::nullopt, op, tm);
          })
          .Case<NpuWrite32Op>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            uint64_t addr = op.getAbsoluteAddress().value_or(0);
            appendWrite32(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "WRITE32",
                         op->getName().getStringRef(), addr, op, tm);
          })
          .Case<NpuBlockWriteOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            uint64_t addr = op.getAbsoluteAddress().value_or(0);
            appendBlockWrite(instructions, op, symTab, blockWriteDataCache);
            pushLocEntry(locmap, before, byteOffset(), "BLOCKWRITE",
                         op->getName().getStringRef(), addr, op, tm);
          })
          .Case<NpuMaskWrite32Op>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            uint64_t addr = op.getAbsoluteAddress().value_or(0);
            appendMaskWrite32(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "MASKWRITE",
                         op->getName().getStringRef(), addr, op, tm);
          })
          .Case<NpuLoadPdiOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendLoadPdi(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "LOAD_PDI",
                         op->getName().getStringRef(), std::nullopt, op, tm);
          })
          .Case<NpuAddressPatchOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendAddressPatch(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "ADDRESS_PATCH",
                         op->getName().getStringRef(), op.getAddr(), op, tm);
          })
          .Case<NpuPreemptOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendPreempt(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "PREEMPT",
                         op->getName().getStringRef(), std::nullopt, op, tm);
          })
          .Case<NpuCreateScratchpadOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendCreateScratchpad(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "CREATE_SCRATCHPAD",
                         op->getName().getStringRef(), std::nullopt, op, tm);
          })
          .Case<NpuUpdateFromScratchpadOp>([&](auto op) {
            count++;
            uint32_t before = byteOffset();
            appendUpdateRegFromScratchpad(instructions, op);
            pushLocEntry(locmap, before, byteOffset(), "UPDATE_FROM_SCRATCHPAD",
                         op->getName().getStringRef(), std::nullopt, op, tm);
          });
    }
  }

  // Finalize the TXN header (overwrites the 4 reserved words).
  aie_runtime::txn_prepend_header(instructions, count, devInfo);
  return success();
}

LogicalResult xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
    ModuleOp module, std::vector<uint32_t> &instructions, StringRef deviceName,
    StringRef sequenceName, std::vector<TxnLocEntry> *locmap) {
  DeviceOp deviceOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);
  if (!deviceOp) {
    return failure();
  }
  OpBuilder builder = OpBuilder::atBlockBegin(deviceOp.getBody());
  AIE::RuntimeSequenceOp seq =
      AIE::RuntimeSequenceOp::getForSymbolInDeviceOrError(deviceOp,
                                                          sequenceName);
  if (!seq) {
    return failure();
  }

  const AIETargetModel &tm = deviceOp.getTargetModel();
  Block &entry = seq.getBody().front();
  for (auto &o : entry) {
    auto packetOp = dyn_cast<AIEX::NpuControlPacketOp>(o);
    if (!packetOp)
      continue;

    uint32_t before =
        static_cast<uint32_t>(instructions.size() * sizeof(uint32_t));

    uint32_t size = 0;
    auto data = packetOp.getData();
    if (data)
      size = data->size();

    auto words = reserveAndGetTail(instructions, 2 + size);

    if (!data && packetOp.getLength())
      size = *packetOp.getLength();

    auto parity = [](uint32_t n) {
      uint32_t p = 0;
      while (n) {
        p += n & 1;
        n >>= 1;
      }
      return (p % 2) == 0;
    };

    // stream header is attached here instead of by shim dma
    int col = packetOp.getColumnFromAddr();
    int row = packetOp.getRowFromAddr();
    auto destTile = TileOp::getOrCreate(builder, deviceOp, col, row);
    auto info = destTile->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
    uint32_t hdr = 0;
    if (info)
      hdr = (info.getPktType() & 0x7) << 12 | (info.getPktId() & 0xff);
    else
      destTile->emitWarning("Expected controller_id attribute");
    words[0] = hdr | (0x1 & parity(hdr)) << 31;

    // control packet header
    uint32_t addr = packetOp.getAddress() & 0xFFFFF;
    uint32_t beats = size - 1;
    uint32_t opc = packetOp.getOpcode();
    uint32_t id = packetOp.getStreamId();
    hdr = id << 24 | opc << 22 | beats << 20 | addr;
    words[1] = hdr | (0x1 & parity(hdr)) << 31;

    // configuration data
    if (opc == 0x0 || opc == 0x2)
      for (unsigned i = 0; i < size; i++)
        words[i + 2] = data.value()[i];

    uint32_t after =
        static_cast<uint32_t>(instructions.size() * sizeof(uint32_t));
    pushLocEntry(locmap, before, after, "CONTROL_PACKET",
                 packetOp->getName().getStringRef(), packetOp.getAddress(),
                 packetOp, tm);
  }
  return success();
}

// Render an mlir::Location as a JSON object, recursing into NameLoc, FusedLoc,
// and CallSiteLoc so a consumer can reconstruct the file:line:col + named
// location structure carried by the source op.
static llvm::json::Value locToJSON(mlir::Location loc) {
  using namespace llvm;
  if (auto f = dyn_cast<mlir::FileLineColLoc>(loc))
    return json::Value(json::Object{
        {"kind", "file"},
        {"file", f.getFilename().str()},
        {"line", static_cast<int64_t>(f.getLine())},
        {"col", static_cast<int64_t>(f.getColumn())},
    });
  if (auto n = dyn_cast<mlir::NameLoc>(loc)) {
    json::Object o{{"kind", "name"}, {"name", n.getName().str()}};
    o["child"] = locToJSON(n.getChildLoc());
    return json::Value(std::move(o));
  }
  if (auto fused = dyn_cast<mlir::FusedLoc>(loc)) {
    json::Array children;
    for (mlir::Location c : fused.getLocations())
      children.push_back(locToJSON(c));
    return json::Value(
        json::Object{{"kind", "fused"}, {"children", std::move(children)}});
  }
  if (auto cs = dyn_cast<mlir::CallSiteLoc>(loc))
    return json::Value(json::Object{
        {"kind", "callsite"},
        {"callee", locToJSON(cs.getCallee())},
        {"caller", locToJSON(cs.getCaller())},
    });
  if (isa<mlir::OpaqueLoc>(loc))
    return json::Value(json::Object{{"kind", "opaque"}});
  return json::Value(json::Object{{"kind", "unknown"}});
}

void xilinx::AIE::emitNpuLocmapJSON(llvm::raw_ostream &output,
                                    llvm::StringRef deviceName,
                                    llvm::StringRef binaryName,
                                    const std::vector<TxnLocEntry> &locmap) {
  llvm::json::Object root;
  root["version"] = 1;
  root["device"] = deviceName.str();
  root["binary"] = binaryName.str();

  llvm::json::Array opsArr;
  for (const auto &e : locmap) {
    llvm::json::Object o;
    o["byte_offset"] = static_cast<int64_t>(e.byteOffset);
    o["byte_size"] = static_cast<int64_t>(e.byteSize);
    o["opcode"] = e.opcodeName;
    o["source_op"] = e.sourceOpName;
    if (e.address)
      o["address"] = llvm::formatv("{0:X}", *e.address).str();
    if (!e.registerName.empty()) {
      o["register"] = e.registerName;
      o["register_module"] = e.registerModule;
    }
    if (e.loc)
      o["loc"] = locToJSON(*e.loc);
    opsArr.push_back(std::move(o));
  }
  root["operations"] = std::move(opsArr);

  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
}
