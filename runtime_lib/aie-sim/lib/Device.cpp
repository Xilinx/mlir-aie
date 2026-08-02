//===- Device.cpp -------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// See include/aiesim/Device.h for the contract. Every number below is
// grounded in a specific line of either mlir-aie (the AIE dialect's target
// model, which is what the compiler assumes) or aie-rt (the vendored driver,
// which is what actually programs the hardware and is what the sim boundary
// is speaking to). Where both exist they are cross-checked; where only one
// exists that is called out.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Device.h"

namespace aiesim {

DecodedAddress decodeAddress(const DeviceModel &dev, uint64_t addr) {
  if (addr < dev.baseAddr)
    return {0, 0, 0, false};

  uint64_t local = addr - dev.baseAddr;

  // Inverts _XAie_GetTileAddr's [regOff | row | col] layout
  // (xaie_helper.h:151-155); assumes ColShift > RowShift, true for both
  // generations here. aie-rt's _XAie_GetRowfromRegOff /
  // _XAie_GetColfromRegOff are NOT this inverse and must not be substituted
  // -- they decode a TXN command header instead; see docs/AIESimulator.md 4.1.
  uint64_t rowFieldMask = (1ULL << (dev.colShift - dev.rowShift)) - 1;
  uint32_t col = static_cast<uint32_t>(local >> dev.colShift);
  uint32_t row = static_cast<uint32_t>((local >> dev.rowShift) & rowFieldMask);
  uint32_t regOff = static_cast<uint32_t>(local & ((1ULL << dev.rowShift) - 1));

  return {col, row, regOff, dev.contains(col, row)};
}

namespace {

// Per-tile constants shared by every AIE2-family device this simulator
// builds (AIE2/npu1, AIE2P/npu2, and the Versal xcve2802 shape, which is an
// AIE2TargetModel subclass with no overrides for any of these). Grounded in
// two sources that agree exactly wherever both cover a field:
//
//   (a) mlir-aie's AIETargetModel.h (AIE2TargetModel, the base every one of
//       these devices derives from; BaseNPU1TargetModel/BaseNPU2TargetModel/
//       VE2802TargetModel override only shape, never these fields):
//         getColumnShift/getRowShift ....... AIETargetModel.h:738-739
//         getLocalMemorySize (core data) ... AIETargetModel.h:640 (0x10000)
//         getMemTileSize .................... AIETargetModel.h:711 (0x80000)
//         getNumLocks (16 core/shim, 64 memtile) .. AIETargetModel.h:645-647
//         getNumBDs (16 core/shim, 48 memtile) .... AIETargetModel.h:654-656
//
//   (b) aie-rt's per-generation register-init tables. The "aie2ipu" tables
//       (real AIE2 / npu1's XAIE_DEV_GEN_AIE2IPU) and the "aie2p" tables
//       (AIE2P / npu2's XAIE_DEV_GEN_AIE2P_STRIX_B0) are numerically
//       IDENTICAL for every field below -- confirmed by reading both, not
//       assumed:
//         third_party/aie-rt/driver/src/global/xaie2ipugbl_reginit.c
//           ProgMemSize/ProgMemHostOffset/DataMemSize: 2270-2273
//           core data mem Size=0x10000, memtile mem Size=0x80000: 2298, 2306
//           lock counts 16/16/64: 2358, 2383, 2408
//           BD counts 16 (core), BD counts 48 + 6 chan (memtile): 391-405
//           BD counts 16 + 2 chan (core-ish/shim-ish): 627-641, 878-892
//         third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c
//           ProgMemSize/ProgMemHostOffset/DataMemSize: 181-184
//           core data mem Size=0x10000, memtile mem Size=0x80000: 2187, 2195
//           lock counts 16/16/64: 2387, 2412, 2437
//           MemTile BDs=48, NumChannels=6: 1656, 1670
//           Core tile BDs=16, NumChannels=2: 1893, 1907
//           Shim BDs=16, NumChannels=2: 2144, 2158
//
//   XAIE_BASE_ADDR = 0x40000000 has no aie-rt counterpart to cross-check --
//   it is a constant mlir-aie itself picks when it builds the XAie_Config
//   passed to XAie_CfgInitialize (lib/Targets/AIERT.cpp:187,264), not a
//   property of the silicon. aie-rt just stores whatever it is given
//   (driver/src/global/xaiegbl.c:2xx), so there is nothing in aie-rt to
//   compare it against; it is trusted as given by the one host program that
//   actually drives this hardware.
//
//   coreProgMemSize, progMemHostOffset, and every DMA-channel count are
//   aie-rt-only: AIETargetModel.h has no field for ELF program-memory
//   layout or per-tile-type DMA channel counts (getNumBDs covers buffer
//   descriptors, not channels), so those four have only source (b).
DeviceModel fillAIE2Family(uint32_t numCols, uint32_t numRows,
                           uint32_t numMemTileRows, Generation gen) {
  DeviceModel dev{};
  dev.generation = gen;
  dev.numCols = numCols;
  dev.numRows = numRows;
  dev.numMemTileRows = numMemTileRows;

  dev.colShift = 25;         // AIETargetModel.h:738
  dev.rowShift = 20;         // AIETargetModel.h:739
  // The HOST program's base address, which is the only one this simulator
  // ever sees: the generated mlir_aie_init_libxaie() sets
  // XAieConfig->BaseAddr = 0x20000000000 (lib/Targets/AIETargetXAIEV2.cpp:383)
  // and never calls XAie_SetupPartitionConfig, so DevInst->NumCols is still 0
  // when XAie_CfgInitialize runs and the copy at
  // third_party/aie-rt/driver/src/global/xaiegbl.c:198-202 does take effect.
  //
  // Do NOT use AIERT.cpp's XAIE_BASE_ADDR (0x40000000). That is the
  // COMPILER-side CDO path, it is a different number, and there it never even
  // takes effect: AIERT.cpp:280 calls XAie_SetupPartitionConfig first, which
  // makes NumCols nonzero, so the same copy is skipped and the live base stays
  // XAIE_PARTITION_BASE_ADDR (0x0, AIERT.cpp:190). An earlier version of this
  // file used 0x40000000 and would have rejected every access a real host
  // program makes. Found by the aie-rt integration test, which is the whole
  // reason that tier exists.
  dev.baseAddr = 0x20000000000;

  dev.coreDataMemSize = 0x00010000; // 64KB, AIETargetModel.h:640
  dev.coreProgMemSize = 16 * 1024;  // aie-rt only, ProgMemSize (see above)
  dev.memTileMemSize = 0x00080000;  // 512KB, AIETargetModel.h:711
  // aie-rt only: XAIE2PGBL_CORE_MODULE_PROGRAM_MEMORY (xaie2pgbl_params.h:38)
  // and XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY (xaiemlgbl_params.h:41) are
  // both 0x00020000.
  dev.progMemHostOffset = 0x00020000;

  dev.numCoreLocks = 16;    // AIETargetModel.h:645-647
  dev.numMemTileLocks = 64; // AIETargetModel.h:646
  dev.numShimLocks = 16;    // AIETargetModel.h:645-647 (non-MemTile branch)

  // aie-rt only: XAie_DmaMod::NumChannels, one count per generation's DMA
  // module, applies to S2MM and MM2S alike (separate ChCtrlBase per
  // direction, same NumChannels).
  dev.numCoreDmaChannelsS2MM = 2;
  dev.numCoreDmaChannelsMM2S = 2;
  dev.numMemTileDmaChannelsS2MM = 6;
  dev.numMemTileDmaChannelsMM2S = 6;
  dev.numShimDmaChannelsS2MM = 2;
  dev.numShimDmaChannelsMM2S = 2;

  dev.numCoreBds = 16;    // AIETargetModel.h:654-656
  dev.numMemTileBds = 48; // AIETargetModel.h:654-656
  dev.numShimBds = 16;    // AIETargetModel.h:654-656 (default branch)

  return dev;
}

} // namespace

DeviceModel makeAIE2Device(uint32_t numCols) {
  // BaseNPU1TargetModel: rows()=6 (1 shim + 1 memtile + 4 core rows),
  // getNumMemTileRows()=1. AIETargetModel.h:842-846.
  return fillAIE2Family(numCols, /*numRows=*/6, /*numMemTileRows=*/1,
                        Generation::AIE2);
}

DeviceModel makeAIE2PDevice(uint32_t numCols) {
  // BaseNPU2TargetModel: rows()=6, getNumMemTileRows()=1. Same shape as
  // NPU1 -- the generations differ in ISA/core (AIEArch::AIE2 vs AIE2p,
  // lib/Dialect/AIE/IR/AIETargetModel.cpp:1599) and in a couple of
  // shim-only properties (burst encodings, block-format support), never
  // in the address shifts or per-tile resource counts.
  // AIETargetModel.h:894-906.
  return fillAIE2Family(numCols, /*numRows=*/6, /*numMemTileRows=*/1,
                        Generation::AIE2P);
}

bool makeDeviceFromName(const std::string &name, DeviceModel &out,
                        std::string &error) {
  // Device names and shapes as declared in the AIE dialect:
  //   include/aie/Dialect/AIE/IR/AIEAttrs.td:110-126 (the AIEDevice enum)
  //   lib/Dialect/AIE/IR/AIEDialect.cpp:182-241 (name -> TargetModel)
  if (name == "npu1") {
    // AIEDialect.cpp:217-218: npu1 -> NPUmodel4col
    // (VirtualizedNPU1TargetModel(4)).
    out = makeAIE2Device(4);
    return true;
  }
  if (name == "npu1_1col") {
    out = makeAIE2Device(1);
    return true;
  }
  if (name == "npu1_2col") {
    out = makeAIE2Device(2);
    return true;
  }
  if (name == "npu1_3col") {
    out = makeAIE2Device(3);
    return true;
  }
  // There is no "npu1_4col": npu1 itself is the full 4-column Phoenix array
  // (AIEAttrs.td:115-118, AIEDialect.cpp:185-188,217-218), so that name
  // falls through to the "unknown" error below rather than being guessed.
  if (name == "npu2") {
    // NPU2TargetModel::columns() == 8. AIETargetModel.h:924.
    out = makeAIE2PDevice(8);
    return true;
  }
  if (name == "npu2_1col") {
    out = makeAIE2PDevice(1);
    return true;
  }
  if (name == "npu2_2col") {
    out = makeAIE2PDevice(2);
    return true;
  }
  if (name == "npu2_3col") {
    out = makeAIE2PDevice(3);
    return true;
  }
  if (name == "npu2_4col") {
    out = makeAIE2PDevice(4);
    return true;
  }
  if (name == "npu2_5col") {
    out = makeAIE2PDevice(5);
    return true;
  }
  if (name == "npu2_6col") {
    out = makeAIE2PDevice(6);
    return true;
  }
  if (name == "npu2_7col") {
    out = makeAIE2PDevice(7);
    return true;
  }
  if (name == "xcve2802") {
    // VE2802TargetModel: columns()=38, rows()=11, getNumMemTileRows()=2
    // (AIETargetModel.h:805-833), and it overrides nothing else -- it is an
    // AIE2TargetModel subclass, so it shares fillAIE2Family's shifts,
    // memory sizes, lock counts and BD counts. Only the array shape
    // differs from the NPU1/NPU2 6-row/1-memtile-row layout.
    out = fillAIE2Family(38, /*numRows=*/11, /*numMemTileRows=*/2,
                         Generation::AIE2);
    return true;
  }
  if (name == "xcvc1902" || name == "xcve2302") {
    // Real devices (AIEAttrs.td:112-113), but AIE1 (xcvc1902) and a second
    // non-NPU AIE2 shape (xcve2302) that nothing in this task asked for and
    // that would just be more unverified geometry to carry, so this is a
    // deliberate refusal rather than a silent guess.
    error = "device \"" + name +
            "\" is a recognised AIE dialect device but is not modeled by "
            "this simulator";
    return false;
  }
  error = "unknown device name \"" + name +
          "\"; expected one of: npu1, npu1_1col, npu1_2col, npu1_3col, "
          "npu2, npu2_1col, npu2_2col, npu2_3col, npu2_4col, npu2_5col, "
          "npu2_6col, npu2_7col, xcve2802";
  return false;
}

} // namespace aiesim
