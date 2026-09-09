//===- device_test.cpp ----------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for the address decode and device tables in lib/Device.cpp. See that
// file for where each grounded number comes from; this file just checks the
// arithmetic and the tables against those grounded values, so a future edit
// that quietly changes one fails a test instead of a simulation.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Device.h"

#include "TestSupport.h"

using namespace aiesim;

namespace {

// Builds a flat address the way aie-rt does: _XAie_GetTileAddr's
// (Row << RowShift) | (Col << ColShift), plus the intra-tile register
// offset, plus the device base address the IO backend adds
// (BaseAddr + RegOff in every xaie_*.c IO backend). Used to construct test
// inputs, independent of decodeAddress's own implementation.
uint64_t encodeAddr(const DeviceModel &dev, uint32_t col, uint32_t row,
                    uint32_t regOff) {
  uint64_t tileAddr = (static_cast<uint64_t>(row) << dev.rowShift) |
                      (static_cast<uint64_t>(col) << dev.colShift);
  return dev.baseAddr + tileAddr + regOff;
}

void checkRoundTrip(const DeviceModel &dev, const char *label) {
  // A handful of (col, row, regOff) tuples, including the partition
  // boundaries and a couple of representative register offsets.
  struct Case {
    uint32_t col, row, regOff;
  };
  Case cases[] = {
      {0, 0, 0},
      {0, 0, 0x1000},
      {dev.numCols - 1, 0, 4},
      {0, dev.numMemTileRows, 0x2000},
      {dev.numCols - 1, dev.numRows - 1, 0xFFF0},
      {1, 2, 0x10},
  };
  for (const Case &c : cases) {
    uint64_t addr = encodeAddr(dev, c.col, c.row, c.regOff);
    DecodedAddress d = decodeAddress(dev, addr);
    AIESIM_CHECK(d.valid);
    AIESIM_CHECK_EQ(d.col, c.col);
    AIESIM_CHECK_EQ(d.row, c.row);
    AIESIM_CHECK_EQ(d.regOff, c.regOff);
    if (!d.valid || d.col != c.col || d.row != c.row || d.regOff != c.regOff) {
      std::fprintf(stderr, "  (in %s round trip: col=%u row=%u regOff=%u)\n",
                   label, c.col, c.row, c.regOff);
    }
  }
}

void checkOutOfPartition(const DeviceModel &dev, const char *label) {
  // One column step and one row step past the partition, plus an address
  // that undercuts the base address entirely.
  uint64_t overColAddr = encodeAddr(dev, dev.numCols, 0, 0);
  AIESIM_CHECK(!decodeAddress(dev, overColAddr).valid);

  uint64_t overRowAddr = encodeAddr(dev, 0, dev.numRows, 0);
  AIESIM_CHECK(!decodeAddress(dev, overRowAddr).valid);

  DecodedAddress underBase = decodeAddress(dev, dev.baseAddr - 1);
  AIESIM_CHECK(!underBase.valid);
  (void)label;
}

void checkTileTypes(const DeviceModel &dev) {
  AIESIM_CHECK(dev.tileTypeAt(0) == TileType::Shim);
  for (uint32_t r = 1; r <= dev.numMemTileRows; ++r)
    AIESIM_CHECK(dev.tileTypeAt(r) == TileType::MemTile);
  for (uint32_t r = dev.numMemTileRows + 1; r < dev.numRows; ++r)
    AIESIM_CHECK(dev.tileTypeAt(r) == TileType::Core);
  AIESIM_CHECK(dev.tileTypeAt(dev.numRows) == TileType::Invalid);
}

// Every field grounded in lib/Device.cpp's fillAIE2Family, common to
// AIE2, AIE2P and the xcve2802 shape. A change to any of these should be a
// deliberate edit to a comment-and-citation pair in Device.cpp, never a
// silent drift, so this checks all of them.
void checkAIE2FamilyConstants(const DeviceModel &dev) {
  AIESIM_CHECK_EQ(dev.colShift, 25u);
  AIESIM_CHECK_EQ(dev.rowShift, 20u);
  // The HOST program's base address (AIETargetXAIEV2.cpp:383), not the
  // compiler-side CDO one. See the comment in Device.cpp: 0x40000000 was
  // wrong for both paths and would have rejected every real host access.
  AIESIM_CHECK_EQ(dev.baseAddr, static_cast<uint64_t>(0x20000000000));

  AIESIM_CHECK_EQ(dev.coreDataMemSize, 0x00010000u);
  AIESIM_CHECK_EQ(dev.coreProgMemSize, 16u * 1024u);
  AIESIM_CHECK_EQ(dev.memTileMemSize, 0x00080000u);
  AIESIM_CHECK_EQ(dev.progMemHostOffset, 0x00020000u);

  AIESIM_CHECK_EQ(dev.numCoreLocks, 16u);
  AIESIM_CHECK_EQ(dev.numMemTileLocks, 64u);
  AIESIM_CHECK_EQ(dev.numShimLocks, 16u);

  AIESIM_CHECK_EQ(dev.numCoreDmaChannelsS2MM, 2u);
  AIESIM_CHECK_EQ(dev.numCoreDmaChannelsMM2S, 2u);
  AIESIM_CHECK_EQ(dev.numMemTileDmaChannelsS2MM, 6u);
  AIESIM_CHECK_EQ(dev.numMemTileDmaChannelsMM2S, 6u);
  AIESIM_CHECK_EQ(dev.numShimDmaChannelsS2MM, 2u);
  AIESIM_CHECK_EQ(dev.numShimDmaChannelsMM2S, 2u);

  AIESIM_CHECK_EQ(dev.numCoreBds, 16u);
  AIESIM_CHECK_EQ(dev.numMemTileBds, 48u);
  AIESIM_CHECK_EQ(dev.numShimBds, 16u);
}

} // namespace

int main() {
  DeviceModel npu1 = makeAIE2Device(4);
  DeviceModel npu2 = makeAIE2PDevice(8);

  // --- Round trip, both generations. ---
  checkRoundTrip(npu1, "AIE2/npu1");
  checkRoundTrip(npu2, "AIE2P/npu2");

  // --- Out-of-partition addresses decode to valid=false. ---
  checkOutOfPartition(npu1, "AIE2/npu1");
  checkOutOfPartition(npu2, "AIE2P/npu2");

  // --- tileTypeAt: Shim at row 0, MemTile rows, Core above, Invalid past
  // the last row. ---
  checkTileTypes(npu1);
  checkTileTypes(npu2);

  // --- Geometry sanity, per device, against the values grounded in
  // Device.cpp. A future edit that changes them should fail here. ---
  AIESIM_CHECK(npu1.generation == Generation::AIE2);
  AIESIM_CHECK_EQ(npu1.numCols, 4u);
  AIESIM_CHECK_EQ(npu1.numRows, 6u);
  AIESIM_CHECK_EQ(npu1.numMemTileRows, 1u);
  checkAIE2FamilyConstants(npu1);

  AIESIM_CHECK(npu2.generation == Generation::AIE2P);
  AIESIM_CHECK_EQ(npu2.numCols, 8u);
  AIESIM_CHECK_EQ(npu2.numRows, 6u);
  AIESIM_CHECK_EQ(npu2.numMemTileRows, 1u);
  checkAIE2FamilyConstants(npu2);

  // --- makeDeviceFromName: accepted names. ---
  struct NamedCase {
    const char *name;
    uint32_t numCols;
  };
  NamedCase accepted[] = {
      {"npu1", 4},      {"npu1_1col", 1}, {"npu1_2col", 2}, {"npu1_3col", 3},
      {"npu2", 8},      {"npu2_1col", 1}, {"npu2_2col", 2}, {"npu2_3col", 3},
      {"npu2_4col", 4}, {"npu2_5col", 5}, {"npu2_6col", 6}, {"npu2_7col", 7},
  };
  for (const NamedCase &nc : accepted) {
    DeviceModel dev;
    std::string error;
    bool ok = makeDeviceFromName(nc.name, dev, error);
    AIESIM_CHECK(ok);
    AIESIM_CHECK_EQ(dev.numCols, nc.numCols);
    if (!ok)
      std::fprintf(stderr, "  (makeDeviceFromName(\"%s\") failed: %s)\n",
                   nc.name, error.c_str());
  }

  // xcve2802: Versal AIE2, a different shape (38 cols x 11 rows, 2 memtile
  // rows) from the NPU1/NPU2 6-row/1-memtile-row layout, but the same
  // AIE2-family per-tile constants.
  {
    DeviceModel dev;
    std::string error;
    AIESIM_CHECK(makeDeviceFromName("xcve2802", dev, error));
    AIESIM_CHECK(dev.generation == Generation::AIE2);
    AIESIM_CHECK_EQ(dev.numCols, 38u);
    AIESIM_CHECK_EQ(dev.numRows, 11u);
    AIESIM_CHECK_EQ(dev.numMemTileRows, 2u);
    checkAIE2FamilyConstants(dev);
    checkTileTypes(dev);
    checkRoundTrip(dev, "xcve2802");
    checkOutOfPartition(dev, "xcve2802");
  }

  // --- makeDeviceFromName: rejections, with a non-empty error naming what
  // was accepted. ---
  const char *rejected[] = {
      "garbage",   "",
      "npu1_4col", // does not exist; npu1 itself is the 4-column array.
      "npu3",
      "xcvc1902", // real device, but not modeled by this simulator.
  };
  for (const char *name : rejected) {
    DeviceModel dev;
    std::string error;
    bool ok = makeDeviceFromName(name, dev, error);
    AIESIM_CHECK(!ok);
    AIESIM_CHECK(!error.empty());
  }

  return aiesim_test::summarize("device");
}
