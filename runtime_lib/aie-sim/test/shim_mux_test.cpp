//===- shim_mux_test.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the shim PL-interface mux (lib/ShimMux.cpp): the two
// registers that decide whether a shim tile's south stream-switch ports face
// the PL, the shim DMA or the NoC.
//
// The field placement is the part worth pinning. Both registers pack four
// two-bit fields, but they cover DIFFERENT port sets ({2,3,6,7} for the mux,
// {2,3,4,5} for the demux) and start at different LSBs, so a decoder that
// gets one right by construction can still get the other wrong -- and would
// do so silently, since every field's reset value is the same 0.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include <string>

using namespace aiesim;

namespace {

struct ErrorCapture {
  std::string message;
  DiagnosticHandler handler() {
    return [this](const std::string &m) { message = m; };
  }
};

DeviceModel makeTestDevice() {
  // One column, shim plus a memtile and a core row. Only row 0 matters here;
  // the others exist so the mux is tested on a tile that is a shim BECAUSE of
  // its row, the same way a real device decides.
  DeviceModel dev{};
  dev.generation = Generation::AIE2P;
  dev.numCols = 1;
  dev.numRows = 3;
  dev.numMemTileRows = 1;
  dev.coreDataMemSize = 64 * 1024;
  dev.coreProgMemSize = 16 * 1024;
  dev.memTileMemSize = 512 * 1024;
  dev.numCoreLocks = 16;
  dev.numMemTileLocks = 64;
  dev.numShimLocks = 16;
  return dev;
}

// xaie2pgbl_params.h:19155,19175.
constexpr uint32_t kMuxConfig = 0x0001F000;
constexpr uint32_t kDemuxConfig = 0x0001F004;

// Encoding, xaie_plif.c:37-39.
constexpr uint32_t kPL = 0, kDMA = 1, kNoC = 2;

//===----------------------------------------------------------------------===//
// The two field maps, written out as the registers document them rather than
// computed, so this test disagrees with ShimMux.cpp if either drifts.
// MUX_CONFIG_SOUTH{2,3,6,7}_LSB = 8,10,12,14 (xaie2pgbl_params.h:19158-19172);
// DEMUX_CONFIG_SOUTH{2,3,4,5}_LSB = 4,6,8,10 (:19178-19192).
//===----------------------------------------------------------------------===//

struct Field {
  uint32_t port;
  uint32_t lsb;
};
constexpr Field kMuxFields[] = {{2, 8}, {3, 10}, {6, 12}, {7, 14}};
constexpr Field kDemuxFields[] = {{2, 4}, {3, 6}, {4, 8}, {5, 10}};

//===----------------------------------------------------------------------===//
// Test 1: every field of both registers decodes at its own position, and only
// there. Each field is set in isolation so a decoder reading a neighbour's
// bits fails rather than happening to agree.
//===----------------------------------------------------------------------===//

void testEachFieldDecodesAtItsOwnPosition() {
  Array array(makeTestDevice(), nullptr);
  Tile *shim = array.tile(0, 0);
  AIESIM_CHECK(shim != nullptr);
  ShimMuxModule *mux = shim->shimMux();
  AIESIM_CHECK(mux != nullptr);
  if (!mux)
    return;

  for (const Field &f : kMuxFields) {
    shim->regs().write(kMuxConfig, kDMA << f.lsb);
    AIESIM_CHECK(mux->slaveEndpoint(f.port) == ShimPortEndpoint::ShimDma);
    for (const Field &other : kMuxFields)
      if (other.port != f.port)
        AIESIM_CHECK(mux->slaveEndpoint(other.port) == ShimPortEndpoint::PL);
  }

  for (const Field &f : kDemuxFields) {
    shim->regs().write(kDemuxConfig, kNoC << f.lsb);
    AIESIM_CHECK(mux->masterEndpoint(f.port) == ShimPortEndpoint::NoC);
    for (const Field &other : kDemuxFields)
      if (other.port != f.port)
        AIESIM_CHECK(mux->masterEndpoint(other.port) == ShimPortEndpoint::PL);
  }
}

//===----------------------------------------------------------------------===//
// Test 2: the two registers are independent. The demux covers south 4 and 5
// where the mux covers 6 and 7, so a decoder that shared one field table
// between them would answer for a port the other register never mentions.
//===----------------------------------------------------------------------===//

void testMuxAndDemuxAreSeparateRegisters() {
  Array array(makeTestDevice(), nullptr);
  Tile *shim = array.tile(0, 0);
  ShimMuxModule *mux = shim->shimMux();
  AIESIM_CHECK(mux != nullptr);
  if (!mux)
    return;

  // Mux: south 7 -> DMA. Demux: south 5 -> NoC. Neither port appears in the
  // other register's field set.
  shim->regs().write(kMuxConfig, kDMA << 14);
  shim->regs().write(kDemuxConfig, kNoC << 10);

  AIESIM_CHECK(mux->slaveEndpoint(7) == ShimPortEndpoint::ShimDma);
  AIESIM_CHECK(mux->masterEndpoint(5) == ShimPortEndpoint::NoC);
  // Same index, other direction: south MASTER 7 is not a port the demux
  // steers, and south SLAVE 5 is not one the mux steers.
  AIESIM_CHECK(mux->masterEndpoint(7) == ShimPortEndpoint::PL);
  AIESIM_CHECK(mux->slaveEndpoint(5) == ShimPortEndpoint::PL);
}

//===----------------------------------------------------------------------===//
// Test 3: both registers read back. aie-rt configures them with
// XAie_MaskWrite32 (xaie_plif.c:641,700), which READS first, so a claim that
// only handled writes would fault on the read -- and an unclaimed read is the
// fatal direction (Array.h). This is also what makes a second field settable
// without clearing the first, which the read-modify-write below checks.
//===----------------------------------------------------------------------===//

void testRegistersReadBackForMaskWrite() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *shim = array.tile(0, 0);
  ShimMuxModule *mux = shim->shimMux();
  AIESIM_CHECK(mux != nullptr);
  if (!mux)
    return;

  shim->regs().write(kMuxConfig, kDMA << 10); // south 3 -> DMA
  AIESIM_CHECK_EQ(shim->regs().read(kMuxConfig), kDMA << 10);
  AIESIM_CHECK(err.message.empty());

  // The read-modify-write aie-rt would do to add south 7 without disturbing
  // south 3.
  uint32_t v = shim->regs().read(kMuxConfig);
  v = (v & ~(0x3u << 14)) | (kDMA << 14);
  shim->regs().write(kMuxConfig, v);

  AIESIM_CHECK(mux->slaveEndpoint(3) == ShimPortEndpoint::ShimDma);
  AIESIM_CHECK(mux->slaveEndpoint(7) == ShimPortEndpoint::ShimDma);
  AIESIM_CHECK(err.message.empty());
}

//===----------------------------------------------------------------------===//
// Test 4: reset state is PL everywhere, including ports neither register
// covers. Every *_DEFVAL in both registers is 0x0, and
// XAie_EnablePlToAieStrmPort's note says a device reset leaves AIE<->PL
// enabled -- so 0 means PL rather than "unconfigured".
//===----------------------------------------------------------------------===//

void testResetStateIsPl() {
  Array array(makeTestDevice(), nullptr);
  Tile *shim = array.tile(0, 0);
  ShimMuxModule *mux = shim->shimMux();
  AIESIM_CHECK(mux != nullptr);
  if (!mux)
    return;

  for (uint32_t port = 0; port < 8; ++port)
    AIESIM_CHECK(mux->slaveEndpoint(port) == ShimPortEndpoint::PL);
  for (uint32_t port = 0; port < 6; ++port)
    AIESIM_CHECK(mux->masterEndpoint(port) == ShimPortEndpoint::PL);
  AIESIM_CHECK_EQ(static_cast<uint32_t>(ShimPortEndpoint::PL), kPL);
}

//===----------------------------------------------------------------------===//
// Test 5: the channel-to-port map. aie-rt fixes WHICH two ports each
// direction may use and never names a channel, so this pins the assignment
// mlir-aie's lowering makes -- the one the designs we replay are programmed
// with. See shimDmaSouthPort() for the citations.
//===----------------------------------------------------------------------===//

void testChannelToSouthPortMap() {
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::MM2S, 0), 3);
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::MM2S, 1), 7);
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::S2MM, 0), 2);
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::S2MM, 1), 3);

  // A shim has two channels per direction; anything above that has no port.
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::MM2S, 2), -1);
  AIESIM_CHECK_EQ(shimDmaSouthPort(DmaDirection::S2MM, 2), -1);

  // Every port the map hands out must be one its register can actually steer
  // to the DMA, or the mux would never report ShimDma for it.
  for (uint32_t ch = 0; ch < 2; ++ch) {
    int mm2s = shimDmaSouthPort(DmaDirection::MM2S, ch);
    AIESIM_CHECK(mm2s == 3 || mm2s == 7); // XAie_EnableShimDmaToAieStrmPort
    int s2mm = shimDmaSouthPort(DmaDirection::S2MM, ch);
    AIESIM_CHECK(s2mm == 2 || s2mm == 3); // XAie_EnableAieToShimDmaStrmPort
  }
}

//===----------------------------------------------------------------------===//
// Test 6: only a shim tile has one. A core or memtile has no PL interface,
// and 0x1F000 is a different register entirely there (the memory module's
// Lock0_Value), so installing a mux on one would claim a range another
// component owns.
//===----------------------------------------------------------------------===//

void testOnlyShimTilesHaveAMux() {
  Array array(makeTestDevice(), nullptr);
  AIESIM_CHECK(array.tile(0, 0)->shimMux() != nullptr);  // shim
  AIESIM_CHECK(array.tile(0, 1)->shimMux() == nullptr);  // memtile
  AIESIM_CHECK(array.tile(0, 2)->shimMux() == nullptr);  // core
}

} // namespace

int main() {
  testEachFieldDecodesAtItsOwnPosition();
  testMuxAndDemuxAreSeparateRegisters();
  testRegistersReadBackForMaskWrite();
  testResetStateIsPl();
  testChannelToSouthPortMap();
  testOnlyShimTilesHaveAMux();
  return aiesim_test::summarize("shim_mux");
}
