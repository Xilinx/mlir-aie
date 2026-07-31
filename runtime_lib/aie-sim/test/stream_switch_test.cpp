//===- stream_switch_test.cpp -------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for the stream switch in lib/StreamSwitch.cpp. See that file's
// header comment for where every register offset and bit field comes from,
// and for what is and is not grounded in aie-rt.
//
// Device.cpp's makeAIE2PDevice is a separate, parallel task in this same
// shared worktree. By the time this was written it had gone from a
// `return {}` stub to a real implementation (test/device_test.cpp exercises
// it), but this test still builds its own DeviceModel by hand, following the
// convention test/lock_test.cpp already established: that keeps this test's
// obligations (a shim/memtile/core partition with two adjacent core rows)
// explicit at the call site and independent of further concurrent edits to
// a file this task does not own.
//
// The register offsets below are a small standalone table, not a reuse of
// StreamSwitch.cpp's private tables: this test target does not get the
// third_party/aie-rt private include path StreamSwitch.cpp builds against,
// so it could not include the vendored headers even if it wanted to. They
// are cited against the same aie-rt source StreamSwitch.cpp cites; keeping
// them independent means a mismatch between the two shows up as a failing
// test rather than being hidden by sharing one set of numbers.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include "TestSupport.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

DeviceModel makeTestDevice() {
  // One column, four rows: shim (0), one memtile row (1), and TWO core rows
  // (2 and 3) so the north/south inter-tile route has a real neighbour to
  // land on. Sizes/counts match the real AIE2P values Device.cpp grounds
  // (lib/Device.cpp's fillAIE2Family) for realism; none of them are read by
  // the stream switch itself.
  DeviceModel dev{};
  dev.generation = Generation::AIE2P;
  dev.numCols = 1;
  dev.numRows = 4;
  dev.numMemTileRows = 1;
  dev.coreDataMemSize = 64 * 1024;
  dev.coreProgMemSize = 16 * 1024;
  dev.memTileMemSize = 512 * 1024;
  dev.numCoreLocks = 16;
  dev.numMemTileLocks = 64;
  dev.numShimLocks = 16;
  return dev;
}

// Core-tile stream-switch register offsets, AIE2P. See StreamSwitch.cpp's
// file header for the full citation list (xaie2pgbl_params.h /
// xaie2pgbl_reginit.c); repeated here as plain numbers for the reason given
// in the file comment above. Only local (non-South/West/North/East)
// bundles are used to observe delivered words in tests below, other than
// the two tiles' North/South pair the cross-tile test exists to exercise:
// South/West/North/East master ports get auto-drained to a neighbour tile
// every cycle by the switch's own inter-tile wiring, so they are not stable
// places to read a word back from inside a single tile.
constexpr uint32_t kMstrCtrl = 0x0003F00C;   // MASTER_CONFIG_TILE_CTRL
constexpr uint32_t kMstrFifo0 = 0x0003F010;  // MASTER_CONFIG_FIFO0
constexpr uint32_t kMstrNorth0 = 0x0003F034; // MASTER_CONFIG_NORTH0
constexpr uint32_t kSlvDma0 = 0x0003F104;    // SLAVE_CONFIG_DMA_0
constexpr uint32_t kSlvSouth0 = 0x0003F114;  // SLAVE_CONFIG_SOUTH_0
constexpr uint32_t kSlotDma0Slot0 = 0x0003F210; // SLAVE_DMA_0_SLOT0, slot 0
constexpr uint32_t kSlotDma0Slot1 = 0x0003F214; // ... slot 1 (+SlotOffset)
constexpr uint32_t kSlvConfigBase = 0x0003F100; // SlvConfigBaseAddr

// _XAie_GetSlaveIdx's own formula
// (third_party/aie-rt/driver/src/common/xaie_helper.c:246-268): the
// physical slave index a circuit-mode master's Configuration field names is
// (regOff - SlvConfigBaseAddr) / 4.
uint32_t physIdx(uint32_t slaveRegOff) {
  return (slaveRegOff - kSlvConfigBase) / 4;
}

// Register-value packers matching StreamSwitch.cpp's bit layout
// (xaie2pgbl_params.h:3630-3648 master, :4090-4100 slave, :4390-4412 slot).
uint32_t packMasterCircuit(uint32_t slaveIdx) {
  return (1u << 31) | (slaveIdx & 0x7Fu); // MASTER_ENABLE=1, PACKET_ENABLE=0
}
uint32_t packMasterPacket(uint32_t arbitor, uint32_t mselEnMask) {
  return (1u << 31) | (1u << 30) | ((mselEnMask & 0xFu) << 3) |
        (arbitor & 0x7u);
}
uint32_t packSlaveEnable(bool packetMode) {
  return (1u << 31) | (packetMode ? (1u << 30) : 0u);
}
uint32_t packSlot(uint32_t id, uint32_t mask, uint32_t msel,
                  uint32_t arbitor) {
  return (1u << 8) | ((id & 0x1Fu) << 24) | ((mask & 0x1Fu) << 16) |
        ((msel & 0x3u) << 4) | (arbitor & 0x7u);
}

//===----------------------------------------------------------------------===//

void testUnconfiguredMasterPassesNothing() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;
  StreamSwitchModule *sw = core->streamSwitch();
  AIESIM_CHECK(sw != nullptr);
  if (!sw)
    return;

  // Push into a slave port whose master side is never configured: no
  // register write at all touches kMstrCtrl or kMstrNorth0.
  StreamPort *dmaSlave = sw->slavePort(PortBundle::DMA, 0);
  AIESIM_CHECK(dmaSlave != nullptr);
  dmaSlave->push(0xAAAA, true);

  array.advance(20);

  AIESIM_CHECK(!sw->masterPort(PortBundle::Ctrl, 0)->canPop());
  AIESIM_CHECK(!sw->masterPort(PortBundle::North, 0)->canPop());
  AIESIM_CHECK(diag.empty());
}

void testCircuitSwitchedConnection() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;
  RegisterFile &regs = core->regs();

  // Real register writes: DMA slave port 0 enabled (circuit mode), Ctrl
  // master port 0 fed by it (Configuration = DMA0's physical slave index).
  regs.write(kSlvDma0, packSlaveEnable(/*packetMode=*/false));
  regs.write(kMstrCtrl, packMasterCircuit(physIdx(kSlvDma0)));

  StreamSwitchModule *sw = core->streamSwitch();
  StreamPort *slave = sw->slavePort(PortBundle::DMA, 0);
  StreamPort *master = sw->masterPort(PortBundle::Ctrl, 0);

  slave->push(0x1111, /*tlast=*/false);
  slave->push(0x2222, /*tlast=*/false);
  slave->push(0x3333, /*tlast=*/true);

  // Drain interleaved with stepping rather than assuming a FIFO depth: the
  // switch moves at most one word per connection per cycle, and the master
  // FIFO is deliberately small, so a word can sit in the source until the
  // destination is popped.
  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  for (int cyc = 0; cyc < 20 && words.size() < 3; ++cyc) {
    array.advance(1);
    while (master->canPop()) {
      uint32_t w;
      bool tl;
      master->pop(w, tl);
      words.push_back(w);
      tlasts.push_back(tl);
    }
  }

  AIESIM_CHECK_EQ(words.size(), size_t{3});
  if (words.size() == 3) {
    AIESIM_CHECK_EQ(words[0], 0x1111u);
    AIESIM_CHECK_EQ(words[1], 0x2222u);
    AIESIM_CHECK_EQ(words[2], 0x3333u);
    AIESIM_CHECK(!tlasts[0]);
    AIESIM_CHECK(!tlasts[1]);
    AIESIM_CHECK(tlasts[2]);
  }
  AIESIM_CHECK(diag.empty());
}

void testTwoTileRoute() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *a = array.tile(0, 2); // Core.
  Tile *b = array.tile(0, 3); // Core, the tile north of a.
  AIESIM_CHECK(a != nullptr);
  AIESIM_CHECK(b != nullptr);
  if (!a || !b)
    return;

  // A: DMA slave 0 (source) -> North master 0 (circuit-switched).
  a->regs().write(kSlvDma0, packSlaveEnable(/*packetMode=*/false));
  a->regs().write(kMstrNorth0, packMasterCircuit(physIdx(kSlvDma0)));

  // B: South slave 0 -- the far side of A's North master, per
  // Components.h's installStreamSwitch doc comment -- routed on to Ctrl
  // master 0 so the word is observable from a local (non-wired) port.
  b->regs().write(kSlvSouth0, packSlaveEnable(/*packetMode=*/false));
  b->regs().write(kMstrCtrl, packMasterCircuit(physIdx(kSlvSouth0)));

  StreamPort *src = a->streamSwitch()->slavePort(PortBundle::DMA, 0);
  StreamPort *dst = b->streamSwitch()->masterPort(PortBundle::Ctrl, 0);

  src->push(0xBEEF, /*tlast=*/true);

  bool delivered = false;
  uint32_t word = 0;
  bool tlast = false;
  for (int cyc = 0; cyc < 20 && !delivered; ++cyc) {
    array.advance(1);
    if (dst->canPop()) {
      dst->pop(word, tlast);
      delivered = true;
    }
  }

  AIESIM_CHECK(delivered);
  AIESIM_CHECK_EQ(word, 0xBEEFu);
  AIESIM_CHECK(tlast);
  AIESIM_CHECK(diag.empty());
}

// Packet switching: two single-word packets, distinguished only by the
// header's packet id (our own header-word convention -- see
// StreamSwitch.cpp's file header -- bits [4:0]), demultiplexed by slot
// rules to two different local master ports.
void testPacketSwitchDemux() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;
  RegisterFile &regs = core->regs();

  // DMA slave 0 in packet mode, with two slot rules: id 1 selects
  // (arbiter 0, msel 0), id 2 selects (arbiter 0, msel 1).
  regs.write(kSlvDma0, packSlaveEnable(/*packetMode=*/true));
  regs.write(kSlotDma0Slot0,
            packSlot(/*id=*/1, /*mask=*/0, /*msel=*/0, /*arbitor=*/0));
  regs.write(kSlotDma0Slot1,
            packSlot(/*id=*/2, /*mask=*/0, /*msel=*/1, /*arbitor=*/0));

  // Ctrl listens only to msel 0; FIFO listens only to msel 1. Both are
  // local bundles, so nothing drains them but this test.
  regs.write(kMstrCtrl, packMasterPacket(/*arbitor=*/0, /*mselEnMask=*/0x1));
  regs.write(kMstrFifo0, packMasterPacket(/*arbitor=*/0, /*mselEnMask=*/0x2));

  StreamSwitchModule *sw = core->streamSwitch();
  StreamPort *slave = sw->slavePort(PortBundle::DMA, 0);
  StreamPort *ctrlMaster = sw->masterPort(PortBundle::Ctrl, 0);
  StreamPort *fifoMaster = sw->masterPort(PortBundle::FIFO, 0);

  // Packet id 1: single-word packet (the header word doubles as the only
  // payload word here), expected at Ctrl.
  slave->push(/*header, id=1*/ 1u, /*tlast=*/true);
  bool got1 = false;
  uint32_t w1 = 0;
  bool tl1 = false;
  for (int cyc = 0; cyc < 10 && !got1; ++cyc) {
    array.advance(1);
    if (ctrlMaster->canPop()) {
      ctrlMaster->pop(w1, tl1);
      got1 = true;
    }
  }
  AIESIM_CHECK(got1);
  AIESIM_CHECK_EQ(w1, 1u);
  AIESIM_CHECK(tl1);
  AIESIM_CHECK(!fifoMaster->canPop());

  // Packet id 2: expected at FIFO instead, never at Ctrl.
  slave->push(/*header, id=2*/ 2u, /*tlast=*/true);
  bool got2 = false;
  uint32_t w2 = 0;
  bool tl2 = false;
  for (int cyc = 0; cyc < 10 && !got2; ++cyc) {
    array.advance(1);
    if (fifoMaster->canPop()) {
      fifoMaster->pop(w2, tl2);
      got2 = true;
    }
  }
  AIESIM_CHECK(got2);
  AIESIM_CHECK_EQ(w2, 2u);
  AIESIM_CHECK(tl2);
  AIESIM_CHECK(!ctrlMaster->canPop());

  AIESIM_CHECK(diag.empty());
}

} // namespace

int main() {
  testUnconfiguredMasterPassesNothing();
  testCircuitSwitchedConnection();
  testTwoTileRoute();
  testPacketSwitchDemux();
  return aiesim_test::summarize("stream_switch");
}
