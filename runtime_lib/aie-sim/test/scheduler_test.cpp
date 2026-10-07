//===- scheduler_test.cpp -------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for the active-set scheduler in lib/Array.cpp (Array::wake(),
// Array::advance(), Array::runUntilQuiescent(), Steppable::busy()). Register
// offsets/packers are independently duplicated from stream_switch_test.cpp
// and dma_test.cpp rather than shared, following this directory's existing
// convention (see those files' own header comments for why): a mismatch
// between independent copies fails a test instead of hiding behind one
// shared table.
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

// One column: shim (0), one memtile row (1), two core rows (2, 3), matching
// stream_switch_test.cpp's makeTestDevice so a North/South hop is available.
DeviceModel makeTestDevice() {
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

// Core-tile stream-switch registers, AIE2P (see StreamSwitch.cpp's file
// header for the grounding citations).
constexpr uint32_t kMstrCtrl = 0x0003F00C;
constexpr uint32_t kMstrNorth0 = 0x0003F034;
constexpr uint32_t kSlvDma0 = 0x0003F104;
constexpr uint32_t kSlvSouth0 = 0x0003F114;
constexpr uint32_t kSlvConfigBase = 0x0003F100;

uint32_t physIdx(uint32_t slaveRegOff) {
  return (slaveRegOff - kSlvConfigBase) / 4;
}
uint32_t packMasterCircuit(uint32_t slaveIdx) {
  return (1u << 31) | (slaveIdx & 0x7Fu);
}
uint32_t packSlaveEnable() { return 1u << 31; }

// Core-tile DMA registers, AIE2P (see Dma.cpp's file header). Only the
// fields testDeterministicRerunMatches needs: a single-dim MM2S BD with an
// optional lock gate, and the queue/status registers to start and observe
// channel 0.
constexpr uint32_t kBdBase = 0x1D000;
constexpr uint32_t kBdStride = 0x20;
constexpr uint32_t kCtrlBase = 0x1DE00;
constexpr uint32_t kCtrlStride = 0x8;
constexpr uint32_t kNumCh = 2;
constexpr uint32_t kStatusBase = 0x1DF00;
constexpr uint32_t kStatusDirStride = 0x10;

// Core-tile lock registers, AIE2P (see Lock.cpp's file header).
constexpr uint32_t kLockValueBase = 0x0001F000;
constexpr uint32_t kLockValueStride = 0x10;

uint32_t packBdWord0(uint32_t lengthWords, uint32_t addrWordOffset) {
  return (lengthWords & 0x3FFFu) | ((addrWordOffset & 0x3FFFu) << 14);
}
uint32_t packBdWord5(uint8_t lockAcqId, int32_t lockAcqVal, bool lockAcqEn,
                     uint8_t lockRelId, int32_t lockRelVal) {
  return (static_cast<uint32_t>(lockAcqId) & 0xFu) |
        ((static_cast<uint32_t>(lockAcqVal) & 0x7Fu) << 5) |
        ((lockAcqEn ? 1u : 0u) << 12) |
        ((static_cast<uint32_t>(lockRelId) & 0xFu) << 13) |
        ((static_cast<uint32_t>(lockRelVal) & 0x7Fu) << 18) |
        (1u << 25); // ValidBd.
}
uint32_t mm2sQueueOff(uint32_t ch) {
  return kCtrlBase + ch * kCtrlStride + kCtrlStride * kNumCh + 4;
}
uint32_t mm2sStatusOff(uint32_t ch) {
  return kStatusBase + ch * 4 + kStatusDirStride;
}

//===----------------------------------------------------------------------===//
// Finding 1: an idle array must cost O(1) per advance() call. A regression
// to the old "step every component every cycle" behaviour would make this
// loop run for a very long time instead of returning immediately.
//===----------------------------------------------------------------------===//

void testIdleArrayAdvanceLandsOnExactCycleCount() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  AIESIM_CHECK_EQ(array.cycle(), uint64_t{0});

  // Nothing was ever configured, so nothing is in the active set; this must
  // jump the counter rather than loop it.
  const uint64_t big = 10'000'000'000ull;
  array.advance(big);
  AIESIM_CHECK_EQ(array.cycle(), big);

  array.advance(1);
  AIESIM_CHECK_EQ(array.cycle(), big + 1);
}

//===----------------------------------------------------------------------===//
// Finding 2: a configured route still delivers in exactly the cycle count
// the two-phase step()/commit() model implies -- one cycle per tile hop, per
// StreamSwitch.cpp's commit() comment. Two tiles, one hop: cycle 1 moves the
// word into tile b's slave FIFO (staged, published only at commit()); cycle
// 2 is the first cycle b's own step() can see it and relay it on to the
// local Ctrl sink.
//===----------------------------------------------------------------------===//

void testConfiguredRouteExactCycleCount() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *a = array.tile(0, 2);
  Tile *b = array.tile(0, 3);
  AIESIM_CHECK(a != nullptr);
  AIESIM_CHECK(b != nullptr);
  if (!a || !b)
    return;

  a->regs().write(kSlvDma0, packSlaveEnable());
  a->regs().write(kMstrNorth0, packMasterCircuit(physIdx(kSlvDma0)));
  b->regs().write(kSlvSouth0, packSlaveEnable());
  b->regs().write(kMstrCtrl, packMasterCircuit(physIdx(kSlvSouth0)));

  StreamPort *src = a->streamSwitch()->slavePort(PortBundle::DMA, 0);
  StreamPort *dst = b->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  src->push(0xCAFEu, /*tlast=*/true);

  int delivered = -1;
  for (int cyc = 1; cyc <= 20 && delivered < 0; ++cyc) {
    array.advance(1);
    if (dst->canPop())
      delivered = cyc;
  }

  AIESIM_CHECK_EQ(delivered, 2);
  AIESIM_CHECK(diag.empty());
  if (dst->canPop()) {
    uint32_t word;
    bool tlast;
    dst->pop(word, tlast);
    AIESIM_CHECK_EQ(word, 0xCAFEu);
    AIESIM_CHECK(tlast);
  }
}

//===----------------------------------------------------------------------===//
// Finding 3: determinism. A lock-gated MM2S transfer (touches locks, DMA and
// the stream switch, so all three wake()/busy() paths this change added are
// exercised) run identically on two fresh arrays must produce the same
// cycle count and the same final observable state.
//===----------------------------------------------------------------------===//

struct ScriptResult {
  uint64_t cycles;
  std::vector<uint32_t> delivered;
  uint32_t completedBds;
  uint32_t lockValueAfterRelease;
};

ScriptResult runScript() {
  DeviceModel dev = makeTestDevice();
  Array array(dev, nullptr);
  std::vector<std::string> diag;
  array.setDiagnosticHandler([&](const std::string &m) { diag.push_back(m); });

  Tile *core = array.tile(0, 2);
  uint32_t src[4] = {10, 20, 30, 40};
  core->memory()->write(0, src, sizeof(src));

  // Wire DMA0's MM2S output to the local Ctrl bundle, same shape as
  // dma_test.cpp's wireMm2sToCtrl.
  core->regs().write(kSlvDma0, packSlaveEnable());
  core->regs().write(kMstrCtrl, packMasterCircuit(physIdx(kSlvDma0)));

  // BD 0: 4 words from offset 0, gated on lock 0 == 1 (not armed yet),
  // releasing lock 1 by 1 on completion.
  uint32_t base = kBdBase + 0 * kBdStride;
  core->regs().write(base + 0, packBdWord0(/*lengthWords=*/4,
                                           /*addrWordOffset=*/0));
  core->regs().write(base + 20,
                     packBdWord5(/*lockAcqId=*/0, /*lockAcqVal=*/1,
                                /*lockAcqEn=*/true, /*lockRelId=*/1,
                                /*lockRelVal=*/1));

  core->regs().write(mm2sQueueOff(0), /*startBd=*/0);

  // A handful of cycles stalled on the lock: exercises the DMA's own
  // busy()-driven retry, not a wake().
  array.advance(5);

  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  AIESIM_CHECK(!sink->canPop());
  uint32_t stalledStatus = core->regs().read(mm2sStatusOff(0));
  AIESIM_CHECK((stalledStatus & 0x4u) != 0); // StalledLockAcq.

  // Arms the lock through a plain register write: this is the wake() this
  // change added to Lock.cpp's value-register handler, and it is the DMA's
  // own persistent busy() (not that wake()) that actually resumes the
  // channel -- see Lock.cpp/Dma.cpp's wake() comments.
  core->regs().write(kLockValueBase + 0 * kLockValueStride, 1);

  std::vector<uint32_t> delivered;
  for (int cyc = 0; cyc < 200 && delivered.size() < 4; ++cyc) {
    array.advance(1);
    while (sink->canPop()) {
      uint32_t word;
      bool tlast;
      sink->pop(word, tlast);
      delivered.push_back(word);
      (void)tlast;
    }
  }

  AIESIM_CHECK(array.runUntilQuiescent(200));
  AIESIM_CHECK(diag.empty());

  uint32_t lock1 = core->regs().read(kLockValueBase + 1 * kLockValueStride) &
                   0x3Fu;
  return {array.cycle(), delivered,
         core->dma()->completedBds(DmaDirection::MM2S, 0), lock1};
}

void testDeterministicRerunMatches() {
  ScriptResult first = runScript();
  ScriptResult second = runScript();

  AIESIM_CHECK_EQ(first.cycles, second.cycles);
  AIESIM_CHECK_EQ(first.completedBds, uint32_t{1});
  AIESIM_CHECK_EQ(first.completedBds, second.completedBds);
  AIESIM_CHECK_EQ(first.lockValueAfterRelease, uint32_t{1});
  AIESIM_CHECK_EQ(first.lockValueAfterRelease, second.lockValueAfterRelease);

  AIESIM_CHECK_EQ(first.delivered.size(), size_t{4});
  AIESIM_CHECK_EQ(first.delivered.size(), second.delivered.size());
  for (size_t i = 0; i < first.delivered.size() && i < second.delivered.size();
      ++i)
    AIESIM_CHECK_EQ(first.delivered[i], second.delivered[i]);
}

} // namespace

int main() {
  testIdleArrayAdvanceLandsOnExactCycleCount();
  testConfiguredRouteExactCycleCount();
  testDeterministicRerunMatches();
  return aiesim_test::summarize("scheduler");
}
