//===- dma_test.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the DMA module (lib/Dma.cpp), runnable with no aie-rt call,
// no aiecc, and no NPU.
//
// LIMITATIONS -- read before extending this file:
//
//  * Device.cpp (makeAIE2PDevice / decodeAddress) is a placeholder as of
//    this writing: makeAIE2PDevice(...) returns a zero-initialised
//    DeviceModel, and decodeAddress always reports "invalid" (see
//    lib/Device.cpp). So this file builds a DeviceModel by hand instead
//    (matching the real npu2/AIE2P numbers Dma.cpp is grounded against --
//    see that file's header comment for citations), and never calls
//    Array::write32/read32. It goes directly at Tile::regs() with the
//    same tile-relative register offsets Dma.cpp itself decodes.
//    Array::advance()/runUntilQuiescent() do not depend on decodeAddress,
//    so cycle-stepping works normally.
//
//  * Lock.cpp and StreamSwitch.cpp are ALSO placeholders as of this
//    writing (installLocks/installStreamSwitch do nothing), so
//    tile.locks() and tile.streamSwitch() are both null for every tile
//    built here. This file exercises what Dma.cpp itself owns by using
//    the test-only hooks dmaTestSetPort/dmaTestSetLocks (defined in
//    lib/Dma.cpp, forward-declared again below -- the two files share no
//    header), which attach a hand-written StreamPort/LockModule to a
//    channel directly, standing in for the stream switch and lock module
//    until those land.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include <deque>
#include <string>

using namespace aiesim;

// Defined in lib/Dma.cpp. Not shared through a header: this file and
// lib/Dma.cpp are the only two files this change owns, and Components.h is
// the contract header, not ours to extend.
namespace aiesim {
void dmaTestSetPort(DmaModule &dma, DmaDirection dir, uint32_t channel,
                    StreamPort *port);
void dmaTestSetLocks(DmaModule &dma, LockModule *locks);
} // namespace aiesim

namespace {

//===----------------------------------------------------------------------===//
// A minimal AIE2P (npu2) DeviceModel, built by hand (see the file
// comment). One column: row 0 shim, row 1 memtile, rows 2-3 core.
//
// Numbers: colShift/rowShift from AIETargetModel.h:738-739 (the AIE2
// class, shared by npu1 and npu2); coreDataMemSize/memTileMemSize from
// AIETargetModel.h:640 (getLocalMemorySize) and :711 (getMemTileSize);
// lock/BD counts from :645 (getNumLocks) and :657 (getNumBDs). None of
// colShift/rowShift/baseAddr actually matter to these tests -- they would
// only matter to Array::write32/read32 via decodeAddress, which this file
// does not use -- they are filled in for documentation only.
//===----------------------------------------------------------------------===//

DeviceModel makeTestDevice() {
  DeviceModel dev{};
  dev.generation = Generation::AIE2P;
  dev.numCols = 1;
  dev.numRows = 4;
  dev.numMemTileRows = 1;
  dev.colShift = 25;
  dev.rowShift = 20;
  dev.baseAddr = 0;
  dev.coreDataMemSize = 0x10000;
  dev.coreProgMemSize = 0x4000;
  dev.memTileMemSize = 0x80000;
  dev.progMemHostOffset = 0;
  dev.numCoreLocks = 16;
  dev.numMemTileLocks = 64;
  dev.numShimLocks = 16;
  dev.numCoreDmaChannelsS2MM = 2;
  dev.numCoreDmaChannelsMM2S = 2;
  dev.numMemTileDmaChannelsS2MM = 6;
  dev.numMemTileDmaChannelsMM2S = 6;
  dev.numShimDmaChannelsS2MM = 2;
  dev.numShimDmaChannelsMM2S = 2;
  dev.numCoreBds = 16;
  dev.numMemTileBds = 48;
  dev.numShimBds = 16;
  return dev;
}

// Register offsets, mirrored from lib/Dma.cpp's kCoreLayout/kShimLayout
// (see that file for the grounding citations for every one of these
// numbers). Not shared through a header for the same reason as the
// dmaTestSetPort/dmaTestSetLocks declarations above.
namespace CoreRegs {
constexpr uint32_t kBdBase = 0x1D000;
constexpr uint32_t kBdStride = 0x20;
constexpr uint32_t kCtrlBase = 0x1DE00;
constexpr uint32_t kCtrlStride = 0x8;
constexpr uint32_t kNumCh = 2;
constexpr uint32_t kStatusBase = 0x1DF00;
constexpr uint32_t kStatusDirStride = 0x10;
} // namespace CoreRegs

namespace ShimRegs {
constexpr uint32_t kBdBase = 0x1D000;
constexpr uint32_t kBdStride = 0x20;
constexpr uint32_t kCtrlBase = 0x1D200;
constexpr uint32_t kCtrlStride = 0x8;
constexpr uint32_t kNumCh = 2;
} // namespace ShimRegs

uint32_t ctrlQueueOff(uint32_t ctrlBase, uint32_t ctrlStride, uint32_t numCh,
                     DmaDirection dir, uint32_t ch) {
  uint32_t dirOff = dir == DmaDirection::MM2S ? ctrlStride * numCh : 0;
  return ctrlBase + ch * ctrlStride + dirOff + 4; // +4: the start-queue
                                                  // word follows the ctrl
                                                  // word (Dma.cpp).
}

uint32_t statusOff(uint32_t statusBase, uint32_t statusDirStride,
                   DmaDirection dir, uint32_t ch) {
  uint32_t dirOff = dir == DmaDirection::MM2S ? statusDirStride : 0;
  return statusBase + ch * 4 + dirOff;
}

//===----------------------------------------------------------------------===//
// Test doubles for the two components this file does not own.
//===----------------------------------------------------------------------===//

// Stands in for a stream-switch port: an unbounded FIFO so MM2S never has
// to stall on backpressure in these tests (the stall path itself is
// exercised through the lock, not the stream, below).
class FifoPort : public StreamPort {
public:
  bool canPush() const override { return true; }
  void push(uint32_t word, bool tlast) override {
    words.push_back(word);
    tlasts.push_back(tlast);
  }
  bool canPop() const override { return !words.empty(); }
  void pop(uint32_t &word, bool &tlast) override {
    word = words.front();
    words.pop_front();
    tlast = tlasts.front();
    tlasts.pop_front();
  }

  std::deque<uint32_t> words;
  std::deque<bool> tlasts;
};

// Lock id 0 gates on `armed`; every other id always succeeds. Lets a test
// hold a channel stalled and then let it through.
class FakeLockModule : public LockModule {
public:
  bool step() override { return false; }
  bool tryAcquire(uint32_t id, int32_t value) override {
    (void)value;
    acquireAttempts++;
    return id == 0 ? armed : true;
  }
  void release(uint32_t id, int32_t value) override {
    (void)id;
    (void)value;
    ++releaseCount;
  }
  int32_t value(uint32_t) const override { return 0; }
  uint32_t count() const override { return 1; }

  bool armed = false;
  int releaseCount = 0;
  int acquireAttempts = 0;
};

//===----------------------------------------------------------------------===//
// BD register writers. Bit positions mirror lib/Dma.cpp's kCoreLayout /
// kShimLayout exactly; see that file for citations. All defaults (packet
// mode off, iteration dimension untouched, tlast not suppressed) leave
// those fields at the harmless "not used" value.
//===----------------------------------------------------------------------===//

struct BdSpec {
  uint32_t addrWordOffset = 0; // Core/memtile: word offset into memory.
  uint64_t shimByteAddr = 0;   // Shim only.
  uint32_t lengthWords = 0;
  // Raw register encodings: pass N-1 for step sizes, N as-is for wraps.
  // {0,0,0} is "unconfigured" (StepSize=1, Wrap=0) for every dim.
  uint32_t stepSizeMinus1[3] = {0, 0, 0};
  uint32_t wrap[3] = {0, 0, 0};
  uint8_t nextBd = 0;
  bool useNextBd = false;
  bool validBd = true;
  bool lockAcqEn = false;
  uint8_t lockAcqId = 0;
  int32_t lockAcqVal = 1;
  uint8_t lockRelId = 0;
  int32_t lockRelVal = 1;
};

uint32_t packLockAndControlWord(const BdSpec &s) {
  // Identical bit layout in core word5 and shim word7.
  return (static_cast<uint32_t>(s.lockAcqId) & 0xFu) |
        ((static_cast<uint32_t>(s.lockAcqVal) & 0x7Fu) << 5) |
        ((s.lockAcqEn ? 1u : 0u) << 12) |
        ((static_cast<uint32_t>(s.lockRelId) & 0xFu) << 13) |
        ((static_cast<uint32_t>(s.lockRelVal) & 0x7Fu) << 18) |
        ((s.validBd ? 1u : 0u) << 25) | ((s.useNextBd ? 1u : 0u) << 26) |
        ((static_cast<uint32_t>(s.nextBd) & 0xFu) << 27);
}

void writeCoreBd(Tile &tile, uint32_t bdId, const BdSpec &s) {
  uint32_t base = CoreRegs::kBdBase + bdId * CoreRegs::kBdStride;
  uint32_t w0 =
      (s.lengthWords & 0x3FFFu) | ((s.addrWordOffset & 0x3FFFu) << 14);
  uint32_t w2 = (s.stepSizeMinus1[0] & 0x1FFFu) |
               ((s.stepSizeMinus1[1] & 0x1FFFu) << 13);
  uint32_t w3 = (s.stepSizeMinus1[2] & 0x1FFFu) |
               ((s.wrap[0] & 0xFFu) << 13) | ((s.wrap[1] & 0xFFu) << 21);
  uint32_t w5 = packLockAndControlWord(s);
  tile.regs().write(base + 0, w0);
  tile.regs().write(base + 4, 0);  // packet/out-of-order-bd-id: unused.
  tile.regs().write(base + 8, w2);
  tile.regs().write(base + 12, w3);
  tile.regs().write(base + 16, 0); // iteration dimension: left default.
  tile.regs().write(base + 20, w5);
}

void writeShimBd(Tile &tile, uint32_t bdId, const BdSpec &s) {
  uint32_t base = ShimRegs::kBdBase + bdId * ShimRegs::kBdStride;
  uint32_t w1 = static_cast<uint32_t>(s.shimByteAddr) & 0xFFFFFFFCu;
  uint32_t w2 = static_cast<uint32_t>(s.shimByteAddr >> 32) & 0xFFFFu;
  uint32_t w3 =
      (s.stepSizeMinus1[0] & 0xFFFFFu) | ((s.wrap[0] & 0x3FFu) << 20);
  uint32_t w4 =
      (s.stepSizeMinus1[1] & 0xFFFFFu) | ((s.wrap[1] & 0x3FFu) << 20);
  uint32_t w5 = s.stepSizeMinus1[2] & 0xFFFFFu;
  uint32_t w7 = packLockAndControlWord(s);
  tile.regs().write(base + 0, s.lengthWords);
  tile.regs().write(base + 4, w1);
  tile.regs().write(base + 8, w2);
  tile.regs().write(base + 12, w3);
  tile.regs().write(base + 16, w4);
  tile.regs().write(base + 20, w5);
  tile.regs().write(base + 24, 0); // iteration dimension: left default.
  tile.regs().write(base + 28, w7);
}

void startQueue(Tile &tile, uint32_t ctrlBase, uint32_t ctrlStride,
               uint32_t numCh, DmaDirection dir, uint32_t ch,
               uint8_t startBd) {
  uint32_t v = startBd & 0xFu; // repeatCount-1 = 0 (one shot), enToken = 0.
  tile.regs().write(ctrlQueueOff(ctrlBase, ctrlStride, numCh, dir, ch), v);
}

// Fails the check and records the message rather than letting Array's
// default handler abort the whole test binary.
struct ErrorCapture {
  std::string message;
  DiagnosticHandler handler() {
    return [this](const std::string &m) { message = m; };
  }
};

//===----------------------------------------------------------------------===//
// Test 1: a 1-D BD moves the right bytes out of tile memory in the right
// order.
//===----------------------------------------------------------------------===//

void testOneDMove() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);
  AIESIM_CHECK(core->dma() != nullptr);

  uint32_t src[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 8; // No sizes/strides: D0 falls back to spanning the
                        // whole BD (see Dma.cpp's computeAddress comment).
  writeCoreBd(*core, /*bdId=*/0, spec);

  FifoPort port;
  dmaTestSetPort(*core->dma(), DmaDirection::MM2S, 0, &port);

  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/0);

  bool quiescent = array.runUntilQuiescent(1000);
  AIESIM_CHECK(quiescent);
  AIESIM_CHECK(err.message.empty());

  AIESIM_CHECK_EQ(port.words.size(), static_cast<size_t>(8));
  for (uint32_t i = 0; i < 8; ++i)
    AIESIM_CHECK_EQ(port.words[i], src[i]);
  for (uint32_t i = 0; i + 1 < port.tlasts.size(); ++i)
    AIESIM_CHECK(!port.tlasts[i]);
  AIESIM_CHECK(port.tlasts.back());

  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

//===----------------------------------------------------------------------===//
// Test 2: the n-D generator produces the exact address sequence for a
// transpose-shaped BD, matching
// test/unit_tests/aie2/30_aie2_nd_dma_transpose_repeat/aie.mlir's
// `sizes = [2, 8, 8] strides = [1, 1, 8]` (dims listed outermost-first:
// dim2, dim1, dim0), on a memory pre-filled so the value read back at
// each step IS that step's linear address -- so the pushed sequence can
// be compared directly against a hand-computed sequence.
//===----------------------------------------------------------------------===//

void testNdGenerator() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  uint32_t src[128];
  for (uint32_t i = 0; i < 128; ++i)
    src[i] = i;
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  // sizes = [2, 8, 8] strides = [1, 1, 8] => dim2 (outermost): wrap 2,
  // step 1; dim1: wrap 8, step 1; dim0 (innermost/fastest): wrap 8, step
  // 8. D2 has no wrap register (always the outermost, unbounded dim), so
  // leaving BdSpec::wrap[2] at 0 is exactly right -- it is never read.
  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 128;
  spec.stepSizeMinus1[0] = 8 - 1;
  spec.wrap[0] = 8;
  spec.stepSizeMinus1[1] = 1 - 1;
  spec.wrap[1] = 8;
  spec.stepSizeMinus1[2] = 1 - 1;
  writeCoreBd(*core, /*bdId=*/1, spec);

  FifoPort port;
  dmaTestSetPort(*core->dma(), DmaDirection::MM2S, 0, &port);
  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/1);

  AIESIM_CHECK(array.runUntilQuiescent(2000));
  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(port.words.size(), static_cast<size_t>(128));

  // Hand-computed expected sequence: D0 fastest (mod 8), then D1 (mod 8),
  // then D2 (unbounded remainder) -- the same nested-loop order
  // test/unit_tests/aie2/29_aie2_nd_dma_even_odd/test.cpp's
  // populate_expected() uses for its own (different) stride/size choice.
  for (uint32_t l = 0; l < 128; ++l) {
    uint32_t d0 = l % 8;
    uint32_t d1 = (l / 8) % 8;
    uint32_t d2 = l / 64;
    uint32_t expected = d0 * 8 + d1 * 1 + d2 * 1;
    AIESIM_CHECK_EQ(port.words[l], expected);
  }

  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

//===----------------------------------------------------------------------===//
// Test 3: a channel stalls rather than proceeding when its lock acquire
// cannot succeed, and completedBds only counts once the BD actually
// finishes.
//===----------------------------------------------------------------------===//

void testLockStall() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  uint32_t src[4] = {100, 101, 102, 103};
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 4;
  spec.lockAcqEn = true;
  spec.lockAcqId = 0;
  spec.lockAcqVal = 1;
  spec.lockRelId = 1;
  spec.lockRelVal = 1;
  writeCoreBd(*core, /*bdId=*/2, spec);

  FifoPort port;
  dmaTestSetPort(*core->dma(), DmaDirection::MM2S, 0, &port);
  FakeLockModule fakeLocks; // armed = false: acquire always fails.
  dmaTestSetLocks(*core->dma(), &fakeLocks);

  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/2);

  // Several cycles with the lock unavailable: nothing should move, and
  // the channel must not spin/error, just stall.
  array.advance(20);
  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK(port.words.empty());
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 0u);
  AIESIM_CHECK(fakeLocks.acquireAttempts > 0);

  // The status register's StalledLockAcq bit (lsb 2) must reflect this.
  uint32_t status = core->regs().read(
      statusOff(CoreRegs::kStatusBase, CoreRegs::kStatusDirStride,
               DmaDirection::MM2S, 0));
  AIESIM_CHECK((status & 0x4u) != 0);

  // Let the lock through and let the transfer finish.
  fakeLocks.armed = true;
  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK(err.message.empty());

  AIESIM_CHECK_EQ(port.words.size(), static_cast<size_t>(4));
  for (uint32_t i = 0; i < 4; ++i)
    AIESIM_CHECK_EQ(port.words[i], src[i]);
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
  AIESIM_CHECK_EQ(fakeLocks.releaseCount, 1);

  status = core->regs().read(
      statusOff(CoreRegs::kStatusBase, CoreRegs::kStatusDirStride,
               DmaDirection::MM2S, 0));
  AIESIM_CHECK((status & 0x4u) == 0);
}

//===----------------------------------------------------------------------===//
// Test 4 (bonus, beyond the required minimum): shim DMA moves words
// between DDR (Array::ddrRead/ddrWrite) and the stream, using the 64-bit
// address assembled from the low/high registers. Exercises the one piece
// of address decode with a non-obvious register encoding (see Dma.cpp's
// kShimLayout.addrLo comment).
//===----------------------------------------------------------------------===//

void testShimDdr() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *shim = array.tile(0, 0);
  AIESIM_CHECK(shim != nullptr);
  AIESIM_CHECK(shim->dma() != nullptr);

  const uint64_t ddrAddr = 0x1'0001'0000ull; // 4-byte aligned, > 32 bits
                                             // wide to exercise AddrHigh.
  uint32_t src[4] = {0xAAAA0000u, 0xAAAA0001u, 0xAAAA0002u, 0xAAAA0003u};
  AIESIM_CHECK(array.ddrWrite(ddrAddr, src, sizeof(src)));

  BdSpec spec;
  spec.shimByteAddr = ddrAddr;
  spec.lengthWords = 4;
  writeShimBd(*shim, /*bdId=*/0, spec);

  FifoPort port;
  dmaTestSetPort(*shim->dma(), DmaDirection::MM2S, 0, &port);
  startQueue(*shim, ShimRegs::kCtrlBase, ShimRegs::kCtrlStride,
            ShimRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/0);

  AIESIM_CHECK(array.runUntilQuiescent(1000));
  AIESIM_CHECK(err.message.empty());

  AIESIM_CHECK_EQ(port.words.size(), static_cast<size_t>(4));
  for (uint32_t i = 0; i < 4; ++i)
    AIESIM_CHECK_EQ(port.words[i], src[i]);
  AIESIM_CHECK_EQ(shim->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

} // namespace

int main() {
  testOneDMove();
  testNdGenerator();
  testLockStall();
  testShimDdr();
  return aiesim_test::summarize("dma");
}
