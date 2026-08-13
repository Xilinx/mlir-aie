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
//  * Lock.cpp and StreamSwitch.cpp are REAL now (they were placeholders
//    when this file was first written, which is why Dma.cpp still carries
//    a `dmaTestSetPort` test-only hook -- see below for the one case that
//    still needs it). Every Core-tile test here drives the genuinely
//    installed LockModule and StreamSwitchModule through ordinary register
//    writes, the same interface aie-rt itself would use: a BD's lock
//    acquire is armed via the real LockN_Value register, and a channel's
//    stream side is observed by circuit-switching it to a local
//    (non-wired) stream-switch bundle and reading that bundle's port
//    directly, exactly as test/stream_switch_test.cpp does.
//
//  * The one place a hand-written test double is still unavoidable:
//    testShimDdr. aie-rt's Aie2PShimStrmMstr/Aie2PShimStrmSlv both give the
//    DMA bundle NumPorts == 0 on a Shim tile
//    (third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c:304-307,
//    348-351) -- a Shim's DMA reaches the NoC through a mechanism this
//    stream-switch model has no port for at all, so
//    `streamSwitch()->slavePort(PortBundle::DMA, ch)` has no valid index to
//    return on a Shim tile in the real module either. `dmaTestSetPort`
//    stands in for that missing port, not for a placeholder that no longer
//    exists.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include <deque>
#include <string>
#include <vector>

using namespace aiesim;

// Defined in lib/Dma.cpp. Not shared through a header: this file and
// lib/Dma.cpp are the only two files this change owns, and Components.h is
// the contract header, not ours to extend.
namespace aiesim {
void dmaTestSetPort(DmaModule &dma, DmaDirection dir, uint32_t channel,
                    StreamPort *port);
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
// dmaTestSetPort declaration above.
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

namespace MemTileRegs {
constexpr uint32_t kBdBase = 0xA0000;
constexpr uint32_t kBdStride = 0x20;
constexpr uint32_t kCtrlBase = 0xA0600;
constexpr uint32_t kCtrlStride = 0x8;
constexpr uint32_t kNumCh = 6;
// SIX bits here where the other two tile kinds have four, because 48 BDs do
// not fit in four: XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_
// START_BD_ID_MASK is 0x3F against MEMORY_MODULE's and NOC_MODULE's 0x0F.
constexpr uint32_t kStartBdMask = 0x3Fu;
} // namespace MemTileRegs

// Core-tile stream-switch register offsets, AIE2P. Independently duplicated
// from StreamSwitch.cpp's kCoreLayout (and from stream_switch_test.cpp's
// own copy of the same numbers) rather than shared: this test target does
// not get StreamSwitch.cpp's private aie-rt include path, so it could not
// include the vendored headers even if it wanted to, and a mismatch
// between the two independent copies shows up as a failing test rather
// than being hidden by sharing one set of numbers.
namespace SsRegs {
constexpr uint32_t kMstrCtrl = 0x0003F00C;      // MASTER_CONFIG_TILE_CTRL
constexpr uint32_t kMstrDma0 = 0x0003F004;      // MASTER_CONFIG_DMA0
constexpr uint32_t kSlvCtrl = 0x0003F10C;       // SLAVE_CONFIG_TILE_CTRL
constexpr uint32_t kSlvDma0 = 0x0003F104;       // SLAVE_CONFIG_DMA_0
constexpr uint32_t kSlvConfigBase = 0x0003F100; // SlvConfigBaseAddr
} // namespace SsRegs

// LockN_Value register (the direct-write path, distinct from the
// acquire/release request range): XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE
// (third_party/aie-rt/driver/src/global/xaie2pgbl_params.h:10645); stride
// is XAie_LockMod::LockSetValOff, see Lock.cpp's kValueRegOff.
namespace LockRegs {
constexpr uint32_t kValueBase = 0x0001F000;
constexpr uint32_t kValueStride = 0x10;
constexpr uint32_t kValueMask = 0x3Fu;
} // namespace LockRegs

// _XAie_GetSlaveIdx's own formula
// (third_party/aie-rt/driver/src/common/xaie_helper.c:246-268): the
// physical slave index a circuit-mode master's Configuration field names is
// (regOff - SlvConfigBaseAddr) / 4.
uint32_t physIdx(uint32_t slaveRegOff) {
  return (slaveRegOff - SsRegs::kSlvConfigBase) / 4;
}
uint32_t packMasterCircuit(uint32_t slaveIdx) {
  return (1u << 31) | (slaveIdx & 0x7Fu); // MASTER_ENABLE=1, PACKET_ENABLE=0
}
uint32_t packSlaveEnable(bool packetMode) {
  return (1u << 31) | (packetMode ? (1u << 30) : 0u);
}

// Wires DMA0's own slave port (the MM2S output side) to the Ctrl master, a
// local (non-wired) bundle, so words the channel pushes are directly
// observable through the REAL stream switch. Mirrors
// stream_switch_test.cpp's testCircuitSwitchedConnection wiring exactly.
void wireMm2sToCtrl(Tile &core) {
  core.regs().write(SsRegs::kSlvDma0, packSlaveEnable(/*packetMode=*/false));
  core.regs().write(SsRegs::kMstrCtrl,
                    packMasterCircuit(physIdx(SsRegs::kSlvDma0)));
}

// Steps `array` one cycle at a time, draining `port` after every cycle,
// until `count` words have been collected or `maxCycles` elapse. Needed
// because the real stream switch's FIFOs are deliberately small (a "small
// FIFO per port", StreamSwitch.cpp's kFifoDepth): a single
// runUntilQuiescent() call would stop advancing the moment the DMA stalls
// on a full port, well before a multi-word transfer finishes draining.
void drainInterleaved(Array &array, StreamPort &port, size_t count,
                      std::vector<uint32_t> &words, std::vector<bool> &tlasts,
                      int maxCycles) {
  for (int cyc = 0; cyc < maxCycles && words.size() < count; ++cyc) {
    array.advance(1);
    while (port.canPop()) {
      uint32_t w;
      bool tl;
      port.pop(w, tl);
      words.push_back(w);
      tlasts.push_back(tl);
    }
  }
}

//===----------------------------------------------------------------------===//
// Test double for the one component this file cannot reach through the
// real stream switch: a Shim tile's DMA bundle (see the file comment).
//===----------------------------------------------------------------------===//

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
  // Iteration dimension. Unlike the address dims above, its wrap is stored
  // minus-one too (xaie_dma_aieml.c:362), so {0,0,0} here is StepSize=1,
  // Wrap=1, IterCurr=0 -- one execution at offset 0, the untouched default.
  uint32_t iterStepSizeMinus1 = 0;
  uint32_t iterWrapMinus1 = 0;
  uint32_t iterCurr = 0;
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
  uint32_t w4 = (s.iterStepSizeMinus1 & 0x1FFFu) |
               ((s.iterWrapMinus1 & 0x3Fu) << 13) |
               ((s.iterCurr & 0x3Fu) << 19);
  uint32_t w5 = packLockAndControlWord(s);
  tile.regs().write(base + 0, w0);
  tile.regs().write(base + 4, 0); // packet/out-of-order-bd-id: unused.
  tile.regs().write(base + 8, w2);
  tile.regs().write(base + 12, w3);
  tile.regs().write(base + 16, w4);
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
  uint32_t w6 = (s.iterStepSizeMinus1 & 0xFFFFFu) |
               ((s.iterWrapMinus1 & 0x3Fu) << 20) |
               ((s.iterCurr & 0x3Fu) << 26);
  uint32_t w7 = packLockAndControlWord(s);
  tile.regs().write(base + 0, s.lengthWords);
  tile.regs().write(base + 4, w1);
  tile.regs().write(base + 8, w2);
  tile.regs().write(base + 12, w3);
  tile.regs().write(base + 16, w4);
  tile.regs().write(base + 20, w5);
  tile.regs().write(base + 24, w6);
  tile.regs().write(base + 28, w7);
}

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

// `startBdMask` is a parameter and not a constant because the field is not the
// same width on every tile kind -- see MemTileRegs::kStartBdMask. Masking here
// rather than trusting the caller is what makes this write what the hardware
// register can actually hold.
void startQueue(Tile &tile, uint32_t ctrlBase, uint32_t ctrlStride,
               uint32_t numCh, DmaDirection dir, uint32_t ch, uint32_t startBd,
               uint32_t repeatCount = 1, uint32_t startBdMask = 0xFu) {
  uint32_t v = (startBd & startBdMask) |
              (((repeatCount - 1) & 0xFFu) << 16); // enToken = 0.
  tile.regs().write(ctrlQueueOff(ctrlBase, ctrlStride, numCh, dir, ch), v);
}

// A memtile BD is 8 words with its own field placement, mirroring
// lib/Dma.cpp's kMemTileLayout. Only the fields these tests set are packed;
// the rest stay at the harmless "not used" zero, as with the core and shim
// writers above.
void writeMemTileBd(Tile &tile, uint32_t bdId, const BdSpec &s) {
  uint32_t base = MemTileRegs::kBdBase + bdId * MemTileRegs::kBdStride;
  uint32_t w0 = s.lengthWords & 0x1FFFFu;
  uint32_t w1 = (s.addrWordOffset & 0x7FFFFu) |
               ((s.useNextBd ? 1u : 0u) << 19) |
               ((static_cast<uint32_t>(s.nextBd) & 0x3Fu) << 20);
  uint32_t w2 =
      (s.stepSizeMinus1[0] & 0x1FFFFu) | ((s.wrap[0] & 0x3FFu) << 17);
  uint32_t w3 =
      (s.stepSizeMinus1[1] & 0x1FFFFu) | ((s.wrap[1] & 0x3FFu) << 17);
  uint32_t w4 = s.stepSizeMinus1[2] & 0x1FFFFu;
  uint32_t w6 = (s.iterStepSizeMinus1 & 0x1FFFFu) |
               ((s.iterWrapMinus1 & 0x3Fu) << 17) |
               ((s.iterCurr & 0x1Fu) << 23);
  uint32_t w7 = (static_cast<uint32_t>(s.lockAcqId) & 0xFFu) |
               ((static_cast<uint32_t>(s.lockAcqVal) & 0x7Fu) << 8) |
               ((s.lockAcqEn ? 1u : 0u) << 15) |
               ((static_cast<uint32_t>(s.lockRelId) & 0xFFu) << 16) |
               ((static_cast<uint32_t>(s.lockRelVal) & 0x7Fu) << 24) |
               ((s.validBd ? 1u : 0u) << 31);
  tile.regs().write(base + 0, w0);
  tile.regs().write(base + 4, w1);
  tile.regs().write(base + 8, w2);
  tile.regs().write(base + 12, w3);
  tile.regs().write(base + 16, w4);
  tile.regs().write(base + 20, 0); // D3 step size: unused.
  tile.regs().write(base + 24, w6);
  tile.regs().write(base + 28, w7);
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
// Test 1: a 1-D BD moves the right words out of tile memory in the right
// order, through the real stream switch.
//===----------------------------------------------------------------------===//

void testOneDMove() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core != nullptr);
  AIESIM_CHECK(core->dma() != nullptr);
  AIESIM_CHECK(core->streamSwitch() != nullptr);

  uint32_t src[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 8; // No sizes/strides: D0 falls back to spanning the
                        // whole BD (see Dma.cpp's computeAddress comment).
  writeCoreBd(*core, /*bdId=*/0, spec);

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);

  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/0);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 8, words, tlasts, /*maxCycles=*/500);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(8));
  for (uint32_t i = 0; i < 8 && i < words.size(); ++i)
    AIESIM_CHECK_EQ(words[i], src[i]);
  for (size_t i = 0; i + 1 < tlasts.size(); ++i)
    AIESIM_CHECK(!tlasts[i]);
  if (!tlasts.empty())
    AIESIM_CHECK(tlasts.back());

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

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/1);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 128, words, tlasts, /*maxCycles=*/2000);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(128));

  // Hand-computed expected sequence: D0 fastest (mod 8), then D1 (mod 8),
  // then D2 (unbounded remainder) -- the same nested-loop order
  // test/unit_tests/aie2/29_aie2_nd_dma_even_odd/test.cpp's
  // populate_expected() uses for its own (different) stride/size choice.
  for (uint32_t l = 0; l < 128 && l < words.size(); ++l) {
    uint32_t d0 = l % 8;
    uint32_t d1 = (l / 8) % 8;
    uint32_t d2 = l / 64;
    uint32_t expected = d0 * 8 + d1 * 1 + d2 * 1;
    AIESIM_CHECK_EQ(words[l], expected);
  }

  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

//===----------------------------------------------------------------------===//
// Test 3: a channel stalls rather than proceeding when its lock acquire
// cannot succeed, and completedBds only counts once the BD actually
// finishes. Drives the REAL LockModule (armed through the LockN_Value
// register, exactly as a host would pre-set a lock's initial value) rather
// than a fake.
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
  spec.lockAcqVal = 1; // Exact-match acquire: needs lock 0's counter
                       // (real LockModule, reset value 0) to read 1.
  spec.lockRelId = 1;
  spec.lockRelVal = 1;
  writeCoreBd(*core, /*bdId=*/2, spec);

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);

  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/2);

  // Several cycles with lock 0 still at its reset value of 0: the exact
  // match acquire (value=1) cannot succeed, so nothing should move, and
  // the channel must not spin/error, just stall.
  array.advance(20);
  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK(!sink->canPop());
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 0u);

  // The status register's StalledLockAcq bit (lsb 2) must reflect this.
  uint32_t status = core->regs().read(
      statusOff(CoreRegs::kStatusBase, CoreRegs::kStatusDirStride,
               DmaDirection::MM2S, 0));
  AIESIM_CHECK((status & 0x4u) != 0);

  // The same wait, attributed. The status bit says a channel is stalled right
  // now; the timeline says for how long and why, which is what the readings
  // record turns into an interval. Checked here rather than in readings_test
  // because this is the only place a real lock stall is set up.
  const std::vector<Array::TimelineTrack> &timeline = array.timeline();
  const Array::TimelineTrack *dmaTrack = nullptr;
  for (const Array::TimelineTrack &t : timeline)
    if (t.entity == "tile:0,2/dma")
      dmaTrack = &t;
  AIESIM_CHECK(dmaTrack != nullptr);
  if (dmaTrack) {
    // Every scheduled cycle so far was the lock wait, so it coalesces into a
    // single span. More than one span here would mean the DMA was credited
    // with work it never did.
    AIESIM_CHECK_EQ(dmaTrack->spans.size(), static_cast<size_t>(1));
    AIESIM_CHECK(std::string(dmaTrack->spans[0].category) == "lock");
    uint64_t stalled = 0;
    for (const Array::TimelineSpan &s : dmaTrack->spans)
      stalled += s.end - s.start;
    AIESIM_CHECK(stalled > 0);
    AIESIM_CHECK(stalled <= array.cycle());
  }

  // Arm lock 0 through the real LockN_Value register (Lock.cpp's
  // setRawValue / xaie_locks_aieml.c:150-164 direct-write path) and let the
  // transfer finish.
  core->regs().write(LockRegs::kValueBase + 0 * LockRegs::kValueStride, 1);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 4, words, tlasts, /*maxCycles=*/500);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(4));
  for (uint32_t i = 0; i < 4 && i < words.size(); ++i)
    AIESIM_CHECK_EQ(words[i], src[i]);
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);

  // Release side: lock 1's counter must now read back lockRelVal (1),
  // confirming the release ran against the real LockModule.
  uint32_t lock1 = core->regs().read(LockRegs::kValueBase +
                                     1 * LockRegs::kValueStride) &
                   LockRegs::kValueMask;
  AIESIM_CHECK_EQ(lock1, 1u);

  status = core->regs().read(
      statusOff(CoreRegs::kStatusBase, CoreRegs::kStatusDirStride,
               DmaDirection::MM2S, 0));
  AIESIM_CHECK((status & 0x4u) == 0);
}

//===----------------------------------------------------------------------===//
// Test 4 (regression for the BufferLen==0 finding): a zero-length BD must
// move zero words, while still performing its lock acquire and release
// and completing normally. Before the fix, `stepChannel` called
// moveOneWord unconditionally and an unsigned `beatsDone(0) <
// lengthWords(0)` compared false, so the BD "completed" after silently
// moving one word it should never have touched.
//===----------------------------------------------------------------------===//

void testZeroLengthBdMovesNoDataStillLocks() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  uint32_t poison = 0xDEADBEEFu;
  AIESIM_CHECK(core->memory()->write(0, &poison, sizeof(poison)));

  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 0; // The case under test: BufferLen=0 is a legitimate
                        // encoding (aie-rt's LenActualOffset is 0U for
                        // every AIE2/AIE2P tile type -- see Dma.cpp's file
                        // header), used to acquire and release a lock with
                        // no data movement.
  spec.lockAcqEn = true;
  spec.lockAcqId = 2;
  spec.lockAcqVal = 1;
  spec.lockRelId = 3;
  spec.lockRelVal = 1;
  writeCoreBd(*core, /*bdId=*/4, spec);

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);

  // Arm lock 2 so the acquire (still required for a zero-length BD) can
  // succeed.
  core->regs().write(LockRegs::kValueBase + 2 * LockRegs::kValueStride, 1);

  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/4);

  AIESIM_CHECK(array.runUntilQuiescent(200));
  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK(!sink->canPop()); // Zero words moved, not one.
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);

  // Lock 3's counter must show the release ran even though no data moved.
  uint32_t lock3 = core->regs().read(LockRegs::kValueBase +
                                     3 * LockRegs::kValueStride) &
                   LockRegs::kValueMask;
  AIESIM_CHECK_EQ(lock3, 1u);
}

//===----------------------------------------------------------------------===//
// Test 5: S2MM, previously untested entirely in this file. A local
// (non-wired) stream-switch port stands in for whatever producer a real
// design would wire up, circuit-switched into DMA0's master port so the
// S2MM channel reads through the REAL stream switch.
//===----------------------------------------------------------------------===//

void testS2mmMove() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  AIESIM_CHECK(core->dma() != nullptr);
  AIESIM_CHECK(core->streamSwitch() != nullptr);

  RegisterFile &regs = core->regs();
  regs.write(SsRegs::kSlvCtrl, packSlaveEnable(/*packetMode=*/false));
  regs.write(SsRegs::kMstrDma0, packMasterCircuit(physIdx(SsRegs::kSlvCtrl)));

  BdSpec spec;
  spec.addrWordOffset = 0;
  spec.lengthWords = 4;
  writeCoreBd(*core, /*bdId=*/3, spec);
  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::S2MM, 0, /*startBd=*/3);

  StreamPort *source = core->streamSwitch()->slavePort(PortBundle::Ctrl, 0);
  source->push(200, false);
  source->push(201, false);
  source->push(202, false);
  source->push(203, true);

  AIESIM_CHECK(array.runUntilQuiescent(500));
  AIESIM_CHECK(err.message.empty());

  uint32_t dst[4] = {};
  AIESIM_CHECK(core->memory()->read(0, dst, sizeof(dst)));
  AIESIM_CHECK_EQ(dst[0], 200u);
  AIESIM_CHECK_EQ(dst[1], 201u);
  AIESIM_CHECK_EQ(dst[2], 202u);
  AIESIM_CHECK_EQ(dst[3], 203u);
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::S2MM, 0), 1u);
}

//===----------------------------------------------------------------------===//
// Test 6 (bonus, beyond the required minimum): shim DMA moves words
// between DDR (Array::ddrRead/ddrWrite) and the stream, using the 64-bit
// address assembled from the low/high registers. Exercises the one piece
// of address decode with a non-obvious register encoding (see Dma.cpp's
// kShimLayout.addrLo comment). Uses dmaTestSetPort: see the file header
// for why a Shim tile's DMA bundle has no real port to attach to.
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

//===----------------------------------------------------------------------===//
// Test 7: the Iteration dimension. One BD executed N times by the task
// queue's repeat count walks its own base by Iter.StepSize each execution --
// the rolled form of an N-deep BD chain, which is what Xilinx/mlir-aie#3538
// exposes on `aie.dma_bd`. Both halves aie-rt documents are pinned here: the
// address walk, and IterCurr advancing in the register.
//===----------------------------------------------------------------------===//

uint32_t readCoreIterCurr(Tile &tile, uint32_t bdId) {
  uint32_t w =
      tile.regs().read(CoreRegs::kBdBase + bdId * CoreRegs::kBdStride + 16);
  return (w & 0x01F80000u) >> 19;
}

void testBdIterationWalksItsOwnBase() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  uint32_t src[32];
  for (uint32_t i = 0; i < 32; ++i)
    src[i] = i;
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  // 4 words per execution, base advancing 4 words per execution, wrapping
  // after 4: one BD covering the same 16 words a 4-deep chain would.
  BdSpec spec;
  spec.lengthWords = 4;
  spec.iterStepSizeMinus1 = 4 - 1;
  spec.iterWrapMinus1 = 4 - 1;
  writeCoreBd(*core, /*bdId=*/1, spec);

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/1,
            /*repeatCount=*/4);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 16, words, tlasts, /*maxCycles=*/2000);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(16));
  for (uint32_t i = 0; i < 16 && i < words.size(); ++i)
    AIESIM_CHECK_EQ(words[i], i);

  // Read the counter back out of the REGISTER, not out of channel state:
  // that read-back is the observable separating "a counter the hardware
  // advances" from "a settable starting offset", which is the open review
  // question on #3538. Four executions at Wrap=4 return it to 0.
  AIESIM_CHECK_EQ(readCoreIterCurr(*core, 1), 0u);
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

//===----------------------------------------------------------------------===//
// Test 8: IterCurr is also a settable STARTING offset -- a BD written with
// IterCurr=2 begins two iteration steps in, and leaves 3 behind.
//===----------------------------------------------------------------------===//

void testBdIterationStartsWhereIterCurrSaid() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *core = array.tile(0, 2);
  uint32_t src[32];
  for (uint32_t i = 0; i < 32; ++i)
    src[i] = i;
  AIESIM_CHECK(core->memory()->write(0, src, sizeof(src)));

  BdSpec spec;
  spec.lengthWords = 4;
  spec.iterStepSizeMinus1 = 4 - 1;
  spec.iterWrapMinus1 = 4 - 1;
  spec.iterCurr = 2;
  writeCoreBd(*core, /*bdId=*/1, spec);

  wireMm2sToCtrl(*core);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  startQueue(*core, CoreRegs::kCtrlBase, CoreRegs::kCtrlStride,
            CoreRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/1);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 4, words, tlasts, /*maxCycles=*/2000);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(4));
  for (uint32_t i = 0; i < 4 && i < words.size(); ++i)
    AIESIM_CHECK_EQ(words[i], 2 * 4 + i);
  AIESIM_CHECK_EQ(readCoreIterCurr(*core, 1), 3u);
}

//===----------------------------------------------------------------------===//
// Test 9: a memtile channel started above BD 15 runs the BD it named.
//
// The start-BD field is six bits on a memtile and four everywhere else, so a
// four-bit read of a start at 24 silently aliases onto BD 8 -- a descriptor
// the design never wrote. Two decoy BDs at 8 and 9 hold different data, so
// this fails on the CONTENT rather than on a fault, which is what catches the
// aliasing even when the aliased-to BD happens to be valid.
//===----------------------------------------------------------------------===//

void testMemTileStartsAboveBd15() {
  Array array(makeTestDevice(), nullptr);
  ErrorCapture err;
  array.setDiagnosticHandler(err.handler());

  Tile *mem = array.tile(0, 1);
  AIESIM_CHECK(mem != nullptr);
  AIESIM_CHECK(mem->getType() == TileType::MemTile);

  uint32_t src[8] = {70, 71, 72, 73, 74, 75, 76, 77};
  AIESIM_CHECK(mem->memory()->write(0, src, sizeof(src)));
  uint32_t decoy[4] = {900, 901, 902, 903};
  AIESIM_CHECK(mem->memory()->write(64, decoy, sizeof(decoy)));

  // A memtile BD address indexes a west/own/east space, so the tile's OWN
  // memory starts one whole memtile up rather than at zero (Dma.cpp's
  // effectiveMemory). Byte 0 here would name the west neighbour.
  constexpr uint32_t kOwn = 0x80000 / 4;

  BdSpec real;
  real.addrWordOffset = kOwn;
  real.lengthWords = 8;
  // No lock traffic: a memtile BD's lock ids are banded the same west/own/east
  // way as its addresses, so the default id 0 would name a neighbour this
  // one-column test device does not have. Which BD ran is the question here.
  real.lockRelVal = 0;
  writeMemTileBd(*mem, /*bdId=*/24, real);

  BdSpec fake;
  fake.addrWordOffset = kOwn + 16; // Byte 64.
  fake.lengthWords = 4;
  fake.lockRelVal = 0;
  writeMemTileBd(*mem, /*bdId=*/8, fake);
  writeMemTileBd(*mem, /*bdId=*/9, fake);

  // Memtile switch: slave DMA_0 (0xB0100, the MM2S output side) into the Ctrl
  // master (0xB0018), the same local-bundle trick wireMm2sToCtrl uses on a
  // core tile. Configuration names the slave by physical index, which is
  // (0xB0100 - SlvConfigBaseAddr 0xB0100) / 4 = 0.
  mem->regs().write(0x000B0100, packSlaveEnable(/*packetMode=*/false));
  mem->regs().write(0x000B0018, packMasterCircuit(0));
  StreamPort *sink = mem->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  AIESIM_CHECK(sink != nullptr);

  startQueue(*mem, MemTileRegs::kCtrlBase, MemTileRegs::kCtrlStride,
            MemTileRegs::kNumCh, DmaDirection::MM2S, 0, /*startBd=*/24,
            /*repeatCount=*/1, MemTileRegs::kStartBdMask);

  std::vector<uint32_t> words;
  std::vector<bool> tlasts;
  drainInterleaved(array, *sink, 8, words, tlasts, /*maxCycles=*/500);

  AIESIM_CHECK(err.message.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(8));
  for (uint32_t i = 0; i < 8 && i < words.size(); ++i)
    AIESIM_CHECK_EQ(words[i], src[i]);
  AIESIM_CHECK_EQ(mem->dma()->completedBds(DmaDirection::MM2S, 0), 1u);
}

} // namespace

int main() {
  testOneDMove();
  testNdGenerator();
  testLockStall();
  testZeroLengthBdMovesNoDataStillLocks();
  testS2mmMove();
  testShimDdr();
  testBdIterationWalksItsOwnBase();
  testBdIterationStartsWhereIterCurrSaid();
  testMemTileStartsAboveBd15();
  return aiesim_test::summarize("dma");
}
