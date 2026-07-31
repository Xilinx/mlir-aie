//===- Dma.cpp --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The per-tile DMA: buffer descriptors, channel control/task-queue, the n-D
// address generator, and lock interaction, for AIE2 and AIE2P (AIE1 is out
// of scope, per docs/AIESimulator.md).
//
// GROUNDING. Every register offset and bit field below is read out of the
// vendored aie-rt tables, not recalled from memory:
//   third_party/aie-rt/driver/src/global/xaie2pgbl_params.h  (offsets/masks)
//   third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c (which struct
//     field maps to which word/Idx: Aie2PTileDmaProp, Aie2PMemTileDmaProp,
//     Aie2PShimDmaProp and their BdEn/Pkt/Lock/AddrMode/Buffer sub-tables)
//   third_party/aie-rt/driver/src/dma/xaie_dma_aieml.c        (how a BD is
//     packed into words: _XAieMl_{Tile,MemTile,Shim}DmaWriteBd/ReadBd)
//   third_party/aie-rt/driver/src/dma/xaie_dma.c               (channel
//     control/start-queue address arithmetic and field packing:
//     _XAie_DmaChannelControl, XAie_DmaChannelSetStartQueueGeneric,
//     _XAieMl_DmaGetChannelStatus in xaie_dma_aieml.c)
// AIE2 (xaiemlgbl_params.h) and AIE2P (xaie2pgbl_params.h) were spot-checked
// side by side for the core-tile block -- BD base 0x1D000, ctrl 0x1DE00,
// start-queue 0x1DE04, status 0x1DF00, and every bit position used below --
// and are identical; both generations run through the very same aie-rt C
// functions (xaie_dma_aieml.c has no #ifdef on generation). One layout
// table therefore serves both aiesim::Generation values; see kCoreLayout
// etc. below.
//
// CROSS-CHECK. mlir-aie's own lowering independently confirms the base
// address arithmetic without going through aie-rt at all:
// lib/Dialect/AIE/IR/AIETargetModel.cpp:822-869 (AIE2TargetModel::
// getDmaBdAddress / getDmaControlAddress) computes the identical
// `0x1D000 + bd*0x20`, `0x1DE00 + ch*0x8 (+0x10 for MM2S)` and the memtile/
// shim equivalents from first principles. Also
// lib/Targets/AIETargetShared.cpp:86-133 (generateXAieDmaSetMultiDimAddr)
// confirms the n-D dimension order (MLIR lists dims outermost-first, D0 is
// always the last list entry) and that stepsize/wrap are always 32-bit-word
// granular, matching xaie_dma.c:443-447's doc comment.
//
// WHAT THIS FILE DOES NOT MODEL (ungrounded or explicitly out of scope; see
// the per-case error() calls below and the accompanying report):
//   * Packet-switched header insertion (PktEn/PktType/PktId are decoded,
//     but the on-wire header word format was not grounded from source, so
//     a BD with PktEn=1 is a hard error rather than a silent guess).
//   * The Iteration dimension's address offset (IterCurr/Iter.Wrap/
//     Iter.StepSize): decoded, but only used to detect the case where it
//     would matter (IterWrap>1 or IterCurr!=0), which is a hard error
//     rather than a silently wrong address.
//   * Zero-padding (memtile D0-D2 pad before/after): not in the required
//     field list, not decoded at all.
//   * AXI/NoC shim properties (SMID, AxCache, AxQoS, BurstLen,
//     SecureAccess), compression, out-of-order completion, FoT mode,
//     controller ID, and channel reset: no bus/compute modelling in this
//     simulator needs them, so they are left unread.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

using namespace aiesim;

namespace {

//===----------------------------------------------------------------------===//
// Bit-field helpers, mirroring aie-rt's XAie_GetField / XAie_SetField
// (third_party/aie-rt/driver/src/global/xaiegbl_defs.h:64-65) exactly.
//===----------------------------------------------------------------------===//

struct Field {
  uint32_t word = 0; // Which 32-bit BD word this lives in. Ignored (0) for
                     // the single-word channel/queue/status registers.
  uint32_t lsb = 0;
  uint32_t mask = 0; // mask == 0 is the "field does not exist" sentinel.
};

uint32_t getField(uint32_t word, Field f) { return (word & f.mask) >> f.lsb; }
uint32_t getField(const uint32_t *words, Field f) {
  return getField(words[f.word], f);
}
uint32_t setField(uint32_t word, Field f, uint32_t v) {
  return (word & ~f.mask) | ((v << f.lsb) & f.mask);
}

// BD lock-acquire/release values are a 7-bit field
// (LOCK_ACQ_VALUE_WIDTH/LOCK_REL_VALUE_WIDTH == 7 for all three tile
// kinds) carrying a signed quantity -- LockModule::tryAcquire takes
// int32_t and "a positive value means acquire greater-equal" per
// Components.h. aie-rt's own (s8) cast of a bare 7-bit extraction never
// actually goes negative (0..127 fits inside int8_t's positive range), so
// it cannot be the intended read-back of a negative acquire value; we sign
// extend from bit 6 ourselves, which is what correctly inverts the write
// side (XAie_SetField applied to a negative signed int truncates to the
// same two's-complement bit pattern this reads back).
int32_t signExtend7(uint32_t raw) {
  raw &= 0x7F;
  return (raw & 0x40) ? static_cast<int32_t>(raw) - 128
                       : static_cast<int32_t>(raw);
}

//===----------------------------------------------------------------------===//
// Per-tile-kind register layout.
//===----------------------------------------------------------------------===//

struct DmaTileLayout {
  // Buffer descriptor block.
  uint32_t bdBase = 0;
  uint32_t bdStride = 0;
  uint32_t numBds = 0;
  uint32_t numWords = 0;
  uint32_t numAddrDims = 0; // 3 (core/shim) or 4 (memtile).
  bool shimAddressing = false; // true: addrLo/addrHi hold a raw byte
                                // address; false: a word offset (x4 for
                                // bytes), per AddrAlignShift in
                                // Aie2P{Tile,MemTile,Shim}DmaProp.
  Field bufferLen;
  Field addrLo, addrHi; // addrHi unset (mask 0) for core/memtile.
  Field enPkt, pktType, pktId, outOfOrderBdId;
  Field nextBd, useNextBd, validBd, tlastSuppress;
  Field stepSize[4], wrap[4]; // wrap[numAddrDims-1] unset: the outermost
                              // configured dim has no wrap register at all
                              // (Aie2P*MultiDimProp explicitly zeroes it),
                              // so it never wraps within one BD.
  Field iterCurr, iterWrap, iterStepSize;
  Field lockAcqId, lockAcqVal, lockAcqEn, lockRelId, lockRelVal;

  // Channel control / task-queue / status block. Address arithmetic
  // grounded in xaie_dma.c:963-964,1216-1217,1684-1685 (ctrl/queue) and
  // xaie_dma_aieml.c:1095-1097 (status), cross-checked against
  // AIETargetModel.cpp:847-869.
  uint32_t ctrlBase = 0;
  uint32_t ctrlStride = 0; // Also the byte gap between the ctrl word and
                           // the start-queue word (ctrlBase+4).
  uint32_t numChannels = 0;
  uint32_t statusBase = 0;
  uint32_t statusDirStride = 0; // Byte offset from S2MM's status block to
                                // MM2S's; channels within a direction are
                                // always 4 bytes apart
                                // (XAIEML_DMA_STATUS_CHNUM_OFFSET).
  uint32_t startQSizeMax = 4; // Aie2P*DmaChProp.StartQSizeMax, all three.

  Field qStartBd, qRepeatCountM1, qEnToken; // Within the start-queue word.

  // Within the status word. Names match the AieMlDmaChStatus sub-struct.
  Field stStatus, stTaskQSize, stChannelRunning, stTaskQOverflow,
      stStalledLockAcq, stStalledLockRel, stStalledStreamStall, stStalledTct,
      stCurBd;
};

// Core tile (Memory Module). Base/words from
// xaie2pgbl_reginit.c:1889-1929 (Aie2PTileDmaMod); bit positions from
// xaie2pgbl_params.h:8325-8443 (XAIE2PGBL_MEMORY_MODULE_DMA_BD0_*) and
// :10245-10543 (channel ctrl/queue/status).
DmaTileLayout makeCoreLayout() {
  DmaTileLayout l;
  l.bdBase = 0x1D000;
  l.bdStride = 0x20;
  l.numBds = 16;
  l.numWords = 6;
  l.numAddrDims = 3;
  l.shimAddressing = false;

  l.bufferLen = {0, 0, 0x00003FFFu};
  l.addrLo = {0, 14, 0x0FFFC000u};

  l.enPkt = {1, 30, 0x40000000u};
  l.pktType = {1, 16, 0x00070000u};
  l.pktId = {1, 19, 0x00F80000u};
  l.outOfOrderBdId = {1, 24, 0x3F000000u};

  l.stepSize[0] = {2, 0, 0x00001FFFu};
  l.stepSize[1] = {2, 13, 0x03FFE000u};
  l.stepSize[2] = {3, 0, 0x00001FFFu};

  l.wrap[0] = {3, 13, 0x001FE000u};
  l.wrap[1] = {3, 21, 0x1FE00000u};
  // wrap[2]: no register (D2 is the outermost dim, always unbounded).

  l.iterCurr = {4, 19, 0x01F80000u};
  l.iterWrap = {4, 13, 0x0007E000u};
  l.iterStepSize = {4, 0, 0x00001FFFu};

  l.tlastSuppress = {5, 31, 0x80000000u};
  l.nextBd = {5, 27, 0x78000000u};
  l.useNextBd = {5, 26, 0x04000000u};
  l.validBd = {5, 25, 0x02000000u};
  l.lockRelVal = {5, 18, 0x01FC0000u};
  l.lockRelId = {5, 13, 0x0001E000u};
  l.lockAcqEn = {5, 12, 0x00001000u};
  l.lockAcqVal = {5, 5, 0x00000FE0u};
  l.lockAcqId = {5, 0, 0x0000000Fu};

  l.ctrlBase = 0x1DE00;
  l.ctrlStride = 0x8;
  l.numChannels = 2;
  l.statusBase = 0x1DF00;
  l.statusDirStride = 0x10;

  l.qStartBd = {0, 0, 0x0000000Fu};
  l.qRepeatCountM1 = {0, 16, 0x00FF0000u};
  l.qEnToken = {0, 31, 0x80000000u};

  l.stStatus = {0, 0, 0x00000003u};
  l.stStalledLockAcq = {0, 2, 0x00000004u};
  l.stStalledLockRel = {0, 3, 0x00000008u};
  l.stStalledStreamStall = {0, 4, 0x00000010u};
  l.stStalledTct = {0, 5, 0x00000020u};
  l.stTaskQOverflow = {0, 18, 0x00040000u};
  l.stChannelRunning = {0, 19, 0x00080000u};
  l.stTaskQSize = {0, 20, 0x00700000u};
  l.stCurBd = {0, 24, 0x0F000000u};
  return l;
}

// Memtile. Base/words from xaie2pgbl_reginit.c:1652-1692
// (Aie2PMemTileDmaMod); bit positions from
// xaie2pgbl_params.h:20754-20916 (XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_*) and
// :28434-28933 (channel ctrl/queue/status).
DmaTileLayout makeMemTileLayout() {
  DmaTileLayout l;
  l.bdBase = 0xA0000;
  l.bdStride = 0x20;
  l.numBds = 48;
  l.numWords = 8;
  l.numAddrDims = 4;
  l.shimAddressing = false;

  l.enPkt = {0, 31, 0x80000000u};
  l.pktType = {0, 28, 0x70000000u};
  l.pktId = {0, 23, 0x0F800000u};
  l.outOfOrderBdId = {0, 17, 0x007E0000u};
  l.bufferLen = {0, 0, 0x0001FFFFu};

  l.nextBd = {1, 20, 0x03F00000u};
  l.useNextBd = {1, 19, 0x00080000u};
  l.addrLo = {1, 0, 0x0007FFFFu};

  l.tlastSuppress = {2, 31, 0x80000000u};
  l.wrap[0] = {2, 17, 0x07FE0000u};
  l.stepSize[0] = {2, 0, 0x0001FFFFu};

  l.wrap[1] = {3, 17, 0x07FE0000u};
  l.stepSize[1] = {3, 0, 0x0001FFFFu};

  l.wrap[2] = {4, 17, 0x07FE0000u};
  l.stepSize[2] = {4, 0, 0x0001FFFFu};

  l.stepSize[3] = {5, 0, 0x0001FFFFu};
  // wrap[3]: no register (D3 is the outermost dim, always unbounded).

  l.iterCurr = {6, 23, 0x1F800000u};
  l.iterWrap = {6, 17, 0x007E0000u};
  l.iterStepSize = {6, 0, 0x0001FFFFu};

  l.validBd = {7, 31, 0x80000000u};
  l.lockRelVal = {7, 24, 0x7F000000u};
  l.lockRelId = {7, 16, 0x00FF0000u};
  l.lockAcqEn = {7, 15, 0x00008000u};
  l.lockAcqVal = {7, 8, 0x00007F00u};
  l.lockAcqId = {7, 0, 0x000000FFu};

  l.ctrlBase = 0xA0600;
  l.ctrlStride = 0x8;
  l.numChannels = 6;
  l.statusBase = 0xA0660;
  l.statusDirStride = 0x20;

  l.qStartBd = {0, 0, 0x0000000Fu};
  l.qRepeatCountM1 = {0, 16, 0x00FF0000u};
  l.qEnToken = {0, 31, 0x80000000u};

  l.stStatus = {0, 0, 0x00000003u};
  l.stStalledLockAcq = {0, 2, 0x00000004u};
  l.stStalledLockRel = {0, 3, 0x00000008u};
  l.stStalledStreamStall = {0, 4, 0x00000010u};
  l.stStalledTct = {0, 5, 0x00000020u};
  l.stTaskQOverflow = {0, 18, 0x00040000u};
  l.stChannelRunning = {0, 19, 0x00080000u};
  l.stTaskQSize = {0, 20, 0x00700000u};
  l.stCurBd = {0, 24, 0x0F000000u};
  return l;
}

// Shim (NOC Module). Base/words from xaie2pgbl_reginit.c:2140-2170
// (Aie2PShimDmaMod); bit positions from xaie2pgbl_params.h:16303-16449
// (XAIE2PGBL_NOC_MODULE_DMA_BD0_*) and :18671-18874 (channel
// ctrl/queue/status).
DmaTileLayout makeShimLayout() {
  DmaTileLayout l;
  l.bdBase = 0x1D000;
  l.bdStride = 0x20;
  l.numBds = 16;
  l.numWords = 8;
  l.numAddrDims = 3;
  l.shimAddressing = true;

  l.bufferLen = {0, 0, 0xFFFFFFFFu};

  // NOTE: lsb is deliberately 0 here, not the hardware constant 2
  // (XAIE2PGBL_NOC_MODULE_DMA_BD0_1_BASE_ADDRESS_LOW_LSB). aie-rt writes
  // this word as `(Address >> 2) << 2` (xaie_dma_aieml.c:825-828) and
  // reads it back as `GetField(word, lsb=2, mask) << 2`
  // (xaie_dma_aieml.c:969-972): the two shifts cancel, so the byte address
  // contribution is simply `word & mask` with no shift at all. lsb=0
  // reproduces that directly through the shared getField/setField above.
  l.addrLo = {1, 0, 0xFFFFFFFCu};

  l.addrHi = {2, 0, 0x0000FFFFu};
  l.enPkt = {2, 30, 0x40000000u};
  l.outOfOrderBdId = {2, 24, 0x3F000000u};
  l.pktId = {2, 19, 0x00F80000u};
  l.pktType = {2, 16, 0x00070000u};

  l.wrap[0] = {3, 20, 0x3FF00000u};
  l.stepSize[0] = {3, 0, 0x000FFFFFu};

  l.wrap[1] = {4, 20, 0x3FF00000u};
  l.stepSize[1] = {4, 0, 0x000FFFFFu};

  l.stepSize[2] = {5, 0, 0x000FFFFFu};
  // wrap[2]: no register (D2 is the outermost dim, always unbounded).

  l.iterCurr = {6, 26, 0xFC000000u};
  l.iterWrap = {6, 20, 0x03F00000u};
  l.iterStepSize = {6, 0, 0x000FFFFFu};

  l.tlastSuppress = {7, 31, 0x80000000u};
  l.nextBd = {7, 27, 0x78000000u};
  l.useNextBd = {7, 26, 0x04000000u};
  l.validBd = {7, 25, 0x02000000u};
  l.lockRelVal = {7, 18, 0x01FC0000u};
  l.lockRelId = {7, 13, 0x0001E000u};
  l.lockAcqEn = {7, 12, 0x00001000u};
  l.lockAcqVal = {7, 5, 0x00000FE0u};
  l.lockAcqId = {7, 0, 0x0000000Fu};

  l.ctrlBase = 0x1D200;
  l.ctrlStride = 0x8;
  l.numChannels = 2;
  l.statusBase = 0x1D220;
  l.statusDirStride = 0x8;

  l.qStartBd = {0, 0, 0x0000000Fu};
  l.qRepeatCountM1 = {0, 16, 0x00FF0000u};
  l.qEnToken = {0, 31, 0x80000000u};

  l.stStatus = {0, 0, 0x00000003u};
  l.stStalledLockAcq = {0, 2, 0x00000004u};
  l.stStalledLockRel = {0, 3, 0x00000008u};
  l.stStalledStreamStall = {0, 4, 0x00000010u};
  l.stStalledTct = {0, 5, 0x00000020u};
  l.stTaskQOverflow = {0, 18, 0x00040000u};
  l.stChannelRunning = {0, 19, 0x00080000u};
  l.stTaskQSize = {0, 20, 0x00700000u};
  l.stCurBd = {0, 24, 0x0F000000u};
  return l;
}

const DmaTileLayout kCoreLayout = makeCoreLayout();
const DmaTileLayout kMemTileLayout = makeMemTileLayout();
const DmaTileLayout kShimLayout = makeShimLayout();

//===----------------------------------------------------------------------===//
// A buffer descriptor, decoded live from the register file.
//===----------------------------------------------------------------------===//

struct DecodedBd {
  bool validBd = false;
  uint64_t baseAddr = 0; // Byte offset into tile memory, or a DDR byte
                         // address for shim.
  uint32_t lengthWords = 0;
  uint32_t stepSize[4] = {1, 1, 1, 1}; // Already +1 decoded.
  uint32_t wrap[4] = {0, 0, 0, 0};     // Raw register value; 0 means
                                       // "unconfigured" for D0/D1(/D2).
  uint8_t nextBd = 0;
  bool useNextBd = false;
  bool tlastSuppress = false;
  bool pktEn = false;
  uint8_t pktType = 0;
  uint8_t pktId = 0;
  bool lockAcqEn = false;
  uint8_t lockAcqId = 0;
  uint8_t lockRelId = 0;
  int32_t lockAcqVal = 0;
  int32_t lockRelVal = 0;
  uint32_t iterCurr = 0;
  uint32_t iterWrap = 1; // Decoded value is already +1 (register is Wrap-1).
};

//===----------------------------------------------------------------------===//
// Channel state: the task queue and the address generator for whichever BD
// is currently in flight.
//===----------------------------------------------------------------------===//

struct QueueTask {
  uint8_t startBd = 0;
  uint32_t repeatCount = 1; // Already +1 decoded (register is Count-1).
  bool enToken = false;
};

struct ChannelState {
  std::deque<QueueTask> queue;
  bool overflow = false;

  bool running = false;
  uint8_t startBdOfChain = 0;
  uint8_t curBd = 0;
  uint32_t remainingRepeats = 0;

  bool bdLoaded = false;
  DecodedBd bd;
  uint32_t beatsDone = 0;
  bool lockAcquired = false;

  bool stalledLockAcq = false;
  bool stalledStream = false;

  uint32_t completed = 0;

  // Test-only: see DmaModuleImpl::setTestPort.
  StreamPort *testPort = nullptr;
};

//===----------------------------------------------------------------------===//
// DmaModule.
//===----------------------------------------------------------------------===//

class DmaModuleImpl final : public DmaModule {
public:
  DmaModuleImpl(Tile &tile, TileType kind, const DmaTileLayout &layout)
      : tile(tile), kind(kind), layout(layout), s2mm(layout.numChannels),
        mm2s(layout.numChannels) {}

  bool step() override;
  bool busy() const override;
  uint32_t completedBds(DmaDirection dir, uint32_t channel) const override;

  // Register-side entry points, wired up by installDma().
  void onChannelWrite(uint32_t off, uint32_t value);
  uint32_t onStatusRead(uint32_t off) const;

  // Test-only hook. Production code (installDma) never calls this. Lock.cpp
  // and StreamSwitch.cpp are both real now, so every Core/MemTile DMA test
  // drives the genuinely-installed LockModule/StreamSwitchModule through
  // register writes, same as production. The one exception is a Shim tile's
  // DMA bundle: aie-rt's Aie2PShimStrmMstr/Slv give it NumPorts == 0
  // (third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c:304-307,
  // 348-351), so `tile.streamSwitch()->slavePort(PortBundle::DMA, ch)` has
  // no valid index to return on a Shim tile at all (shim DMA reaches the NoC
  // through a mechanism this stream-switch model does not have a port for).
  // A Shim DMA test therefore has no real port to attach to and keeps this
  // hook; see dma_test.cpp's testShimDdr.
  void setTestPort(DmaDirection dir, uint32_t ch, StreamPort *port);

private:
  ChannelState &channel(DmaDirection dir, uint32_t ch);
  const ChannelState &channel(DmaDirection dir, uint32_t ch) const;

  void pushTask(DmaDirection dir, uint32_t ch, uint8_t startBd,
                uint32_t repeatCount, bool enToken);
  DecodedBd fetchBd(uint32_t bdId) const;
  uint64_t computeAddress(const DecodedBd &bd, uint32_t beat) const;
  LockModule *effectiveLocks() const;
  StreamPort *effectivePort(DmaDirection dir, uint32_t ch, ChannelState &c);
  bool stepChannel(ChannelState &c, DmaDirection dir, uint32_t ch);
  bool moveOneWord(ChannelState &c, DmaDirection dir, uint32_t ch);

  Tile &tile;
  TileType kind;
  const DmaTileLayout &layout;
  std::vector<ChannelState> s2mm;
  std::vector<ChannelState> mm2s;
};

ChannelState &DmaModuleImpl::channel(DmaDirection dir, uint32_t ch) {
  return dir == DmaDirection::S2MM ? s2mm.at(ch) : mm2s.at(ch);
}
const ChannelState &DmaModuleImpl::channel(DmaDirection dir,
                                           uint32_t ch) const {
  return dir == DmaDirection::S2MM ? s2mm.at(ch) : mm2s.at(ch);
}

void DmaModuleImpl::setTestPort(DmaDirection dir, uint32_t ch,
                                StreamPort *port) {
  channel(dir, ch).testPort = port;
}

LockModule *DmaModuleImpl::effectiveLocks() const { return tile.locks(); }

StreamPort *DmaModuleImpl::effectivePort(DmaDirection dir, uint32_t ch,
                                         ChannelState &c) {
  if (c.testPort)
    return c.testPort;
  StreamSwitchModule *ss = tile.streamSwitch();
  if (!ss)
    return nullptr;
  return dir == DmaDirection::MM2S ? ss->slavePort(PortBundle::DMA, ch)
                                   : ss->masterPort(PortBundle::DMA, ch);
}

void DmaModuleImpl::pushTask(DmaDirection dir, uint32_t ch, uint8_t startBd,
                             uint32_t repeatCount, bool enToken) {
  ChannelState &c = channel(dir, ch);
  if (c.queue.size() >= layout.startQSizeMax) {
    // Real hardware reports this through the status register's task-
    // queue-overflow bit rather than telling the host the write failed
    // (XAie_Write32 has no failure return here). What happens to the BD id
    // that overflowed is not something we could ground from the driver
    // (it never issues one), so the documented, conservative choice is:
    // drop it, and leave the flag set until the next successful push.
    c.overflow = true;
    return;
  }
  c.overflow = false;
  c.queue.push_back(QueueTask{startBd, repeatCount, enToken});
}

void DmaModuleImpl::onChannelWrite(uint32_t off, uint32_t value) {
  uint32_t rel = off - layout.ctrlBase;
  uint32_t dirBlock = layout.ctrlStride * layout.numChannels;
  uint32_t dirIdx = rel / dirBlock;
  uint32_t withinDir = rel % dirBlock;
  uint32_t ch = withinDir / layout.ctrlStride;
  uint32_t wordOff = withinDir % layout.ctrlStride;
  if (dirIdx > 1 || ch >= layout.numChannels)
    return; // Outside the modelled range; RegisterFile already stored it.

  if (wordOff != 4)
    // The plain ctrl word (bit0 of the channel block): reset,
    // controller-id, FoT mode, out-of-order-enable, (de)compression-
    // enable. None of those have any effect in this model; see the file
    // header. Deliberately not an error() -- writing it is completely
    // normal host behaviour (aie-rt zero-initialises it), we simply do
    // not act on any of its bits.
    return;

  DmaDirection dir =
      dirIdx == 0 ? DmaDirection::S2MM : DmaDirection::MM2S;
  uint8_t startBd = static_cast<uint8_t>(getField(value, layout.qStartBd));
  uint32_t repeatCount = getField(value, layout.qRepeatCountM1) + 1;
  bool enToken = getField(value, layout.qEnToken) != 0;
  pushTask(dir, ch, startBd, repeatCount, enToken);
}

uint32_t DmaModuleImpl::onStatusRead(uint32_t off) const {
  uint32_t rel = off - layout.statusBase;
  uint32_t dirIdx = rel / layout.statusDirStride;
  uint32_t withinDir = rel % layout.statusDirStride;
  uint32_t ch = withinDir / 4; // XAIEML_DMA_STATUS_CHNUM_OFFSET.
  if (dirIdx > 1 || ch >= layout.numChannels)
    return 0; // Reserved word inside the block.

  const ChannelState &c =
      channel(dirIdx == 0 ? DmaDirection::S2MM : DmaDirection::MM2S, ch);
  uint32_t v = 0;
  v = setField(v, layout.stChannelRunning, c.running ? 1 : 0);
  v = setField(v, layout.stTaskQSize,
              static_cast<uint32_t>(c.queue.size()));
  v = setField(v, layout.stTaskQOverflow, c.overflow ? 1 : 0);
  v = setField(v, layout.stStalledLockAcq, c.stalledLockAcq ? 1 : 0);
  v = setField(v, layout.stStalledStreamStall, c.stalledStream ? 1 : 0);
  v = setField(v, layout.stCurBd, c.running ? c.curBd : 0);
  // stStalledLockRel, stStalledTct, stStatus: not modelled. Release never
  // stalls in this model (see the lock-gating note in stepChannel), and
  // the 2-bit "Status" enum's exact meaning is not consumed by any aie-rt
  // function we read (_XAieMl_DmaWaitForDone masks it out); see report.
  return v;
}

DecodedBd DmaModuleImpl::fetchBd(uint32_t bdId) const {
  DecodedBd bd;
  if (bdId >= layout.numBds) {
    tile.getArray().error(
        "DMA: BD id out of range for this tile's DMA channel");
    return bd;
  }
  uint32_t words[8] = {};
  uint32_t base = layout.bdBase + bdId * layout.bdStride;
  for (uint32_t w = 0; w < layout.numWords; ++w)
    words[w] = tile.regs().read(base + w * 4);

  bd.validBd = getField(words, layout.validBd) != 0;
  bd.lengthWords = getField(words, layout.bufferLen);

  uint32_t addrLoRaw = getField(words, layout.addrLo);
  if (layout.shimAddressing) {
    uint64_t hi = getField(words, layout.addrHi);
    bd.baseAddr = (hi << 32) | addrLoRaw;
  } else {
    bd.baseAddr = static_cast<uint64_t>(addrLoRaw) * 4;
  }

  for (uint32_t i = 0; i < layout.numAddrDims; ++i) {
    bd.stepSize[i] = getField(words, layout.stepSize[i]) + 1;
    bd.wrap[i] =
        (i + 1 < layout.numAddrDims) ? getField(words, layout.wrap[i]) : 0;
  }

  bd.nextBd = static_cast<uint8_t>(getField(words, layout.nextBd));
  bd.useNextBd = getField(words, layout.useNextBd) != 0;
  bd.tlastSuppress = getField(words, layout.tlastSuppress) != 0;

  bd.pktEn = getField(words, layout.enPkt) != 0;
  bd.pktType = static_cast<uint8_t>(getField(words, layout.pktType));
  bd.pktId = static_cast<uint8_t>(getField(words, layout.pktId));

  bd.lockAcqEn = getField(words, layout.lockAcqEn) != 0;
  bd.lockAcqId = static_cast<uint8_t>(getField(words, layout.lockAcqId));
  bd.lockRelId = static_cast<uint8_t>(getField(words, layout.lockRelId));
  bd.lockAcqVal = signExtend7(getField(words, layout.lockAcqVal));
  bd.lockRelVal = signExtend7(getField(words, layout.lockRelVal));

  bd.iterCurr = getField(words, layout.iterCurr);
  bd.iterWrap = getField(words, layout.iterWrap) + 1;
  return bd;
}

uint64_t DmaModuleImpl::computeAddress(const DecodedBd &bd,
                                       uint32_t beat) const {
  // The n-D address generator. Grounded against
  // test/unit_tests/aie2/29_aie2_nd_dma_even_odd/aie.mlir +
  // test.cpp: a BD with `sizes = [8, 2, 8] strides = [16, 1, 2]` (MLIR
  // lists dims outermost-first: dim2, dim1, dim0) must reproduce
  //   addr(l) = k*2 + j*1 + i*16
  // for l = i*16 + j*8 + k (k fastest, i slowest), i.e. D0 (wrap=8,
  // step=2) innermost, D2 (step=16) outermost. Decomposing the flat beat
  // index this way, D0-innermost-fastest, is exactly a mixed-radix
  // counter with radixes [wrap0, wrap1, ...] and an unbounded outermost
  // digit.
  //
  // D0's wrap==0 default (no dma_bd sizes/strides at all) is special-
  // cased to span the whole transfer: _XAieMl_{Tile,Shim}DmaInit
  // (xaie_dma_aieml.c:58-94) leaves every DimDesc[i].Wrap at 0, and
  // _XAieMl_DmaSetMultiDim (xaie_dma_aieml.c:168-193) only overwrites the
  // first Tensor->NumDim of them -- confirmed by
  // AIETargetShared.cpp:86-133, which sizes the XAie_DmaTensor to exactly
  // the dma_bd op's own dimension count and never touches dims beyond it.
  // So a plain `dma_bd(buf offset=0 len=N)` (no sizes/strides) leaves
  // D0/D1/D2 all at Wrap=0, StepSize=1, and must still move all N words
  // linearly: the only way that happens is if an unconfigured D0 is read
  // as "this whole BD, stride 1".
  uint32_t rem = beat;
  uint64_t wordOffset = 0;
  for (uint32_t i = 0; i + 1 < layout.numAddrDims; ++i) {
    uint32_t w = bd.wrap[i];
    if (w == 0)
      w = (i == 0) ? std::max<uint32_t>(bd.lengthWords, 1) : 1;
    uint32_t digit = rem % w;
    rem /= w;
    wordOffset += static_cast<uint64_t>(digit) * bd.stepSize[i];
  }
  // The outermost configured dim (D2 for core/shim, D3 for memtile) has no
  // wrap register in the hardware at all (Aie2PTileDmaMultiDimProp /
  // Aie2PMemTileMultiDimProp explicitly zero DmaDimProp[N-1].Wrap), so it
  // never wraps within one BD: whatever remains is its digit.
  wordOffset +=
      static_cast<uint64_t>(rem) * bd.stepSize[layout.numAddrDims - 1];
  return bd.baseAddr + wordOffset * 4;
}

bool DmaModuleImpl::stepChannel(ChannelState &c, DmaDirection dir,
                                uint32_t ch) {
  if (!c.running) {
    if (c.queue.empty())
      return false;
    QueueTask t = c.queue.front();
    c.queue.pop_front();
    c.running = true;
    c.startBdOfChain = t.startBd;
    c.curBd = t.startBd;
    c.remainingRepeats = t.repeatCount;
    c.bdLoaded = false;
  }

  if (!c.bdLoaded) {
    c.bd = fetchBd(c.curBd);
    if (!c.bd.validBd) {
      tile.getArray().error(
          "DMA channel started a BD with ValidBd=0 (host must call "
          "XAie_DmaEnableBd, or equivalent, before queueing it)");
      c.running = false;
      return false;
    }
    if (c.bd.pktEn) {
      tile.getArray().error(
          "DMA BD has packet-header mode enabled (PktEn=1); this DMA "
          "model does not know the on-wire packet header word format "
          "(not grounded from source) and refuses to move data without "
          "one rather than silently drop it");
      c.running = false;
      return false;
    }
    if (c.bd.iterWrap > 1 || c.bd.iterCurr != 0) {
      tile.getArray().error(
          "DMA BD has a non-trivial Iteration dimension configured "
          "(IterCurr != 0 or Iter.Wrap > 1); this DMA model does not "
          "apply the per-repeat iteration address offset and refuses to "
          "silently produce a possibly-wrong address rather than guess");
      c.running = false;
      return false;
    }
    c.bdLoaded = true;
    c.beatsDone = 0;
    c.lockAcquired = !c.bd.lockAcqEn;
    c.stalledLockAcq = false;
  }

  if (!c.lockAcquired) {
    LockModule *locks = effectiveLocks();
    if (!locks) {
      tile.getArray().error(
          "DMA BD requires a lock acquire (LockAcqEn=1) but this tile has "
          "no LockModule installed");
      return false;
    }
    if (!locks->tryAcquire(c.bd.lockAcqId, c.bd.lockAcqVal)) {
      // Stall: do nothing this cycle, retry next cycle. Never spin.
      c.stalledLockAcq = true;
      return false;
    }
    c.stalledLockAcq = false;
    c.lockAcquired = true;
  }

  // BufferLen==0 is a legitimate encoding (aie-rt's LenActualOffset is 0U for
  // every AIE2/AIE2P tile type -- xaie2pgbl_reginit.c / xaiemlgbl_reginit.c
  // -- so this is a literal word count with no off-by-one), used to acquire
  // and release a lock with no data movement at all. Move nothing and fall
  // straight through to BD completion below: `beatsDone < lengthWords` is an
  // unsigned compare, and `0 < 0` is false, so without this check the code
  // below called moveOneWord unconditionally and then "completed" a BD that
  // had moved one word it should never have touched, with a wrong tlast
  // (beatsDone+1 == lengthWords is never true for lengthWords==0) besides.
  if (c.bd.lengthWords != 0) {
    if (!moveOneWord(c, dir, ch))
      return false; // Stream backpressure/starvation stall.

    ++c.beatsDone;
    if (c.beatsDone < c.bd.lengthWords)
      return true;
  }

  // This BD instance is complete.
  //
  // Release is unconditional on AIE2/AIE2P hardware: there is no
  // release-enable bit anywhere in the BD word layout, only LockAcqEn --
  // _XAieMl_{Tile,MemTile,Shim}DmaWriteBd/ReadBd never touch a
  // "LockRelEn" register field, even though the software XAie_DmaDesc
  // struct carries one (used only by the separate, out-of-scope AIE1
  // path in xaie_dma_aie.c). mlir-aie agrees from the other side:
  // lib/Targets/AIERT.cpp sets relEn = false unconditionally with the
  // comment "no RelEn in the arch spec even though the API requires you to
  // set it".
  //
  // So do NOT gate this on lockAcqEn. An earlier version did, on the
  // reasoning that every design pairs an acquire with a release. That is
  // true of what mlir-aie emits today (AIERT.cpp asserts a lock-bearing BD
  // carries both), but it is not what the hardware does, and a BD with
  // acq_en = 0 and a release configured would have had its release silently
  // dropped -- a hang or wrong data, which is exactly the failure mode this
  // model refuses. A BD that configures no locks leaves these fields zero,
  // and releasing zero is a no-op, so unconditional release is safe.
  if (c.bd.lockRelVal != 0) {
    LockModule *locks = effectiveLocks();
    if (!locks) {
      tile.getArray().error("DMA BD configures a lock release but this tile "
                            "has no lock module installed");
      return false;
    }
    locks->release(c.bd.lockRelId, c.bd.lockRelVal);
  }

  c.bdLoaded = false;
  c.lockAcquired = false;

  if (c.bd.useNextBd) {
    c.curBd = c.bd.nextBd; // BD chaining.
    return true;
  }

  if (c.remainingRepeats > 1) {
    --c.remainingRepeats;
    c.curBd = c.startBdOfChain; // Repeat: restart the chain from its head.
    return true;
  }

  ++c.completed;
  c.running = false;
  return true;
}

bool DmaModuleImpl::moveOneWord(ChannelState &c, DmaDirection dir,
                                uint32_t ch) {
  // One 32-bit word per active channel per cycle. Chosen because every
  // interface this file is handed operates at that granularity already:
  // StreamPort::push/pop carry one uint32_t, and the BD's own length,
  // step and wrap fields are all documented as 32-bit-word granular
  // (xaie_dma.c:443-447's doc comment on XAie_DmaSetMultiDimAddr). A
  // 32-byte "beat" would need to invent a wider StreamPort operation that
  // Components.h does not define; a sub-word grain has no addressable
  // meaning at this layer.
  StreamPort *port = effectivePort(dir, ch, c);
  if (!port) {
    tile.getArray().error(
        "DMA channel needs a stream port but none is available (no "
        "StreamSwitchModule installed on this tile, and no test port set "
        "via dmaTestSetPort)");
    return false;
  }

  uint64_t addr = computeAddress(c.bd, c.beatsDone);
  bool isLast = (c.beatsDone + 1 == c.bd.lengthWords) && !c.bd.tlastSuppress;

  if (dir == DmaDirection::MM2S) {
    if (!port->canPush()) {
      c.stalledStream = true;
      return false;
    }
    c.stalledStream = false;
    uint32_t word = 0;
    bool ok = (kind == TileType::Shim)
                  ? tile.getArray().ddrRead(addr, &word, 4)
                  : (tile.memory() && tile.memory()->read(
                                          static_cast<uint32_t>(addr),
                                          &word, 4));
    if (!ok) {
      tile.getArray().error(
          "DMA MM2S BD generated an out-of-range source address");
      return false;
    }
    port->push(word, isLast);
    return true;
  }

  if (!port->canPop()) {
    c.stalledStream = true;
    return false;
  }
  c.stalledStream = false;
  uint32_t word = 0;
  bool tlast = false;
  port->pop(word, tlast);
  bool ok = (kind == TileType::Shim)
                ? tile.getArray().ddrWrite(addr, &word, 4)
                : (tile.memory() &&
                   tile.memory()->write(static_cast<uint32_t>(addr), &word,
                                        4));
  if (!ok) {
    tile.getArray().error(
        "DMA S2MM BD generated an out-of-range destination address");
    return false;
  }
  return true;
}

bool DmaModuleImpl::step() {
  bool moved = false;
  for (uint32_t ch = 0; ch < layout.numChannels; ++ch) {
    moved |= stepChannel(s2mm[ch], DmaDirection::S2MM, ch);
    moved |= stepChannel(mm2s[ch], DmaDirection::MM2S, ch);
  }
  return moved;
}

bool DmaModuleImpl::busy() const {
  for (const ChannelState &c : s2mm)
    if (c.running || !c.queue.empty())
      return true;
  for (const ChannelState &c : mm2s)
    if (c.running || !c.queue.empty())
      return true;
  return false;
}

uint32_t DmaModuleImpl::completedBds(DmaDirection dir, uint32_t ch) const {
  if (ch >= layout.numChannels) {
    tile.getArray().error(
        "DmaModule::completedBds: channel index out of range");
    return 0;
  }
  return channel(dir, ch).completed;
}

} // namespace

//===----------------------------------------------------------------------===//
// installDma
//===----------------------------------------------------------------------===//

void aiesim::installDma(Tile &tile) {
  const DmaTileLayout *layout;
  switch (tile.getType()) {
  case TileType::Core:
    layout = &kCoreLayout;
    break;
  case TileType::MemTile:
    layout = &kMemTileLayout;
    break;
  case TileType::Shim:
    layout = &kShimLayout;
    break;
  case TileType::Invalid:
    return;
  }

  auto dma = std::make_unique<DmaModuleImpl>(tile, tile.getType(), *layout);
  DmaModuleImpl *raw = dma.get();

  // Claim the BD block. No side effect is needed on write: a BD's fields
  // are read live out of the register file at the moment a channel starts
  // (or chains to) it -- see fetchBd() -- rather than cached, which is the
  // design point stated in Array.h. The range is still claimed here, via
  // onWrite, exactly as the installer contract in Components.h asks for,
  // so no later component can register an overlapping range by mistake.
  tile.regs().onWrite(layout->bdBase,
                      layout->bdBase + layout->numBds * layout->bdStride,
                      [](uint32_t, uint32_t) {});

  // Channel control + start-queue block, both directions back to back.
  uint32_t ctrlEnd =
      layout->ctrlBase + layout->ctrlStride * layout->numChannels * 2;
  tile.regs().onWrite(layout->ctrlBase, ctrlEnd,
                      [raw](uint32_t off, uint32_t value) {
                        raw->onChannelWrite(off, value);
                      });

  // Channel status block, both directions: computed, not stored.
  uint32_t statusEnd = layout->statusBase + layout->statusDirStride * 2;
  tile.regs().onRead(layout->statusBase, statusEnd,
                     [raw](uint32_t off) { return raw->onStatusRead(off); });

  tile.setDma(std::move(dma));
  tile.getArray().addSteppable(raw);
}

//===----------------------------------------------------------------------===//
// Test-only hook (see the DmaModuleImpl comment above for why this exists).
// Declared again, verbatim, in dma_test.cpp: both files are owned by this
// same change, and there is no shared header to put it in without touching
// Components.h, which is the contract this file does not own.
//===----------------------------------------------------------------------===//

namespace aiesim {

void dmaTestSetPort(DmaModule &dma, DmaDirection dir, uint32_t channel,
                    StreamPort *port) {
  static_cast<DmaModuleImpl &>(dma).setTestPort(dir, channel, port);
}

} // namespace aiesim
