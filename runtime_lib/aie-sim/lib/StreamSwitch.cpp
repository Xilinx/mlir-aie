//===- StreamSwitch.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The stream switch: circuit-switched connections and packet-switched routing,
// both read out of the switch configuration registers a host program (via
// aie-rt) writes. Nothing here consults a compiler-emitted routing
// description; see docs/AIESimulator.md section 5.
//
// One offset table serves both generations -- the AIE2 and AIE2P maps are
// byte-identical at every address used here. Provenance for the offsets, the
// slot-mask polarity and the port counts is docs/AIESimulator.md 8a.4, which
// also records the two facts aie-rt does not model (the on-the-wire packet
// header layout, and arbitration between ports contending for one master);
// both fault rather than guess, at the point each is used.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace aiesim;

namespace {

// Core,DMA,Ctrl,FIFO,South,West,North,East,Trace -- aiesim::PortBundle.
constexpr uint32_t kNumBundles = 9;
constexpr uint32_t kPortOffset = 0x4;  // Distance between ports of one bundle.
constexpr uint32_t kNumSlaveSlots = 4; // NumSlaveSlots, every StrmMod instance.
constexpr uint32_t kSlotOffsetPerPort = 0x10; // SlotOffsetPerPort, ditto.
constexpr uint32_t kSlotOffset = 0x4;  // SlotOffset (distance between slots).
constexpr size_t kFifoDepth = 2;       // "a small FIFO per port" (task text).

// Universal bit fields (xaie2pgbl_params.h:3630-3648 for master,
// :4090-4100 for slave, :4390-4412 for slot; reused verbatim by the Shim
// and MemTile XAie_StrmMod instances at xaie2pgbl_reginit.c:1369-1432, so
// one set of constants covers every bundle and tile type).
constexpr uint32_t kMstrEnableMask = 0x80000000;    // bit 31
constexpr uint32_t kMstrPktEnableMask = 0x40000000; // bit 30
constexpr uint32_t kDropHeaderMask = 0x00000080;    // bit 7
constexpr uint32_t kConfigurationMask = 0x0000007F; // bits [6:0]
// Packet-mode split of the 7-bit Configuration field
// (xaie_ss.c:44-47: XAIE_SS_MASTER_PORT_ARBITOR_LSB/MASK,
// XAIE_SS_MASTER_PORT_MSELEN_LSB/MASK).
constexpr uint32_t kArbitorMask = 0x00000007; // bits [2:0] of Configuration
constexpr uint32_t kMSelEnLsb = 3;
constexpr uint32_t kMSelEnMask = 0x00000078; // bits [6:3] of Configuration

constexpr uint32_t kSlvEnableMask = 0x80000000;    // bit 31
constexpr uint32_t kSlvPktEnableMask = 0x40000000; // bit 30

constexpr uint32_t kSlotIdLsb = 24, kSlotIdMask = 0x1F000000;
constexpr uint32_t kSlotMaskLsb = 16, kSlotMaskMask = 0x001F0000;
constexpr uint32_t kSlotEnableMask = 0x00000100; // bit 8
constexpr uint32_t kSlotMselLsb = 4, kSlotMselMask = 0x00000030;
constexpr uint32_t kSlotArbitMask = 0x00000007; // bits [2:0]

// Our own convention, not aie-rt's: nothing vendored defines the on-the-wire
// header layout. Width matches XAIE_PACKET_ID_MAX (xaiegbl.h:49).
// docs/AIESimulator.md 8a.4.
constexpr uint32_t kHeaderIdMask = 0x1F; // bits [4:0]

//===----------------------------------------------------------------------===//
// Per-tile-type register layout.
//===----------------------------------------------------------------------===//

// One bundle's port counts and register bases within a tile.
struct BundlePorts {
  uint8_t mstrCount;
  uint32_t mstrBase;
  uint8_t slvCount;
  uint32_t slvBase;
  uint32_t slotBase; // Base of (port 0, slot 0); meaningless if slvCount == 0.
};

struct TileLayout {
  BundlePorts bundle[kNumBundles];
  uint32_t slvConfigBase;  // XAie_StrmMod::SlvConfigBaseAddr; see
                          // physicalSlaveIndex() below.
  uint32_t regionBegin, regionEnd; // Register range this switch claims.
};

// Core tile (XAIEGBL_TILE_TYPE_AIETILE). Ports and bases from
// Aie2PTileStrmMstr / Aie2PTileStrmSlv / Aie2PTileStrmSlaveSlot
// (xaie2pgbl_reginit.c:210-292, 518-556). Addresses (xaie2pgbl_params.h):
//   master: Core 0x3F000 (3630), DMA0 0x3F004 (3650), Ctrl 0x3F00C (3690),
//     FIFO0 0x3F010 (3710), South0 0x3F014 (3730), West0 0x3F024 (3810),
//     North0 0x3F034 (3890), East0 0x3F04C (4010)
//   slave: Core 0x3F100 (4090), DMA_0 0x3F104 (4102), Ctrl 0x3F10C (4126),
//     FIFO_0 0x3F110 (4138), South_0 0x3F114 (4150), West_0 0x3F12C (4222),
//     North_0 0x3F13C (4270), East_0 0x3F14C (4318), Trace 0x3F15C (4366)
//   slot0: Core 0x3F200 (4390), DMA_0 0x3F210 (4486), Ctrl 0x3F230 (4678),
//     FIFO_0 0x3F240 (4774), South_0 0x3F250 (4870), West_0 0x3F2B0 (5446),
//     North_0 0x3F2F0 (5830), East_0 0x3F330 (6214), Trace 0x3F370 (6598)
// SlvConfigBaseAddr = 0x3F100 (reginit.c:1337).
constexpr TileLayout kCoreLayout = {
    {
        {1, 0x0003F000, 1, 0x0003F100, 0x0003F200}, // Core
        {2, 0x0003F004, 2, 0x0003F104, 0x0003F210}, // DMA
        {1, 0x0003F00C, 1, 0x0003F10C, 0x0003F230}, // Ctrl
        {1, 0x0003F010, 1, 0x0003F110, 0x0003F240}, // FIFO
        {4, 0x0003F014, 6, 0x0003F114, 0x0003F250}, // South
        {4, 0x0003F024, 4, 0x0003F12C, 0x0003F2B0}, // West
        {6, 0x0003F034, 4, 0x0003F13C, 0x0003F2F0}, // North
        {4, 0x0003F04C, 4, 0x0003F14C, 0x0003F330}, // East
        {0, 0, 2, 0x0003F15C, 0x0003F370},          // Trace (no master side)
    },
    /*slvConfigBase=*/0x0003F100,
    /*regionBegin=*/0x0003F000,
    /*regionEnd=*/0x0003F380,
};

// Shim tile (XAIEGBL_TILE_TYPE_SHIMNOC/PL). Ports and bases from
// Aie2PShimStrmMstr / Aie2PShimStrmSlv / Aie2PShimStrmSlaveSlot
// (xaie2pgbl_reginit.c:298-380, 474-512). Addresses (xaie2pgbl_params.h):
//   master: Ctrl 0x3F000 (12556), FIFO0 0x3F004 (12576),
//     South0 0x3F008 (12596), West0 0x3F020 (12716), North0 0x3F030
//     (12796), East0 0x3F048 (12916)
//   slave: Ctrl 0x3F100 (12996), FIFO_0 0x3F104 (13008),
//     South_0 0x3F108 (13020), West_0 0x3F128 (13116),
//     North_0 0x3F138 (13164), East_0 0x3F148 (13212), Trace 0x3F158
//     (13260)
//   slot0: Ctrl 0x3F200 (13272), FIFO_0 0x3F210 (13368),
//     South_0 0x3F220 (13464), West_0 0x3F2A0 (14232),
//     North_0 0x3F2E0 (14616), East_0 0x3F320 (15000), Trace 0x3F360
//     (15384)
// SlvConfigBaseAddr = 0x3F100 (reginit.c:1371). Core/DMA bundles do not
// exist on a shim tile (NumPorts == 0 both sides).
//
// Disagreement noted per the task instructions rather than silently
// picked: aie-rt's Aie2PShimStrmSlv gives Trace 2 slave ports
// (xaie2pgbl_reginit.c:376-379); mlir-aie's AIE2TargetModel gives shim
// Source(Trace) = 1 (lib/Dialect/AIE/IR/AIETargetModel.cpp:978-979). We
// follow aie-rt (2) because it is the register map ess_Write32 actually
// receives writes against; a design that only ever uses trace slave port 0
// behaves identically either way.
constexpr TileLayout kShimLayout = {
    {
        {0, 0, 0, 0, 0},                             // Core (absent)
        {0, 0, 0, 0, 0},                             // DMA (absent)
        {1, 0x0003F000, 1, 0x0003F100, 0x0003F200},  // Ctrl
        {1, 0x0003F004, 1, 0x0003F104, 0x0003F210},  // FIFO
        {6, 0x0003F008, 8, 0x0003F108, 0x0003F220},  // South
        {4, 0x0003F020, 4, 0x0003F128, 0x0003F2A0},  // West
        {6, 0x0003F030, 4, 0x0003F138, 0x0003F2E0},  // North
        {4, 0x0003F048, 4, 0x0003F148, 0x0003F320},  // East
        {0, 0, 2, 0x0003F158, 0x0003F360},           // Trace
    },
    /*slvConfigBase=*/0x0003F100,
    /*regionBegin=*/0x0003F000,
    /*regionEnd=*/0x0003F370,
};

// MemTile (XAIEGBL_TILE_TYPE_MEMTILE). Ports and bases from
// Aie2PMemTileStrmMstr / Aie2PMemTileStrmSlv / Aie2PMemTileStrmSlaveSlot
// (xaie2pgbl_reginit.c:386-468, 562-600). Addresses (xaie2pgbl_params.h):
//   master: DMA0 0xB0000 (29822), Ctrl 0xB0018 (29942), South0 0xB001C
//     (29962), North0 0xB002C (30042)
//   slave: DMA_0 0xB0100 (30162), Ctrl 0xB0118 (30234), South_0 0xB011C
//     (30246), North_0 0xB0134 (30318), Trace 0xB0144 (30366)
//   slot0: DMA_0 0xB0200 (30378), Ctrl 0xB0260 (30954), South_0 0xB0270
//     (31050), North_0 0xB02D0 (31626), Trace 0xB0310 (32010)
// SlvConfigBaseAddr = 0xB0100 (reginit.c:1405). Core/FIFO/West/East do not
// exist on a memtile (matches mlir-aie: memtiles have no horizontal
// stream-switch links).
constexpr TileLayout kMemTileLayout = {
    {
        {0, 0, 0, 0, 0},                             // Core (absent)
        {6, 0x000B0000, 6, 0x000B0100, 0x000B0200},  // DMA
        {1, 0x000B0018, 1, 0x000B0118, 0x000B0260},  // Ctrl
        {0, 0, 0, 0, 0},                             // FIFO (absent)
        {4, 0x000B001C, 6, 0x000B011C, 0x000B0270},  // South
        {0, 0, 0, 0, 0},                             // West (absent)
        {6, 0x000B002C, 4, 0x000B0134, 0x000B02D0},  // North
        {0, 0, 0, 0, 0},                             // East (absent)
        {0, 0, 1, 0x000B0144, 0x000B0310},           // Trace
    },
    /*slvConfigBase=*/0x000B0100,
    /*regionBegin=*/0x000B0000,
    /*regionEnd=*/0x000B0320,
};

const TileLayout &layoutFor(TileType type) {
  switch (type) {
  case TileType::Core:
    return kCoreLayout;
  case TileType::MemTile:
    return kMemTileLayout;
  case TileType::Shim:
    return kShimLayout;
  case TileType::Invalid:
    break;
  }
  // installStreamSwitch() returns before calling this for Invalid; kept
  // total (rather than UB) in case that invariant ever slips.
  static constexpr TileLayout kEmpty{};
  return kEmpty;
}

//===----------------------------------------------------------------------===//
// A small FIFO per port (task text: "Model a small FIFO per port").
//===----------------------------------------------------------------------===//

class Fifo final : public StreamPort {
public:
  explicit Fifo(size_t depth) : depth(depth) {}

  bool canPush() const override { return q.size() < depth; }
  void push(uint32_t word, bool tlast) override { q.push_back({word, tlast}); }
  bool canPop() const override { return !q.empty(); }
  void pop(uint32_t &word, bool &tlast) override {
    word = q.front().data;
    tlast = q.front().tlast;
    q.pop_front();
  }

  // Not part of StreamPort: lets the packet-switch path look at the next
  // word (to read its header) without committing to consuming it, the way
  // real hardware decides routing combinationally before accepting a beat.
  std::pair<uint32_t, bool> front() const {
    return {q.front().data, q.front().tlast};
  }

private:
  struct Beat {
    uint32_t data;
    bool tlast;
  };
  std::deque<Beat> q;
  size_t depth;
};

//===----------------------------------------------------------------------===//
// Decoded register state.
//===----------------------------------------------------------------------===//

struct MasterState {
  bool enabled = false;
  bool packetMode = false;
  bool dropHeader = false;
  uint8_t slaveIdx = 0;   // Circuit mode: physical slave index feeding this
                          // master (the Configuration field, xaie_ss.c's
                          // "SlaveIdx", computed by _XAie_GetSlaveIdx).
  uint8_t arbitor = 0;    // Packet mode: which arbiter this master listens on.
  uint8_t mselEnMask = 0; // Packet mode: bit i set => this master accepts
                          // packets an arbiter routed to msel value i.
};

struct SlotState {
  bool enabled = false;
  uint8_t id = 0;
  uint8_t mask = 0;
  uint8_t msel = 0;
  uint8_t arbitor = 0;
};

// Tracks a packet in flight on one slave port: the (arbiter, msel) chosen
// by the slot rule its header matched, held until tlast.
struct PacketState {
  bool inPacket = false;
  uint8_t arbitor = 0;
  uint8_t msel = 0;
};

struct SlaveState {
  bool enabled = false;
  bool packetMode = false;
  PacketState pkt;
};

//===----------------------------------------------------------------------===//
// The switch itself.
//===----------------------------------------------------------------------===//

class StreamSwitchImpl final : public StreamSwitchModule {
public:
  StreamSwitchImpl(Tile &tile, const TileLayout &layout)
      : tile(tile), layout(layout) {
    for (uint32_t b = 0; b < kNumBundles; ++b) {
      const BundlePorts &bp = layout.bundle[b];
      for (uint32_t i = 0; i < bp.slvCount; ++i)
        slaveFifo[b].emplace_back(kFifoDepth);
      for (uint32_t i = 0; i < bp.mstrCount; ++i)
        masterFifo[b].emplace_back(kFifoDepth);
      slaveSt[b].resize(bp.slvCount);
      masterSt[b].resize(bp.mstrCount);
      slotSt[b].resize(static_cast<size_t>(bp.slvCount) * kNumSlaveSlots);
    }
  }

  StreamPort *slavePort(PortBundle bundle, uint32_t index) override {
    uint32_t b = static_cast<uint32_t>(bundle);
    if (b >= kNumBundles || index >= slaveFifo[b].size()) {
      tile.getArray().error("stream switch: invalid slave port request");
      return nullptr;
    }
    return &slaveFifo[b][index];
  }

  StreamPort *masterPort(PortBundle bundle, uint32_t index) override {
    uint32_t b = static_cast<uint32_t>(bundle);
    if (b >= kNumBundles || index >= masterFifo[b].size()) {
      tile.getArray().error("stream switch: invalid master port request");
      return nullptr;
    }
    return &masterFifo[b][index];
  }

  bool step() override {
    bool moved = stepCircuitSwitch();
    moved |= stepPacketSwitch();
    moved |= stageInterTileWires();
    return moved;
  }

  // A word sitting in any FIFO still needs to be routed onward, and a slave
  // port mid-packet (header delivered, tlast not yet seen) is committed to a
  // specific arbiter/msel even while its FIFO is momentarily empty between
  // beats. Neither can be left stranded just because this switch itself did
  // nothing on the cycle busy() is asked about -- enabled-but-idle ports are
  // deliberately NOT included: a switch that finished routing everything it
  // was ever going to see must be able to go quiescent, which is what
  // Array::runUntilQuiescent() is for.
  bool busy() const override {
    for (uint32_t b = 0; b < kNumBundles; ++b) {
      for (const Fifo &f : slaveFifo[b])
        if (f.canPop())
          return true;
      for (const Fifo &f : masterFifo[b])
        if (f.canPop())
          return true;
      for (const SlaveState &ss : slaveSt[b])
        if (ss.pkt.inPacket)
          return true;
    }
    return false;
  }

  // Publishes whatever stageInterTileWires() staged this cycle. Splitting
  // the actual cross-tile push out of step() into commit() is what makes
  // propagation exactly one hop per cycle in every direction (Array.h's
  // Steppable contract): every tile's step() this cycle reads FIFOs as they
  // stood at the end of the PREVIOUS cycle, because no tile's commit() has
  // run yet, regardless of which order tiles happen to appear in the
  // steppables list. Before this split, a tile stepped later in that list
  // (e.g. a higher column, going East) could see data pushed by a tile
  // stepped earlier in the SAME cycle and relay it again immediately, so a
  // word crossed an unbounded number of tiles per cycle going one way and
  // exactly one tile per cycle going the other.
  void commit() override {
    for (PendingWire &p : pendingWires) {
      p.dst->push(p.word, p.tlast);
      // The neighbour may have had nothing of its own outstanding and so may
      // not be in the active set right now; it just became responsible for
      // relaying this word and nothing else will prompt it to notice.
      tile.getArray().wake(p.owner);
    }
    pendingWires.clear();
  }

  // Invoked by the RegisterFile handler installStreamSwitch() registers.
  void onRegWrite(uint32_t off, uint32_t value) {
    tile.getArray().wake(this); // Any config write may open a new route.
    for (uint32_t b = 0; b < kNumBundles; ++b) {
      const BundlePorts &bp = layout.bundle[b];
      if (bp.mstrCount &&
          off >= bp.mstrBase &&
          off < bp.mstrBase + kPortOffset * bp.mstrCount) {
        decodeMaster(b, (off - bp.mstrBase) / kPortOffset, value);
        return;
      }
      if (bp.slvCount &&
          off >= bp.slvBase &&
          off < bp.slvBase + kPortOffset * bp.slvCount) {
        decodeSlave(b, (off - bp.slvBase) / kPortOffset, value);
        return;
      }
      if (bp.slvCount) {
        uint32_t span = bp.slvCount * kSlotOffsetPerPort;
        if (off >= bp.slotBase && off < bp.slotBase + span) {
          uint32_t rel = off - bp.slotBase;
          uint32_t port = rel / kSlotOffsetPerPort;
          uint32_t slot = (rel % kSlotOffsetPerPort) / kSlotOffset;
          if (slot < kNumSlaveSlots)
            decodeSlot(b, port, slot, value);
          return;
        }
      }
    }
    // Inside our claimed [regionBegin, regionEnd) but not a register we
    // model. In practice this is unreachable: the per-bundle blocks above
    // are contiguous for every tile type (verified against the addresses
    // cited at the tables above), and the one register block we do not
    // model (deterministic merge) sits outside regionEnd for all three
    // tile types (e.g. core's is at 0x3F800, past our 0x3F380 end). Kept
    // as a safety net rather than silently doing nothing.
    tile.getArray().error("stream switch: write to an unmodelled register");
  }

private:
  //===--------------------------------------------------------------===//
  // Register decode.
  //===--------------------------------------------------------------===//

  void decodeMaster(uint32_t b, uint32_t i, uint32_t value) {
    MasterState &ms = masterSt[b][i];
    ms.enabled = (value & kMstrEnableMask) != 0;
    ms.packetMode = (value & kMstrPktEnableMask) != 0;
    ms.dropHeader = (value & kDropHeaderMask) != 0;
    uint32_t config = value & kConfigurationMask;
    ms.slaveIdx = static_cast<uint8_t>(config);
    ms.arbitor = static_cast<uint8_t>(config & kArbitorMask);
    ms.mselEnMask = static_cast<uint8_t>((config & kMSelEnMask) >> kMSelEnLsb);
  }

  void decodeSlave(uint32_t b, uint32_t i, uint32_t value) {
    SlaveState &ss = slaveSt[b][i];
    ss.enabled = (value & kSlvEnableMask) != 0;
    ss.packetMode = (value & kSlvPktEnableMask) != 0;
    // Reconfiguring a slave port drops any packet in flight on it, mirroring
    // "disable writes reset values" for the analogous slot registers
    // (xaie_ss.c:710-716); a half-routed packet has nowhere sane to resume.
    ss.pkt = PacketState{};
  }

  void decodeSlot(uint32_t b, uint32_t port, uint32_t slot, uint32_t value) {
    SlotState &sl = slotSt[b][port * kNumSlaveSlots + slot];
    sl.enabled = (value & kSlotEnableMask) != 0;
    sl.id = static_cast<uint8_t>((value & kSlotIdMask) >> kSlotIdLsb);
    sl.mask = static_cast<uint8_t>((value & kSlotMaskMask) >> kSlotMaskLsb);
    sl.msel = static_cast<uint8_t>((value & kSlotMselMask) >> kSlotMselLsb);
    sl.arbitor = static_cast<uint8_t>(value & kSlotArbitMask);
  }

  //===--------------------------------------------------------------===//
  // Physical slave index <-> (bundle, index), for the master Config field.
  //
  // Circuit mode names its source slave port by a single "physical" index
  // that runs across every bundle's slave ports back to back, exactly as
  // aie-rt's _XAie_GetSlaveIdx computes it (xaie_helper.c:246-268):
  // (regOff - SlvConfigBaseAddr) / 4. We reproduce that formula rather
  // than a separate lookup table, so the two can never drift apart.
  //===--------------------------------------------------------------===//

  uint32_t physicalSlaveIndex(uint32_t bundle, uint32_t index) const {
    const BundlePorts &bp = layout.bundle[bundle];
    return (bp.slvBase + kPortOffset * index - layout.slvConfigBase) /
           kPortOffset;
  }

  std::vector<StreamRoute> configuredRoutes() const override {
    std::vector<StreamRoute> routes;
    for (uint32_t mb = 0; mb < kNumBundles; ++mb)
      for (uint32_t mi = 0; mi < layout.bundle[mb].mstrCount; ++mi) {
        const MasterState &m = masterSt[mb][mi];
        if (!m.enabled || m.packetMode)
          continue;
        uint32_t sb = 0, si = 0;
        if (!physIdxToBundle(m.slaveIdx, sb, si))
          continue; // Configuration names a slave this layout does not have.
        routes.push_back({static_cast<PortBundle>(sb), si,
                          static_cast<PortBundle>(mb), mi});
      }
    return routes;
  }

  uint32_t packetModeMasters() const override {
    uint32_t n = 0;
    for (uint32_t mb = 0; mb < kNumBundles; ++mb)
      for (uint32_t mi = 0; mi < layout.bundle[mb].mstrCount; ++mi)
        if (masterSt[mb][mi].enabled && masterSt[mb][mi].packetMode)
          ++n;
    return n;
  }

  bool physIdxToBundle(uint32_t physIdx, uint32_t &bundle,
                        uint32_t &index) const {
    for (uint32_t b = 0; b < kNumBundles; ++b) {
      const BundlePorts &bp = layout.bundle[b];
      if (bp.slvCount == 0)
        continue;
      uint32_t base = physicalSlaveIndex(b, 0);
      if (physIdx >= base && physIdx < base + bp.slvCount) {
        bundle = b;
        index = physIdx - base;
        return true;
      }
    }
    return false;
  }

  //===--------------------------------------------------------------===//
  // Circuit-switched routing: one static slave feeds one master, chosen by
  // the master's own Config field (xaie_ss.c:186-260,
  // XAie_StrmConnCctEnable). At most one word per connection per cycle.
  //===--------------------------------------------------------------===//

  bool stepCircuitSwitch() {
    bool moved = false;
    for (uint32_t b = 0; b < kNumBundles; ++b) {
      for (uint32_t i = 0; i < masterFifo[b].size(); ++i) {
        const MasterState &ms = masterSt[b][i];
        if (!ms.enabled || ms.packetMode)
          continue;
        Fifo &dst = masterFifo[b][i];
        if (!dst.canPush())
          continue;
        uint32_t sb, si;
        if (!physIdxToBundle(ms.slaveIdx, sb, si))
          continue; // Config names a slave index this tile type lacks.
        const SlaveState &ss = slaveSt[sb][si];
        if (!ss.enabled || ss.packetMode)
          continue;
        Fifo &src = slaveFifo[sb][si];
        if (!src.canPop())
          continue;
        uint32_t word;
        bool tlast;
        src.pop(word, tlast);
        dst.push(word, tlast);
        moved = true;
      }
    }
    return moved;
  }

  //===--------------------------------------------------------------===//
  // Packet-switched routing. A slave port in packet mode reads its next
  // word as a header when it is not mid-packet, matches it against its 4
  // slot rules (xaie_ss.c:602-657,
  // XAie_StrmPktSwSlaveSlotEnable/_XAie_StrmSlaveSlotConfig), and forwards
  // every word up to and including the tlast beat to whichever enabled
  // packet-mode master(s) share that slot's (arbiter, msel) pair
  // (xaie_ss.c:457-521, _XAie_StrmPktSwMstrPortConfig) -- a real fan-out
  // (multicast), since more than one master may legally share an
  // arbiter/msel. A master with DropHeader set does not receive the header
  // beat itself (xaie_ss.c:502-503 constructs DrpHdr into the same
  // register), only the payload that follows.
  //===--------------------------------------------------------------===//

  bool stepPacketSwitch() {
    bool moved = false;
    // (arbiter, msel) pairs already serviced this cycle, to detect (not
    // resolve -- see file header) real contention between slave ports.
    std::vector<std::pair<uint8_t, uint8_t>> claimed;

    for (uint32_t b = 0; b < kNumBundles; ++b) {
      for (uint32_t i = 0; i < slaveFifo[b].size(); ++i) {
        SlaveState &ss = slaveSt[b][i];
        if (!ss.enabled || !ss.packetMode)
          continue;
        Fifo &src = slaveFifo[b][i];
        if (!src.canPop())
          continue;

        // The front-of-FIFO word is a pending, undelivered header for as
        // long as `inPacket` is false. Matching it against a slot rule below
        // only resolves WHICH (arbiter, msel) it targets; it says nothing
        // about whether a listener exists yet or has room, both of which can
        // `continue` without popping the word (ordinary backpressure or a
        // master not configured yet). `inPacket` must therefore only flip to
        // true where the word is actually popped, further down, not here --
        // otherwise a later retry's `isHeaderWord` goes false for a header
        // that was never delivered, and a DropHeader master configured on
        // that retry wrongly receives it (the bug this fixes).
        bool isHeaderWord = !ss.pkt.inPacket;
        if (isHeaderWord) {
          // Our own header-word convention: bits [4:0] carry the packet id
          // (matches the 5-bit width in xaiegbl.h:49); see the "NOT
          // GROUNDED" note at the top of this file.
          uint32_t headerWord = src.front().first;
          uint8_t id = static_cast<uint8_t>(headerWord & kHeaderIdMask);
          bool matched = false;
          for (uint32_t s = 0; s < kNumSlaveSlots && !matched; ++s) {
            const SlotState &sl = slotSt[b][i * kNumSlaveSlots + s];
            if (!sl.enabled)
              continue;
            // Set mask bits must match (see GROUNDING). A slot id with a
            // bit outside its own mask therefore matches nothing and trips
            // the no-match error below, rather than silently dropping it.
            if ((id & sl.mask) == sl.id) {
              ss.pkt.arbitor = sl.arbitor;
              ss.pkt.msel = sl.msel;
              matched = true;
            }
          }
          if (!matched) {
            tile.getArray().error(
                "stream switch: packet header id " + std::to_string(id) +
                " matched no slave-slot rule");
            uint32_t discardWord;
            bool discardTlast;
            src.pop(discardWord, discardTlast);
            (void)discardWord;
            (void)discardTlast;
            moved = true;
            continue;
          }
        }

        std::vector<std::pair<uint32_t, uint32_t>> matching;
        for (uint32_t mb = 0; mb < kNumBundles; ++mb)
          for (uint32_t mi = 0; mi < masterFifo[mb].size(); ++mi) {
            const MasterState &ms = masterSt[mb][mi];
            if (ms.enabled && ms.packetMode && ms.arbitor == ss.pkt.arbitor &&
                (ms.mselEnMask & (1u << ss.pkt.msel)))
              matching.emplace_back(mb, mi);
          }
        if (matching.empty())
          continue; // Nobody configured for this arbiter/msel yet: hold.

        std::vector<Fifo *> dests;
        for (auto &mbi : matching) {
          if (isHeaderWord && masterSt[mbi.first][mbi.second].dropHeader)
            continue;
          dests.push_back(&masterFifo[mbi.first][mbi.second]);
        }
        if (!std::all_of(dests.begin(), dests.end(),
                          [](Fifo *f) { return f->canPush(); }))
          continue; // A live listener is full: hold everything (no partial
                     // fan-out).

        auto key = std::make_pair(ss.pkt.arbitor, ss.pkt.msel);
        if (std::find(claimed.begin(), claimed.end(), key) !=
            claimed.end()) {
          tile.getArray().error(
              "stream switch: unmodelled arbitration -- two slave ports "
              "contend for the same packet-switch arbiter/msel this cycle");
          continue;
        }
        claimed.push_back(key);

        uint32_t word;
        bool tlast;
        src.pop(word, tlast);
        for (Fifo *dst : dests)
          dst->push(word, tlast);
        moved = true;
        // The word is now actually delivered (or intentionally dropped for
        // every master that asked for DropHeader, if `dests` ended up
        // empty): only now has the switch truly left "awaiting header",
        // and only until tlast ends the packet.
        ss.pkt.inPacket = !tlast;
      }
    }
    return moved;
  }

  //===--------------------------------------------------------------===//
  // Inter-tile wiring (Components.h installStreamSwitch doc comment): a
  // master port on the North bundle of (c, r) is the slave port on the
  // South bundle of (c, r+1), and so on for South/East/West. This is a
  // separate mechanism from the switch's own routing above: it is the wire
  // between two tiles' switches, not a connection either switch chose.
  //===--------------------------------------------------------------===//

  bool stageInterTileWires() {
    bool moved = false;
    for (PortBundle mine : {PortBundle::North, PortBundle::South,
                            PortBundle::East, PortBundle::West}) {
      PortBundle theirs;
      int dCol, dRow;
      if (interTileLink(mine, theirs, dCol, dRow))
        moved |= stageWireToNeighbor(mine, theirs, dCol, dRow);
    }
    return moved;
  }

  // Reads this tile's own master FIFO (our state; safe to pop now, since
  // nothing else on this tile drains a North/South/East/West master FIFO
  // except this same call) and the neighbour's PUBLISHED slave-FIFO state
  // (also safe: the neighbour's commit() has not run this cycle either).
  // The actual delivery -- the write into the neighbour's FIFO -- is staged
  // into pendingWires and only applied from commit(), so it is invisible to
  // every component, on every tile, until every step() this cycle has
  // finished. See the commit() comment above for why that is what makes a
  // hop cost exactly one cycle regardless of direction.
  bool stageWireToNeighbor(PortBundle mine, PortBundle theirs, int dCol,
                           int dRow) {
    uint32_t col = tile.getCol(), row = tile.getRow();
    if (dCol < 0 && col == 0)
      return false; // Array edge: leave the port unconnected.
    if (dRow < 0 && row == 0)
      return false;
    uint32_t ncol = static_cast<uint32_t>(static_cast<int>(col) + dCol);
    uint32_t nrow = static_cast<uint32_t>(static_cast<int>(row) + dRow);
    Tile *neighbor = tile.getArray().tile(ncol, nrow);
    if (!neighbor)
      return false; // Off the edge of the simulated partition.
    StreamSwitchModule *nsw = neighbor->streamSwitch();
    if (!nsw) {
      // installStreamSwitch() runs for every tile before any other
      // component installs, so this means a real bug, not a missing
      // neighbour.
      tile.getArray().error(
          "stream switch: neighbour tile has no switch installed");
      return false;
    }
    uint32_t b = static_cast<uint32_t>(mine);
    bool moved = false;
    for (uint32_t i = 0; i < masterFifo[b].size(); ++i) {
      Fifo &src = masterFifo[b][i];
      if (!src.canPop())
        continue;
      StreamPort *dst = nsw->slavePort(theirs, i);
      if (!dst || !dst->canPush())
        continue;
      uint32_t word;
      bool tlast;
      src.pop(word, tlast);
      pendingWires.push_back({dst, word, tlast, nsw});
      moved = true;
    }
    return moved;
  }

  Tile &tile;
  const TileLayout &layout;
  std::array<std::vector<Fifo>, kNumBundles> slaveFifo;
  std::array<std::vector<Fifo>, kNumBundles> masterFifo;
  std::array<std::vector<MasterState>, kNumBundles> masterSt;
  std::array<std::vector<SlaveState>, kNumBundles> slaveSt;
  std::array<std::vector<SlotState>, kNumBundles> slotSt;

  // A word popped from our own master FIFO this cycle, destined for a
  // neighbour's slave FIFO, held until commit(). See stageWireToNeighbor().
  // `owner` is that neighbour's switch, woken on delivery (see commit()).
  struct PendingWire {
    StreamPort *dst;
    uint32_t word;
    bool tlast;
    Steppable *owner;
  };
  std::vector<PendingWire> pendingWires;
};

} // namespace

bool aiesim::interTileLink(PortBundle master, PortBundle &slave, int &dCol,
                           int &dRow) {
  switch (master) {
  case PortBundle::North: slave = PortBundle::South; dCol = 0; dRow = 1; return true;
  case PortBundle::South: slave = PortBundle::North; dCol = 0; dRow = -1; return true;
  case PortBundle::East:  slave = PortBundle::West;  dCol = 1; dRow = 0; return true;
  case PortBundle::West:  slave = PortBundle::East;  dCol = -1; dRow = 0; return true;
  default: return false;
  }
}

const char *aiesim::portBundleName(PortBundle bundle) {
  switch (bundle) {
  case PortBundle::Core: return "Core";
  case PortBundle::DMA: return "DMA";
  case PortBundle::Ctrl: return "Ctrl";
  case PortBundle::FIFO: return "FIFO";
  case PortBundle::South: return "South";
  case PortBundle::West: return "West";
  case PortBundle::North: return "North";
  case PortBundle::East: return "East";
  case PortBundle::Trace: return "Trace";
  }
  return "?";
}

void aiesim::installStreamSwitch(Tile &tile) {
  if (tile.getType() == TileType::Invalid)
    return;

  const TileLayout &layout = layoutFor(tile.getType());
  auto module = std::make_unique<StreamSwitchImpl>(tile, layout);
  StreamSwitchImpl *raw = module.get();

  tile.regs().onWrite(layout.regionBegin, layout.regionEnd,
                      [raw](uint32_t off, uint32_t value) {
                        raw->onRegWrite(off, value);
                      });

  tile.setStreamSwitch(std::move(module));
  tile.getArray().addSteppable(raw);
}
