//===- StreamSwitch.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The stream switch: circuit-switched connections and packet-switched
// routing, both read out of the switch configuration registers a host
// program (via aie-rt) actually writes. Nothing here consults a
// compiler-emitted routing description; see docs/AIESimulator.md section 5.
//
// GROUNDING. Register offsets and field layouts come from
// third_party/aie-rt/driver/src/global/xaie2pgbl_params.h (AIE2P / npu2).
// They are spot-checked byte-identical to the AIE2 map in
// xaiemlgbl_params.h at every address used here (for example
// STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0 = 0x3F100 in both
// xaie2pgbl_params.h:4090 and xaiemlgbl_params.h:4221, and
// PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL = 0x3F000 in both
// xaie2pgbl_params.h:12556 and xaiemlgbl_params.h:12647), so one table
// below serves both aiesim::Generation values without branching on it.
//
// Behaviour (which register field means what, and how connections are
// made) comes from third_party/aie-rt/driver/src/stream_switch/xaie_ss.c
// and the XAie_StrmMod layout in
// third_party/aie-rt/driver/src/global/xaiegbl_regdef.h:198-231.
//
// Bundle naming and counts are cross-checked against mlir-aie's own model:
// include/aie/Dialect/AIE/IR/AIETargetModel.h
// (getNumDestSwitchboxConnections / getNumSourceSwitchboxConnections,
// implemented for AIE2 in lib/Dialect/AIE/IR/AIETargetModel.cpp:872-1019)
// and the WireBundle enum (include/aie/Dialect/AIE/IR/AIEAttrs.td:53-60).
// Every port count used below matches mlir-aie's, tile type by tile type,
// with one exception noted at kShimLayout below (Shim/PL trace-slave count:
// aie-rt says 2, mlir-aie says 1; we follow aie-rt, since that is the
// actual register map this file drives).
//
// aiesim::PortBundle (Components.h:54-64) has 9 members: Core, DMA, Ctrl,
// FIFO, South, West, North, East, Trace. aie-rt's StrmSwPortType
// (xaiegbl.h:229-240) has the same 9 plus a 10th, UCTRLR, for the AIE2P
// microcontroller tile; PortBundle has no slot for it, so this switch does
// not model uC-tile stream ports (out of scope: DeviceModel/TileType has
// no uC tile type either). mlir-aie's WireBundle
// (AIEAttrs.td:53-57) renames Ctrl to TileControl and separately lists
// PLIO and NOC; those are shim-side names for connections aie-rt still
// treats as the plain South bundle plus a distinct "shim mux" block
// (getNumDestShimMuxConnections), which is not part of the stream switch
// and is out of scope here.
//
// NOT GROUNDED, so faulted rather than guessed:
//   * The bit layout of a packet HEADER WORD on the wire. aie-rt's driver
//     only ever configures match registers (xaie_ss.c); nothing in the
//     vendored tree defines what bits of an in-flight stream word carry
//     the packet id, because that is a datapath fact aie-rt does not
//     model. We fix our own encoding (bits [4:0] = packet id, matching the
//     5-bit width XAIE_PACKET_ID_MAX in xaiegbl.h:49) and say so at the
//     point it is used, below.
//   * The polarity of a slot's MASK field (xaie_ss.c only writes it,
//     xaie_ss.c:644-654, never interprets it). We treat mask bits as
//     don't-care, the only reading that makes a mask meaningful next to an
//     exact id field; documented at the comparison, below.
//   * True hardware arbitration: two slave ports whose slot rules resolve
//     to the same (arbiter, msel) pair contending in the same cycle for
//     the same master. aie-rt's registers only select an arbiter/msel per
//     port; the round-robin/priority policy that resolves simultaneous
//     contention lives in silicon we have no register-level description
//     of. We fault via Array::error() when this happens rather than invent
//     a tie-break (see stepPacketSwitch()).
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

constexpr uint32_t kNumBundles = 9; // Core,DMA,Ctrl,FIFO,South,West,North,East,Trace
constexpr uint32_t kPortOffset = 0x4;        // Distance between ports of one bundle.
constexpr uint32_t kNumSlaveSlots = 4;       // NumSlaveSlots, every StrmMod instance.
constexpr uint32_t kSlotOffsetPerPort = 0x10; // SlotOffsetPerPort, ditto.
constexpr uint32_t kSlotOffset = 0x4;        // SlotOffset (distance between slots).
constexpr size_t kFifoDepth = 2;             // "a small FIFO per port" (task text).

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

// Our own header-word convention; see the "NOT GROUNDED" note above.
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
    moved |= stepInterTileWires();
    return moved;
  }

  // Invoked by the RegisterFile handler installStreamSwitch() registers.
  void onRegWrite(uint32_t off, uint32_t value) {
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

        bool isHeaderWord = !ss.pkt.inPacket;
        if (!ss.pkt.inPacket) {
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
            // Mask bits are treated as don't-care (see file header): a
            // header matches when the "care" bits of id and slot id agree.
            uint8_t care = static_cast<uint8_t>(~sl.mask & 0x1F);
            if ((id & care) == (sl.id & care)) {
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
          ss.pkt.inPacket = true;
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
        if (tlast)
          ss.pkt.inPacket = false;
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

  bool stepInterTileWires() {
    bool moved = false;
    moved |= wireToNeighbor(PortBundle::North, PortBundle::South, 0, 1);
    moved |= wireToNeighbor(PortBundle::South, PortBundle::North, 0, -1);
    moved |= wireToNeighbor(PortBundle::East, PortBundle::West, 1, 0);
    moved |= wireToNeighbor(PortBundle::West, PortBundle::East, -1, 0);
    return moved;
  }

  bool wireToNeighbor(PortBundle mine, PortBundle theirs, int dCol,
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
      dst->push(word, tlast);
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
};

} // namespace

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
