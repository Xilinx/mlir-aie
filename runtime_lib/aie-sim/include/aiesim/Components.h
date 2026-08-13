//===- Components.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The interfaces between the parts of a tile: locks, DMA, stream switch, core.
//
// Each component installs itself on a Tile, claims the register ranges it owns
// on that tile's RegisterFile, and is stepped once per simulated cycle. They
// reach each other only through the narrow interfaces here, so each one can be
// written and tested on its own.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_COMPONENTS_H
#define AIESIM_COMPONENTS_H

#include "aiesim/Array.h"

#include <cstdint>
#include <vector>

namespace aiesim {

/// AIE2/AIE2P locks are semaphores, and the acquire value is signed, with the
/// SIGN selecting the mode:
///
///   value <  0   AcquireGreaterEqual: succeed once the count is at least
///                |value|, then subtract |value|.
///   value >= 0   Acquire: succeed only on an exact match.
///
/// That polarity is not arbitrary and is easy to get backwards. It comes from
/// lib/Targets/AIERT.cpp:329-330, which negates the value when the op is
/// `acquireGE`, and it is visible in the tests, e.g.
/// test/unit_tests/chess_compiler_tests_aie2/08_tile_locks/test.cpp:51 passes
/// -2 to mean "wait for two releases".
class LockModule : public Steppable {
public:
  /// Returns false if the acquire cannot succeed right now. Callers stall;
  /// nobody blocks.
  virtual bool tryAcquire(uint32_t id, int32_t value) = 0;
  virtual void release(uint32_t id, int32_t value) = 0;
  virtual int32_t value(uint32_t id) const = 0;
  virtual uint32_t count() const = 0;
};

/// One directional endpoint of a stream. Both the stream switch and the things
/// that plug into it (DMA channels, core stream ports, cascade) see streams
/// through this.
class StreamPort {
public:
  virtual ~StreamPort() = default;
  virtual bool canPush() const = 0;
  virtual void push(uint32_t word, bool tlast) = 0;
  virtual bool canPop() const = 0;
  /// Precondition: canPop().
  virtual void pop(uint32_t &word, bool &tlast) = 0;
};

/// Stream-switch port bundles, named as in the register map and the AIE
/// dialect. `index` selects within the bundle.
enum class PortBundle {
  Core,
  DMA,
  Ctrl,
  FIFO,
  South,
  West,
  North,
  East,
  Trace,
};

/// One circuit-switched connection inside a tile: which slave port feeds which
/// master port. Packet-mode masters have no entry here -- see
/// docs/AIESimulator.md 8a.1.
struct StreamRoute {
  PortBundle slaveBundle;
  uint32_t slaveIndex;
  PortBundle masterBundle;
  uint32_t masterIndex;
};

/// The per-tile stream switch. Routing comes out of the switch's own
/// configuration registers, so it is whatever the design programmed.
class StreamSwitchModule : public Steppable {
public:
  /// Endpoint a local producer writes into (the switch's slave side).
  virtual StreamPort *slavePort(PortBundle bundle, uint32_t index) = 0;
  /// Endpoint a local consumer reads from (the switch's master side).
  virtual StreamPort *masterPort(PortBundle bundle, uint32_t index) = 0;
  /// Every enabled circuit-mode connection, in bundle then index order.
  virtual std::vector<StreamRoute> configuredRoutes() const = 0;
  /// Masters enabled in PACKET mode, which configuredRoutes() cannot describe.
  /// A non-zero count means that graph is partial.
  virtual uint32_t packetModeMasters() const = 0;
};

/// "Core", "DMA", "North" ... for naming a port in a record.
const char *portBundleName(PortBundle bundle);

/// The fixed fabric wire out of a directional master port: a master on the
/// North bundle of (c, r) IS the slave on the South bundle of (c, r+1), and so
/// on, with the port index preserved. Silicon wiring, not something a design
/// chooses -- unlike a StreamRoute, which a register decided.
///
/// \returns false for a bundle that stays inside its tile.
bool interTileLink(PortBundle master, PortBundle &slave, int &dCol, int &dRow);

/// Direction of a DMA channel, named as in the register map.
enum class DmaDirection {
  S2MM, ///< Stream to memory.
  MM2S, ///< Memory to stream.
};

/// What a shim tile's mux/demux register connects one of its stream switch's
/// south ports to on the outside. Values are the register encoding
/// (XAIE_MUX_DEMUX_CONFIG_TYPE_*, xaie_plif.c:37-39).
enum class ShimPortEndpoint : uint32_t {
  PL = 0,
  ShimDma = 1,
  NoC = 2,
};

const char *shimPortEndpointName(ShimPortEndpoint endpoint);

/// A shim tile's PL-interface mux and demux (aie.shim_mux; aie-rt's
/// XAie_PlIfMod ShimNocMux / ShimNocDeMux), which decide what the tile's south
/// stream-switch ports face.
///
/// Of the three endpoints only ShimDma is a thing this model has: there is no
/// PL and no NoC here. So a port facing either of those has no producer and no
/// consumer, which is the correct simulation of a design that steered it away
/// from the DMA -- not a gap being papered over.
///
/// Not a Steppable: it holds no data and moves none. It answers a question the
/// DMA asks about where its channels come out.
class ShimMuxModule {
public:
  virtual ~ShimMuxModule() = default;
  /// What feeds south SLAVE port `index` of this tile's stream switch.
  virtual ShimPortEndpoint slaveEndpoint(uint32_t index) const = 0;
  /// What south MASTER port `index` of this tile's stream switch drains into.
  virtual ShimPortEndpoint masterEndpoint(uint32_t index) const = 0;
};

/// The south stream-switch port a shim DMA channel reaches the fabric through,
/// or -1 for a channel the hardware gives no such port. A MASTER port index for
/// S2MM, a SLAVE port index for MM2S.
int shimDmaSouthPort(DmaDirection dir, uint32_t channel);

/// The per-tile DMA. Buffer descriptors and channel control live in the tile
/// register file; this observes progress without reaching into them.
class DmaModule : public Steppable {
public:
  /// True while any channel still has work queued or in flight.
  virtual bool busy() const = 0;
  /// Since reset. This is what the task-completion-token registers count.
  virtual uint32_t completedBds(DmaDirection dir, uint32_t channel) const = 0;
};

/// Installers. Each is called once per tile of the appropriate type during
/// array construction, in this order: memory, locks, stream switch, shim mux,
/// DMA, core. Later components may look up earlier ones through the Tile
/// accessors.
///
/// Each claims its own register ranges via Tile::regs().onWrite / onRead,
/// adopts its state onto the tile, and registers itself as a Steppable.
/// installMemory claims the whole data memory, and program memory at
/// DeviceModel::progMemHostOffset, so both are reachable through the register
/// bus -- docs/AIESimulator.md 8a.2 for why that is not a layering violation.
void installMemory(Tile &tile);
void installLocks(Tile &tile);
void installStreamSwitch(Tile &tile);
void installShimMux(Tile &tile);
void installDma(Tile &tile);
void installCore(Tile &tile);

/// Name of the other AIE2-family generation when `off` is one of ITS core
/// registers and not one of `gen`'s, else null. The two windows collide, so a
/// device mismatch shows up as an unclaimed offset rather than a wrong value;
/// this is what lets the diagnostic say which mistake was made.
const char *coreRegisterOnOtherGeneration(Generation gen, uint32_t off);

/// How a core-debug window offset reaches an engine's architectural state.
///
/// The name is stored inline rather than returned as a pointer: the indexed
/// families are computed, not tabulated, so a pointer would have to outlive a
/// scratch buffer.
struct CoreRegisterMapping {
  /// llvm-aie's name for it ("r0", "lc"), for CoreEngine::readRegister. Empty
  /// when `off` is not a scalar register this maps.
  char name[8] = {};
  /// PC has no entry in the engine's register FILE -- it is
  /// CoreEngine::getProgramCounter(). Set instead of `name`.
  bool isProgramCounter = false;

  bool mapped() const { return name[0] != '\0' || isProgramCounter; }
};

/// Maps a core-debug register-window offset onto the engine register that
/// backs it, for the scalar families that are one 32-bit slot each: r0-r31,
/// m0-m7, p0-p7, s0-s3 and sp/lr/ls/le/lc, plus PC.
///
/// Everything else returns an empty mapping by design, not by omission --
/// docs/AIESimulator.md 8a.1.
CoreRegisterMapping coreScalarRegister(Generation gen, uint32_t off);

/// Which tile's data memory a core-local band addresses. On AIE2/AIE2P a core
/// reaches four 64 KB bands; `getMemInternalBaseAddress` returns East
/// unconditionally there, so East is the tile's OWN memory rather than a
/// neighbour, and the four are not symmetric.
enum class MemDirection { South, West, North, Own };

/// Core-local address of the band that maps the tile's OWN data memory, for
/// turning a core address into a data-memory offset.
uint32_t ownMemoryBandBase(Generation gen);
/// Size of one band, i.e. the span `ownMemoryBandBase` covers.
uint32_t memoryBandSize();

/// The CoreMemoryPort a real tile presents to its engine: core-local program
/// memory and the tile's own data memory in the east band. The other three
/// bands and the core's stream/cascade ports fault rather than stall --
/// docs/AIESimulator.md 8a.1.
std::unique_ptr<CoreMemoryPort> makeTileCorePort(Tile &tile);

/// Installs the thing that starts and stops a core: CORE_CONTROL's enable and
/// reset bits drive the engine, and CORE_STATUS reports DONE / ERROR_HALT.
/// Called by installCore(), which owns the offsets.
void installCoreExecution(Tile &tile, uint32_t controlOff, uint32_t statusOff);

} // namespace aiesim

#endif // AIESIM_COMPONENTS_H
