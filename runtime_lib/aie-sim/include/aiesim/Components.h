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

/// The per-tile stream switch. Master ports carry data out of the switch
/// towards a consumer; slave ports carry data into the switch from a producer.
/// Which master is fed by which slave comes out of the switch's own
/// configuration registers, so routing is whatever the design programmed.
class StreamSwitchModule : public Steppable {
public:
  /// Endpoint a local producer writes into (the switch's slave side).
  virtual StreamPort *slavePort(PortBundle bundle, uint32_t index) = 0;
  /// Endpoint a local consumer reads from (the switch's master side).
  virtual StreamPort *masterPort(PortBundle bundle, uint32_t index) = 0;
};

/// Direction of a DMA channel, named as in the register map.
enum class DmaDirection {
  S2MM, ///< Stream to memory.
  MM2S, ///< Memory to stream.
};

/// The per-tile DMA. Buffer descriptors and channel control live in the tile
/// register file; this interface exists so tests and the array can observe
/// progress without reaching into the implementation.
class DmaModule : public Steppable {
public:
  /// True while any channel still has work queued or in flight.
  virtual bool busy() const = 0;
  /// Number of buffer descriptors completed on a channel since reset. This is
  /// what the task-completion-token registers count.
  virtual uint32_t completedBds(DmaDirection dir, uint32_t channel) const = 0;
};

/// Installers. Each is called once per tile of the appropriate type during
/// array construction, in this order: memory, locks, stream switch, DMA,
/// core. Later components may look up earlier ones through the Tile
/// accessors.
///
/// Each installer is responsible for claiming its own register ranges via
/// Tile::regs().onWrite / onRead, adopting its state onto the tile, and
/// registering itself as a Steppable with the array.
///
/// installMemory claims [0, tile.memory()->size()) so that a plain
/// XAie_Read32/Write32 at a data-memory address returns the same bytes as
/// XAie_DataMemRdWord/WrWord: aie-rt's sim IO backend has exactly one pair of
/// entry points (ess_Read32/Write32, xaie_sim.c) for every access regardless
/// of which higher-level API reached it, so data memory must be reachable
/// through the register bus like everything else or a host-side buffer
/// read/write (e.g. mlir_aie_read_buffer_local) faults as unclaimed.
void installMemory(Tile &tile);
void installLocks(Tile &tile);
void installStreamSwitch(Tile &tile);
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
/// Returns an empty mapping for everything else. That is deliberate rather
/// than incomplete: the vector and accumulator families (wl/wh, the bm/am
/// partials) need the part-assembly rule in docs/AIESimulator.md 4.4, and
/// aie-rt's fc/cr/sr/dp have no one-to-one engine register at all -- llvm-aie
/// models those bits as separate named registers (crSat, crRnd, ...), so a
/// single offset would have to be assembled from several. Both are the vector
/// phase's work; guessing either would produce a plausible wrong number, which
/// is the one failure the fault contract cannot catch.
CoreRegisterMapping coreScalarRegister(Generation gen, uint32_t off);

/// Which tile's data memory a core-local band addresses. On AIE2/AIE2P a core
/// reaches four 64 KB bands; `getMemInternalBaseAddress` returns East
/// unconditionally there, so East is the tile's OWN memory rather than a
/// neighbour, and the four are not symmetric.
enum class MemDirection { South, West, North, Own };

/// Core-local address of the band that maps the tile's OWN data memory, so
/// callers that need to turn a core address into a data-memory offset (the
/// region map, readings) take it from the one table that defines the bands
/// instead of restating the constant.
uint32_t ownMemoryBandBase(Generation gen);
/// Size of one band, i.e. the span `ownMemoryBandBase` covers.
uint32_t memoryBandSize();

/// The CoreMemoryPort a real tile presents to its engine: core-local program
/// memory at [0, 0x20000) and the tile's own data memory in the east band.
///
/// The other three bands and the core's stream/cascade ports are not modelled
/// and fault rather than stall, so a design that reaches one fails loudly
/// instead of hanging or reading a plausible wrong word.
std::unique_ptr<CoreMemoryPort> makeTileCorePort(Tile &tile);

/// Installs the thing that starts and stops a core: CORE_CONTROL's enable and
/// reset bits drive the engine, and CORE_STATUS reports DONE / ERROR_HALT.
/// Called by installCore(), which owns the offsets.
void installCoreExecution(Tile &tile, uint32_t controlOff, uint32_t statusOff);

} // namespace aiesim

#endif // AIESIM_COMPONENTS_H
