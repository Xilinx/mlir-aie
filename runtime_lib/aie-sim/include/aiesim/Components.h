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

} // namespace aiesim

#endif // AIESIM_COMPONENTS_H
