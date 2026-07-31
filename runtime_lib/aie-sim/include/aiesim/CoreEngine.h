//===- CoreEngine.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The contract between the array/fabric model and whatever executes AIE core
// instructions.
//
// The fabric model owns every architectural resource that lives OUTSIDE the
// core datapath: program memory, data memory, locks, DMAs, stream switches,
// DDR. A core engine owns only the datapath: fetch a bundle at the current
// program counter, execute it, and call back into the fabric for every memory
// access it makes. That split is what lets the ISA implementation live in
// llvm-aie (where the ISA is defined) while the array model lives here (where
// the array is defined).
//
// The interface is deliberately narrow and free of LLVM types so that
// `AIESimEngineFactory` can be satisfied by a dlopen'd shared object built
// against a different LLVM than the one that built this library. See
// `aie_iss_c_abi.h` for the C ABI that a loadable engine exports.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_COREENGINE_H
#define AIESIM_COREENGINE_H

#include <cstdint>
#include <memory>
#include <string>

namespace aiesim {

/// Every memory access a core makes is routed back through this interface, so
/// the fabric stays the single owner of address decode and of side effects
/// (lock ports, cascade ports, stream ports and DMA-visible memory all live on
/// the far side of it).
///
/// Addresses are CORE-LOCAL: they are what the instruction stream computes,
/// i.e. what the hardware core would put on its data bus. The fabric maps them
/// onto tile-local memory, a neighbour's memory, or a special port.
class CoreMemoryPort {
public:
  virtual ~CoreMemoryPort() = default;

  /// Read `size` bytes at core-local `addr` into `data`.
  /// Returns false if the access is not mapped, which the engine must surface
  /// as a core fault rather than as a silent zero.
  virtual bool read(uint32_t addr, void *data, uint32_t size) = 0;

  /// Write `size` bytes from `data` at core-local `addr`.
  /// Returns false if the access is not mapped.
  virtual bool write(uint32_t addr, const void *data, uint32_t size) = 0;

  /// Acquire `lockId` with `value`. Returns false if the lock is not
  /// currently acquirable; the engine must then stall the core (re-issuing the
  /// same instruction on a later cycle) rather than fail. This models the
  /// blocking lock ports, which are part of the core datapath and not
  /// reachable through the memory map.
  virtual bool tryAcquireLock(uint32_t lockId, int32_t value) = 0;

  /// Release `lockId` with `value`. Release never blocks on AIE2/AIE2P.
  virtual void releaseLock(uint32_t lockId, int32_t value) = 0;

  /// Pop one 32-bit word from stream port `port`. Returns false if the port is
  /// empty, in which case the engine stalls the core.
  virtual bool tryReadStream(uint32_t port, uint32_t *word, bool *tlast) = 0;

  /// Push one 32-bit word to stream port `port`. Returns false if the port is
  /// full, in which case the engine stalls the core.
  virtual bool tryWriteStream(uint32_t port, uint32_t word, bool tlast) = 0;

  /// Pop / push the 512-bit cascade port. Same stall contract as streams.
  virtual bool tryReadCascade(void *data512) = 0;
  virtual bool tryWriteCascade(const void *data512) = 0;

  /// Emit one character on the core's debug/print channel. The fabric merges
  /// these into the simulator's stdout, which is what the existing lit tests
  /// FileCheck.
  virtual void putChar(char c) = 0;
};

/// What one call to `CoreEngine::step()` did. The fabric uses this to keep
/// time and to decide whether the array has gone quiescent.
enum class CoreStepResult {
  /// A bundle retired. The core consumed one issue slot of time.
  Retired,
  /// The core is waiting on a lock, stream, cascade or event. No architectural
  /// state changed; the fabric should re-step it on a later cycle.
  Stalled,
  /// The core executed a DONE and is halted until re-enabled through
  /// CORE_CONTROL.
  Done,
  /// The core hit something the engine cannot execute. `CoreEngine::error()`
  /// carries the reason. The fabric turns this into a hard simulation failure:
  /// an unmodelled instruction must never be silently skipped.
  Fault,
};

/// One simulated AIE core.
///
/// Lifetime: the fabric creates one per core tile at array construction, then
/// drives reset()/enable()/step() from the register writes the host makes
/// through the ess_* ABI. Program memory is NOT owned here: the fabric owns it
/// (the host loads ELF sections into it with ordinary MMIO writes, exactly as
/// on hardware) and the engine fetches through `CoreMemoryPort` on demand,
/// which is what makes self-modifying loads and partial ELF reloads behave.
class CoreEngine {
public:
  virtual ~CoreEngine() = default;

  /// Clear architectural state. Does not touch program memory.
  virtual void reset() = 0;

  /// Set the program counter, in bytes, relative to program-memory base.
  virtual void setProgramCounter(uint32_t pc) = 0;
  virtual uint32_t getProgramCounter() const = 0;

  /// Execute at most one instruction bundle. See CoreStepResult.
  virtual CoreStepResult step() = 0;

  /// Human-readable reason for the last CoreStepResult::Fault.
  virtual std::string error() const = 0;

  /// Read/write an architectural register by its name in the llvm-aie
  /// register set (for example "r0", "p1", "wl2", "lc"). Used by the
  /// core-debug register window that aie-rt exposes and by tests. Returns
  /// false for an unknown name.
  virtual bool readRegister(const std::string &name, void *data,
                            uint32_t size) const = 0;
  virtual bool writeRegister(const std::string &name, const void *data,
                             uint32_t size) = 0;
};

/// Which ISA a core engine should implement.
enum class CoreISA { AIE2, AIE2P, AIE2PS };

/// Creates core engines. The default factory resolves a loadable engine from
/// the Peano install (see `loadPeanoCoreEngineFactory`); a null factory means
/// "no core execution available", which the fabric reports as a clear
/// diagnostic the first time a core is enabled rather than at array
/// construction, so that core-free designs still simulate.
class CoreEngineFactory {
public:
  virtual ~CoreEngineFactory() = default;
  virtual std::unique_ptr<CoreEngine> create(CoreISA isa,
                                             CoreMemoryPort &port) = 0;
  /// Identifies the engine in diagnostics, e.g. "peano-iss 21.0.0git".
  virtual std::string name() const = 0;
};

/// Loads a core-engine factory from a shared object exporting the C ABI in
/// `aie_iss_c_abi.h`. `path` may be empty, in which case the search order is
///   1. $AIE_SIM_CORE_ENGINE
///   2. $PEANO_INSTALL_DIR/lib/libaie-iss.so
/// Returns null and fills `error` if no engine could be loaded.
std::unique_ptr<CoreEngineFactory>
loadCoreEngineFactory(const std::string &path, std::string &error);

} // namespace aiesim

#endif // AIESIM_COREENGINE_H
