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
// The fabric owns every architectural resource OUTSIDE the core datapath:
// program memory, data memory, locks, DMAs, stream switches, DDR. An engine
// owns only the datapath, and calls back into the fabric for every memory
// access. That split lets the ISA implementation live in llvm-aie, where the
// ISA is defined.
//
// The interface carries no LLVM types, so a factory can be satisfied by a
// dlopen'd shared object built against a different LLVM than this library.
// `aie_iss_c_abi.h` has the C ABI a loadable engine exports.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_COREENGINE_H
#define AIESIM_COREENGINE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aiesim {

/// Every memory access a core makes routes through here, so the fabric stays
/// the single owner of address decode and of side effects.
///
/// Addresses are CORE-LOCAL: what the hardware core would put on its data bus.
/// The fabric maps them onto tile-local memory, a neighbour's memory, or a
/// special port.
class CoreMemoryPort {
public:
  virtual ~CoreMemoryPort() = default;

  /// Returns false if the access is not mapped, which the engine must surface
  /// as a core fault rather than as a silent zero.
  virtual bool read(uint32_t addr, void *data, uint32_t size) = 0;

  /// Returns false if the access is not mapped.
  virtual bool write(uint32_t addr, const void *data, uint32_t size) = 0;

  /// Acquire `lockId` with `value`. Returns false if the lock is not currently
  /// acquirable; the engine must then stall the core, re-issuing the same
  /// instruction on a later cycle, rather than fail. The blocking lock ports
  /// are part of the core datapath and are not reachable through the memory
  /// map, which is why they are on this interface.
  virtual bool tryAcquireLock(uint32_t lockId, int32_t value) = 0;

  /// Never blocks on AIE2/AIE2P.
  virtual void releaseLock(uint32_t lockId, int32_t value) = 0;

  /// Returns false if the port is empty, in which case the engine stalls.
  virtual bool tryReadStream(uint32_t port, uint32_t *word, bool *tlast) = 0;

  /// Returns false if the port is full, in which case the engine stalls.
  virtual bool tryWriteStream(uint32_t port, uint32_t word, bool tlast) = 0;

  /// The 512-bit cascade port. Same stall contract as streams.
  virtual bool tryReadCascade(void *data512) = 0;
  virtual bool tryWriteCascade(const void *data512) = 0;

  /// Emit one character on the core's debug/print channel. The fabric merges
  /// these into the simulator's stdout, which is what lit tests FileCheck.
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
/// through the ess_* ABI. Program memory is NOT owned here -- the fabric owns
/// it and the engine fetches through `CoreMemoryPort` on demand, which is what
/// makes self-modifying loads and partial ELF reloads behave.
class CoreEngine {
public:
  virtual ~CoreEngine() = default;

  /// Does not touch program memory.
  virtual void reset() = 0;

  /// Set the program counter, in bytes, relative to program-memory base.
  virtual void setProgramCounter(uint32_t pc) = 0;
  virtual uint32_t getProgramCounter() const = 0;

  /// Executes at most one instruction bundle.
  virtual CoreStepResult step() = 0;

  /// Human-readable reason for the last CoreStepResult::Fault.
  virtual std::string error() const = 0;

  /// By its name in the llvm-aie register set ("r0", "p1", "wl2", "lc").
  /// False for an unknown name.
  virtual bool readRegister(const std::string &name, void *data,
                            uint32_t size) const = 0;
  virtual bool writeRegister(const std::string &name, const void *data,
                             uint32_t size) = 0;

  /// Every distinct opcode this core reached, and whether the engine had
  /// semantics for it. Accumulated as it executes, so one run reports
  /// everything it reached rather than only the instruction it stopped on.
  ///
  /// Empty by default, and that is not the same as reporting zero gaps --
  /// docs/AIESimulator.md 8a.1.
  struct OpcodeUse {
    std::string name;
    bool modelled = false;
  };
  virtual std::vector<OpcodeUse> opcodeCoverage() const { return {}; }

  /// What the schedule cost, which a bundle count cannot show: a structural
  /// hazard holds a bundle at issue, so cycles pass with nothing retiring.
  /// `cycles` exceeds `retiredBundles` by exactly `stallCycles`.
  ///
  /// `tracked` false means the engine does not count, which is not the same as
  /// counting zero.
  struct CycleCounts {
    uint64_t cycles = 0;
    uint64_t retiredBundles = 0;
    uint64_t stallCycles = 0;
    bool tracked = false;
  };
  virtual CycleCounts cycleCounts() const { return {}; }
};

/// Which ISA a core engine should implement.
enum class CoreISA { AIE2, AIE2P, AIE2PS };

/// Creates core engines. A null factory means "no core execution available",
/// which the fabric reports the first time a core is enabled rather than at
/// array construction, so core-free designs still simulate.
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
