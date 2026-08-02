//===- CoreEngineLoader.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Loads an AIE core engine through the C ABI in aie_iss_c_abi.h, and adapts it
// to the CoreEngine interface the array model uses.
//
// The engine is expected to ship with Peano, because instruction semantics
// belong with the backend that defines the ISA. It is loaded with dlopen
// rather than linked, because a Peano distribution ships libLLVM.so but no
// LLVM headers, and mlir-aie and Peano are separately versioned. See
// docs/AIESimulator.md section 4.3.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"
#include "aiesim/aie_iss_c_abi.h"

#include <cstdlib>
#include <cstring>

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace aiesim;

namespace {

int toAbiIsa(CoreISA isa) {
  switch (isa) {
  case CoreISA::AIE2:
    return AIE_ISS_ISA_AIE2;
  case CoreISA::AIE2P:
    return AIE_ISS_ISA_AIE2P;
  case CoreISA::AIE2PS:
    return AIE_ISS_ISA_AIE2PS;
  }
  return -1;
}

/// Forwards the C callbacks onto a CoreMemoryPort. One instance per core; its
/// address is the `ctx` the engine hands back.
struct CallbackBridge {
  CoreMemoryPort *port;
  aie_iss_host_callbacks cb;

  explicit CallbackBridge(CoreMemoryPort &p) : port(&p) {
    std::memset(&cb, 0, sizeof(cb));
    cb.size = sizeof(cb);
    cb.ctx = this;
    cb.read = [](void *c, uint32_t a, void *d, uint32_t n) {
      return static_cast<CallbackBridge *>(c)->port->read(a, d, n) ? 1 : 0;
    };
    cb.write = [](void *c, uint32_t a, const void *d, uint32_t n) {
      return static_cast<CallbackBridge *>(c)->port->write(a, d, n) ? 1 : 0;
    };
    cb.try_acquire_lock = [](void *c, uint32_t id, int32_t v) {
      return static_cast<CallbackBridge *>(c)->port->tryAcquireLock(id, v) ? 1
                                                                           : 0;
    };
    cb.release_lock = [](void *c, uint32_t id, int32_t v) {
      static_cast<CallbackBridge *>(c)->port->releaseLock(id, v);
    };
    cb.try_read_stream = [](void *c, uint32_t p, uint32_t *w, int *tlast) {
      bool last = false;
      bool ok =
          static_cast<CallbackBridge *>(c)->port->tryReadStream(p, w, &last);
      *tlast = last ? 1 : 0;
      return ok ? 1 : 0;
    };
    cb.try_write_stream = [](void *c, uint32_t p, uint32_t w, int tlast) {
      return static_cast<CallbackBridge *>(c)->port->tryWriteStream(
                 p, w, tlast != 0)
                 ? 1
                 : 0;
    };
    cb.try_read_cascade = [](void *c, void *d) {
      return static_cast<CallbackBridge *>(c)->port->tryReadCascade(d) ? 1 : 0;
    };
    cb.try_write_cascade = [](void *c, const void *d) {
      return static_cast<CallbackBridge *>(c)->port->tryWriteCascade(d) ? 1 : 0;
    };
    cb.put_char = [](void *c, char ch) {
      static_cast<CallbackBridge *>(c)->port->putChar(ch);
    };
  }
};

class LoadedCore : public CoreEngine {
public:
  LoadedCore(const aie_iss_api *api, aie_iss_core *core,
             std::unique_ptr<CallbackBridge> bridge)
      : api(api), core(core), bridge(std::move(bridge)) {}
  ~LoadedCore() override { api->destroy(core); }

  void reset() override { api->reset(core); }
  void setProgramCounter(uint32_t pc) override { api->set_pc(core, pc); }
  uint32_t getProgramCounter() const override { return api->get_pc(core); }

  CoreStepResult step() override {
    switch (api->step(core)) {
    case AIE_ISS_RETIRED:
      return CoreStepResult::Retired;
    case AIE_ISS_STALLED:
      return CoreStepResult::Stalled;
    case AIE_ISS_DONE:
      return CoreStepResult::Done;
    default:
      return CoreStepResult::Fault;
    }
  }

  std::string error() const override {
    const char *e = api->error(core);
    return e ? e : "unknown core fault";
  }

  bool readRegister(const std::string &name, void *data,
                    uint32_t size) const override {
    return api->read_register(core, name.c_str(), data, size) != 0;
  }
  bool writeRegister(const std::string &name, const void *data,
                     uint32_t size) override {
    return api->write_register(core, name.c_str(), data, size) != 0;
  }

private:
  const aie_iss_api *api;
  aie_iss_core *core;
  std::unique_ptr<CallbackBridge> bridge;
};

class LoadedFactory : public CoreEngineFactory {
public:
  LoadedFactory(void *handle, const aie_iss_api *api)
      : handle(handle), api(api) {}

  std::unique_ptr<CoreEngine> create(CoreISA isa,
                                     CoreMemoryPort &port) override {
    int abiIsa = toAbiIsa(isa);
    if (abiIsa < 0 || !api->supports_isa(abiIsa))
      return nullptr;
    auto bridge = std::make_unique<CallbackBridge>(port);
    aie_iss_core *core = api->create(abiIsa, &bridge->cb);
    if (!core)
      return nullptr;
    return std::make_unique<LoadedCore>(api, core, std::move(bridge));
  }

  std::string name() const override {
    const char *n = api->engine_name ? api->engine_name() : nullptr;
    return n ? n : "unnamed core engine";
  }

private:
  // The handle is never dlclose'd: engines may keep static state that is still
  // referenced while cores are torn down, and the process exits right after.
  void *handle;
  const aie_iss_api *api;
};

} // namespace

std::unique_ptr<CoreEngineFactory>
aiesim::loadCoreEngineFactory(const std::string &path, std::string &error) {
#ifdef _WIN32
  (void)path;
  error = "loadable core engines are not supported on this platform yet";
  return nullptr;
#else
  std::string soPath = path;
  if (soPath.empty())
    if (const char *env = std::getenv("AIE_SIM_CORE_ENGINE"))
      soPath = env;
  if (soPath.empty())
    if (const char *peano = std::getenv("PEANO_INSTALL_DIR"))
      soPath = std::string(peano) + "/lib/libaie-iss.so";
  if (soPath.empty()) {
    error = "no core engine: set AIE_SIM_CORE_ENGINE or PEANO_INSTALL_DIR";
    return nullptr;
  }

  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char *why = dlerror();
    error = "could not load core engine " + soPath + ": " + (why ? why : "?");
    return nullptr;
  }

  auto getApi =
      reinterpret_cast<aie_iss_get_api_fn>(dlsym(handle, "aie_iss_get_api"));
  if (!getApi) {
    error = soPath + " does not export aie_iss_get_api";
    return nullptr;
  }

  const aie_iss_api *api = getApi(AIE_ISS_ABI_VERSION);
  if (!api) {
    error = soPath + " does not implement core engine ABI version " +
            std::to_string(AIE_ISS_ABI_VERSION);
    return nullptr;
  }
  if (api->abi_version != AIE_ISS_ABI_VERSION || api->size < sizeof(*api)) {
    error = soPath + " reports an incompatible core engine ABI";
    return nullptr;
  }
  return std::make_unique<LoadedFactory>(handle, api);
#endif
}

//===----------------------------------------------------------------------===//
// Core module registers
//===----------------------------------------------------------------------===//
//
// The core-module register block, minus the core itself. A tile always has
// these whether or not a design puts code on it, so a design with no aie.core
// still reads them -- test_library.cpp's mlir_aie_print_tile_status does,
// before anything starts.
//
// Offsets and reset values are quoted from aie-rt's generated register
// database, not inferred: xaiemlgbl_params.h for AIE2 and xaie2pgbl_params.h
// for AIE2P. The two generations agree on CORE_CONTROL, CORE_STATUS,
// TIMER_LOW and TRACE_STATUS and disagree on every architectural register, so
// the layout is per-generation.

namespace {

struct CoreLayout {
  uint32_t control, status, timerLow, traceStatus;
};

CoreLayout layoutFor(Generation gen) {
  switch (gen) {
  case Generation::AIE2:
  case Generation::AIE2P:
    return {0x32000, 0x32004, 0x340F8, 0x340D8};
  }
  return {};
}

/// A half-open range of the core's memory-mapped architectural registers.
struct RegRange {
  uint32_t begin, end;
};

// The core's register file, projected into the tile's address space as a
// debug window: 32-bit values at stride 0x10, in contiguous runs with
// genuinely unmapped gaps between them. Ranges are the CORE_MODULE_CORE_*
// entries of aie-rt's register database, minus CORE_CONTROL/CORE_STATUS/
// PROCESSOR_BUS, which are modelled above rather than held at reset.
//
// The two generations lay this out differently AND collide -- 0x30C00 is R0
// on AIE2 but Q0 on AIE2P, 0x30E00 is M0 versus PC, 0x31000 is P0 versus R0,
// 0x31100 is PC versus R16 -- so a wrong table reads a valid wrong register
// instead of faulting. Keep them separate.

// 230 registers: BM accumulator partials, WL/WH vector halves, Q, E, the
// PC..SR control block, R0..S3.
constexpr RegRange kAIE2PCoreWindow[] = {
    {0x30000, 0x30500}, {0x30800, 0x30B00}, {0x30C00, 0x30C40},
    {0x30D00, 0x30DC0}, {0x30E00, 0x30EA0}, {0x31000, 0x314C0},
};

// 210 registers: AM accumulator partials (9 x 4 x 2, against AIE2P's
// 5 x 4 x 4), WL/WH, R0..S3, the PC..DP control block, Q.
constexpr RegRange kAIE2CoreWindow[] = {
    {0x30000, 0x30480},
    {0x30800, 0x30B00},
    {0x30C00, 0x310C0},
    {0x31100, 0x311A0},
    {0x31200, 0x31240},
};

/// CORE_CONTROL comes out of reset with RESET asserted and ENABLE clear:
/// *_CORE_CONTROL_RESET_DEFVAL is 0x1 at RESET_LSB 1, *_ENABLE_DEFVAL is 0x0
/// at ENABLE_LSB 0. Initialising this register to plain zero would report a
/// core that had already left reset.
constexpr uint32_t kCoreControlReset = 0x2;

bool inWindow(const RegRange *window, size_t n, uint32_t off) {
  for (size_t i = 0; i < n; ++i)
    if (off >= window[i].begin && off < window[i].end)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Offset -> engine register, for the scalar families
//===----------------------------------------------------------------------===//
//
// One indexed family: `count` registers named <prefix><i>, at stride 0x10.
struct ScalarRun {
  uint32_t begin;
  uint32_t count;
  const char *prefix;
};

// Every offset below is the CORE_MODULE_CORE_* value in aie-rt's parameter
// headers -- xaie2pgbl_params.h for AIE2P, xaiemlgbl_params.h for AIE2 -- and
// the families are contiguous at stride 0x10 there.
//
// The generations COLLIDE on three of these: 0x30E00 is AIE2P's PC and AIE2's
// m0, 0x31000 is AIE2P's r0 and AIE2's p0, 0x31100 is AIE2P's r16 and AIE2's
// PC. A shared table would return a valid wrong register rather than fault,
// so these stay separate. See [[aie2-aie2p-core-register-offsets-collide]].
constexpr ScalarRun kAIE2PScalarRuns[] = {
    {0x31000, 32, "r"},
    {0x31200, 8, "m"},
    {0x31400, 8, "p"},
    {0x31480, 4, "s"},
};
constexpr ScalarRun kAIE2ScalarRuns[] = {
    {0x30C00, 32, "r"},
    {0x30E00, 8, "m"},
    {0x31000, 8, "p"},
    {0x31080, 4, "s"},
};

/// A register with a name of its own rather than an index.
struct ScalarSingle {
  uint32_t off;
  const char *name;
};

// The control block, minus the parts that do not map. aie-rt's fc, cr1/cr2
// (cr on AIE2), sr and AIE2's dp are deliberately absent: llvm-aie splits
// those bits across separately named control registers (crSat, crRnd,
// crFPMask, srsSign0, ...), so one offset is an assembly of several and not a
// rename. Probed against a real engine rather than assumed -- sp/lr/ls/le/lc
// answer, fc/pc/dp/cr1/cr2/sr do not.
constexpr ScalarSingle kAIE2PScalarSingles[] = {
    {0x30E20, "sp"}, {0x30E30, "lr"}, {0x30E40, "ls"},
    {0x30E50, "le"}, {0x30E60, "lc"},
};
constexpr ScalarSingle kAIE2ScalarSingles[] = {
    {0x31120, "sp"}, {0x31130, "lr"}, {0x31140, "ls"},
    {0x31150, "le"}, {0x31160, "lc"},
};

constexpr uint32_t kAIE2PProgramCounter = 0x30E00;
constexpr uint32_t kAIE2ProgramCounter = 0x31100;

/// Slot stride of the whole window: one 32-bit value per 16 bytes, in every
/// family without exception.
constexpr uint32_t kSlotStride = 0x10;

} // namespace

aiesim::CoreRegisterMapping aiesim::coreScalarRegister(Generation gen,
                                                       uint32_t off) {
  const bool isAIE2P = gen == Generation::AIE2P;

  CoreRegisterMapping result;
  if (off == (isAIE2P ? kAIE2PProgramCounter : kAIE2ProgramCounter)) {
    result.isProgramCounter = true;
    return result;
  }

  // Only the first word of a slot names a register; the three bytes after it
  // are the same 32-bit value's tail, not a different register.
  if (off % kSlotStride != 0)
    return {};

  const ScalarSingle *singles =
      isAIE2P ? kAIE2PScalarSingles : kAIE2ScalarSingles;
  size_t nSingles = isAIE2P ? std::size(kAIE2PScalarSingles)
                            : std::size(kAIE2ScalarSingles);
  for (size_t i = 0; i < nSingles; ++i)
    if (singles[i].off == off) {
      std::snprintf(result.name, sizeof(result.name), "%s", singles[i].name);
      return result;
    }

  const ScalarRun *runs = isAIE2P ? kAIE2PScalarRuns : kAIE2ScalarRuns;
  size_t nRuns = isAIE2P ? std::size(kAIE2PScalarRuns)
                         : std::size(kAIE2ScalarRuns);
  for (size_t i = 0; i < nRuns; ++i) {
    const ScalarRun &run = runs[i];
    if (off < run.begin || off >= run.begin + run.count * kSlotStride)
      continue;
    unsigned index = (off - run.begin) / kSlotStride;
    std::snprintf(result.name, sizeof(result.name), "%s%u", run.prefix, index);
    return result;
  }

  return {};
}

const char *aiesim::coreRegisterOnOtherGeneration(Generation gen,
                                                  uint32_t off) {
  const bool isAIE2P = gen == Generation::AIE2P;
  const RegRange *own = isAIE2P ? kAIE2PCoreWindow : kAIE2CoreWindow;
  size_t ownSize =
      isAIE2P ? std::size(kAIE2PCoreWindow) : std::size(kAIE2CoreWindow);
  if (inWindow(own, ownSize, off))
    return nullptr;

  const RegRange *other = isAIE2P ? kAIE2CoreWindow : kAIE2PCoreWindow;
  size_t otherSize =
      isAIE2P ? std::size(kAIE2CoreWindow) : std::size(kAIE2PCoreWindow);
  if (!inWindow(other, otherSize, off))
    return nullptr;
  return isAIE2P ? "AIE2" : "AIE2P";
}

void aiesim::installCore(Tile &tile) {
  if (tile.getType() != TileType::Core)
    return;

  const CoreLayout layout = layoutFor(tile.getArray().device().generation);
  RegisterFile &regs = tile.regs();

  regs.claim(layout.control, layout.control + 4);
  regs.write(layout.control, kCoreControlReset);

  // Every CORE_STATUS field is *_DEFVAL 0, and with no core engine installed
  // none of them can leave its reset state.
  regs.onRead(layout.status, layout.status + 4,
              [](uint32_t) -> uint32_t { return 0; });

  // A free-running counter, so it is computed rather than reserved: reporting
  // a constant zero would make a host timing loop spin forever.
  Array &array = tile.getArray();
  regs.onRead(layout.timerLow, layout.timerLow + 4,
              [&array](uint32_t) -> uint32_t {
                return static_cast<uint32_t>(array.cycle());
              });

  // The architectural register window. This is the seam a core engine plugs
  // into: it projects the engine's state here, through
  // CoreEngine::readRegister. With no engine every register sits at its
  // *_REGISTER_VALUE_DEFVAL of 0, which is what a core that has never run
  // reads on hardware too. See docs/AIESimulator.md 4.4.
  const bool isAIE2P = tile.getArray().device().generation == Generation::AIE2P;
  const RegRange *window = isAIE2P ? kAIE2PCoreWindow : kAIE2CoreWindow;
  const size_t windowSize =
      isAIE2P ? std::size(kAIE2PCoreWindow) : std::size(kAIE2CoreWindow);
  const Generation gen = tile.getArray().device().generation;
  for (size_t i = 0; i < windowSize; ++i) {
    regs.reserve(window[i].begin, window[i].end,
                 "core architectural register; no core engine is installed, "
                 "so it holds its *_REGISTER_VALUE_DEFVAL of 0");
    // Layered over the reservation rather than replacing it: onRead wins for
    // reads, and the reservation keeps documenting why an unmapped offset in
    // this window is zero rather than unclaimed.
    regs.onRead(window[i].begin, window[i].end,
                [&tile, gen](uint32_t off) -> uint32_t {
                  CoreEngine *engine = tile.ensureCoreEngine();
                  if (!engine)
                    return 0;
                  CoreRegisterMapping reg = coreScalarRegister(gen, off);
                  if (reg.isProgramCounter)
                    return engine->getProgramCounter();
                  if (!reg.mapped())
                    return 0;
                  uint32_t value = 0;
                  // A name the engine does not know reads 0 rather than
                  // faulting: the window is wider than the scalar map, and a
                  // host walking it must not take the array down.
                  engine->readRegister(reg.name, &value, sizeof(value));
                  return value;
                });
  }

  regs.reserve(layout.traceStatus, layout.traceStatus + 4,
               "trace unit is not modelled; *_TRACE_STATUS_STATE_DEFVAL and "
               "_MODE_DEFVAL are both 0, so an untraced tile reads zero");
}
