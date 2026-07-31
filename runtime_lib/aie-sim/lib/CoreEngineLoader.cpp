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

// Core installation is the last phase of the simulator (docs/AIESimulator.md
// section 7). Until it lands, a design with no core code still simulates and a
// design that enables a core gets a diagnostic from the core-control register
// handler rather than silently doing nothing.
void aiesim::installCore(Tile &) {}
