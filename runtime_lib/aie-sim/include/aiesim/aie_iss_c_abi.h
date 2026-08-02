//===- aie_iss_c_abi.h ------------------------------------------*- C -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The ABI a loadable AIE core engine ("ISS") exports, and the only thing the
// array model and the ISS agree on.
//
// The ISS is expected to be built as part of llvm-aie, where the AIE ISA is
// defined: it needs the MC layer (disassembler, instruction info, subtarget)
// to decode bundles, and instruction semantics are a property of the backend,
// not of mlir-aie. The array model must NOT be built against LLVM headers or
// linked against a specific libLLVM, because mlir-aie and Peano are separately
// versioned and separately built, and a Peano distribution ships libLLVM.so
// but no LLVM headers.
//
// So the boundary is a plain C ABI, resolved with dlopen/dlsym at run time.
// The array model vendors this header; llvm-aie vendors an identical copy.
// Compatibility is checked with AIE_ISS_ABI_VERSION, which must be bumped for
// any change that is not a pure addition at the end of a struct that carries
// its own size.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ISS_C_ABI_H
#define AIE_ISS_C_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AIE_ISS_ABI_VERSION 1u

/* --- Values returned by aie_iss_step. Mirrors aiesim::CoreStepResult. --- */
#define AIE_ISS_RETIRED 0
#define AIE_ISS_STALLED 1
#define AIE_ISS_DONE 2
#define AIE_ISS_FAULT 3

/* --- Target ISA selector. --- */
#define AIE_ISS_ISA_AIE2 0
#define AIE_ISS_ISA_AIE2P 1
#define AIE_ISS_ISA_AIE2PS 2

/*
 * Callbacks the host (the array model) provides. Every memory access, lock
 * operation, stream access and character the core makes comes back out
 * through these, so the ISS models the DATAPATH only and holds no memory of
 * its own beyond architectural registers.
 *
 * `ctx` is the opaque pointer passed to aie_iss_create.
 *
 * Functions returning int use 1 for success and 0 for "not available now":
 *  - read/write: 0 means the address is unmapped, which the ISS reports as a
 *    fault. It must never fabricate data for an unmapped access.
 *  - lock/stream/cascade: 0 means the resource is not ready, which the ISS
 *    reports as AIE_ISS_STALLED without retiring the instruction.
 */
typedef struct aie_iss_host_callbacks {
  /* Size of this struct, for forward compatibility. */
  size_t size;
  void *ctx;

  int (*read)(void *ctx, uint32_t addr, void *data, uint32_t size);
  int (*write)(void *ctx, uint32_t addr, const void *data, uint32_t size);

  int (*try_acquire_lock)(void *ctx, uint32_t lock_id, int32_t value);
  void (*release_lock)(void *ctx, uint32_t lock_id, int32_t value);

  int (*try_read_stream)(void *ctx, uint32_t port, uint32_t *word,
                         int *tlast);
  int (*try_write_stream)(void *ctx, uint32_t port, uint32_t word, int tlast);

  int (*try_read_cascade)(void *ctx, void *data512);
  int (*try_write_cascade)(void *ctx, const void *data512);

  void (*put_char)(void *ctx, char c);
} aie_iss_host_callbacks;

/* Opaque handle to one simulated core. */
typedef struct aie_iss_core aie_iss_core;

/*
 * The vtable an engine shared object exports. The single exported symbol is
 * `aie_iss_get_api`; everything else is reached through the returned struct,
 * so adding entry points does not require new dynamic symbols.
 */
typedef struct aie_iss_api {
  size_t size;
  uint32_t abi_version; /* must equal AIE_ISS_ABI_VERSION */

  /* Free-form identification for diagnostics, e.g. "peano-iss 21.0.0git". */
  const char *(*engine_name)(void);

  /* Returns 1 if this engine can execute `isa`. */
  int (*supports_isa)(int isa);

  aie_iss_core *(*create)(int isa, const aie_iss_host_callbacks *callbacks);
  void (*destroy)(aie_iss_core *core);

  void (*reset)(aie_iss_core *core);
  void (*set_pc)(aie_iss_core *core, uint32_t pc);
  uint32_t (*get_pc)(const aie_iss_core *core);

  /* Executes at most one bundle; returns one of AIE_ISS_*. */
  int (*step)(aie_iss_core *core);

  /*
   * Reason for the last AIE_ISS_FAULT. The returned pointer stays valid until
   * the next call on the same core.
   */
  const char *(*error)(const aie_iss_core *core);

  int (*read_register)(const aie_iss_core *core, const char *name, void *data,
                       uint32_t size);
  int (*write_register)(aie_iss_core *core, const char *name, const void *data,
                        uint32_t size);

  /*
   * Every distinct opcode this core reached, with whether the engine had
   * semantics for it. Calls `sink` once per opcode and returns how many times
   * it did. `name` stays valid for the duration of the call only.
   *
   * This is what makes the semantics gap a NUMBER instead of whatever the run
   * happened to stop on: the engine accumulates the set as it executes, so one
   * run reports everything it reached rather than only the first gap. May be
   * NULL on an engine that does not track it -- check before calling.
   */
  uint32_t (*opcode_coverage)(const aie_iss_core *core, void *ctx,
                              void (*sink)(void *ctx, const char *name,
                                           int modelled));
} aie_iss_api;

/*
 * The one symbol an engine shared object must export.
 * Returns NULL if the engine cannot serve the requested ABI version.
 */
const aie_iss_api *aie_iss_get_api(uint32_t requested_abi_version);

typedef const aie_iss_api *(*aie_iss_get_api_fn)(uint32_t);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AIE_ISS_C_ABI_H */
