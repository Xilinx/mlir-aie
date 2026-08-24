//===- stub.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The upper-granule park, for ping-pong.
//
// The core returns from a slot into the resident, which lives at address 0. So
// on the phase where the host writes the *lower* granule, the core would be
// executing the very granule being written -- and a write to the granule the
// core is fetching from is silently dropped about half the time.
//
// This gives the core somewhere else to be. It is written once during setup and
// never overwritten, so while the core waits here the whole lower granule is
// free to be rewritten.
//
// Mechanically it is an ordinary overlay: linked at a fixed address against the
// resident's symbols, written with a blockwrite. What differs is its lifetime --
// one write, then it is live for the rest of the run.
//
// The handshake it implements:
//
//   core                                   host
//   ----                                   ----
//   parked = 1                             npu.maskpoll parked == 1  (blocks)
//   spin on flag                           write the lower granule
//                                          flag = 1
//   sees flag, clears it
//   parked = 0
//   returns to the resident
//
// `parked` is what makes the wait expressible at all. Without it the host has
// no way to know the core has reached this code -- a runtime sequence can wait
// on a DMA, and otherwise only on memory.

#include <cstdint>

// Buffers the IRON design declares; addresses come from the resident's symbol
// table when this is linked, so nothing here is hardcoded.
extern "C" int32_t flag[];
extern "C" int32_t ovl_parked[];

extern "C" void overlay_stub(void) {
  volatile int32_t *parked = ovl_parked;
  volatile int32_t *f = flag;

  // Announce arrival before waiting. The host is already polling this, and it
  // must not observe the announcement until the core is genuinely here --
  // hence volatile, and hence the store coming first.
  *parked = 1;
  while (*f == 0)
    ;
  *f = 0;
  // Cleared last: the host writes the lower granule only between seeing 1 here
  // and setting the flag, so clearing before returning cannot race with it.
  *parked = 0;
}
