//===- resident.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The part of the program that is always in program memory: the wait loop, the
// call site, and anything an overlay calls back into.
//
// The core calls the overlay directly. slot.ld gives that symbol an absolute
// address, so the call compiles to a jump into the slot and the linker never
// looks for a body -- the body arrives at run time as a program-memory write.

#include <cstdint>

// A helper an overlay may call, so an overlay need not be self-contained.
//
// `retain` is load-bearing and its absence is dangerous rather than merely
// inconvenient. Nothing in the resident calls this: the only caller is an
// overlay, which links separately and afterwards, so the resident's own link
// graph has no edge to it and --gc-sections collects it. The overlay was linked
// against pass 1's addresses, so it would then jump to whatever moved into that
// address in pass 2 -- and `pm.py check` cannot notice a symbol that is simply
// absent unless it is told which symbols the overlays import, which is why it
// now is.
//
// PM_OVERLAY_NO_RETAIN drops it, so a test can show the build catching that.
#ifdef PM_OVERLAY_NO_RETAIN
#define RESIDENT_HELPER __attribute__((noinline, used))
#else
#define RESIDENT_HELPER __attribute__((noinline, used, retain))
#endif

extern "C" RESIDENT_HELPER int32_t ovl_bias(void) { return 100; }

// Spin until the host has finished writing the slot. The pointer must be
// volatile: this has to stay a real fetch loop, and it has to live here in the
// resident rather than in the slot being overwritten. `retain` keeps it alive
// under --gc-sections: the core calls it, but nothing inside this translation
// unit does.
extern "C" __attribute__((used, retain)) void ovl_wait(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}
