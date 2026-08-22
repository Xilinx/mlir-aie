//===- resident.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The part of the program that is always in program memory. It holds the wait
// loop, anything the overlays call back into, and the call site itself.
//
// The core calls the overlay directly: slot.ld gives that symbol an absolute
// address, so the call compiles to a jump into the slot and the linker never
// looks for a body -- the body arrives at run time as a program-memory write.

#include <cstdint>

// Called by overlay 0 through --just-symbols. Exists to prove an overlay can
// reach resident code rather than having to be self-contained.
extern "C" __attribute__((noinline, used, retain)) int32_t ovl_bias(void) {
  return 100;
}

// Spin until the host has finished writing the slot. The pointer must be
// volatile: this has to stay a real fetch loop, and it has to be here in the
// resident rather than in the slot being overwritten. `retain` keeps these two
// alive under --gc-sections: the core calls them, but nothing inside this
// translation unit does.
extern "C" __attribute__((used, retain)) void ovl_wait(volatile int32_t *f) {
  while (*f == 0)
    ;
  *f = 0;
}
