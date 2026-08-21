//===- stack_size_indirect_call_kernel.cc -------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_indirect_call.mlir. An indirect call leaves no
// relocation at the call site naming a callee (the target is a runtime
// register load), so the analysis has to infer the edge from the reverse
// direction instead: target_fn's address escapes into the function-pointer
// variable g_dispatch (a relocation in g_dispatch's own storage pointing at
// target_fn), and indirect_caller separately loads g_dispatch and calls
// through it (a relocation in indirect_caller's code referencing
// g_dispatch). Neither fact alone says indirect_caller can reach target_fn;
// only combining them does. g_dispatch is a plain (non-const, externally
// visible) global specifically so -O0 cannot see enough of this one
// translation unit to devirtualize the call into a direct one.

extern "C" void target_fn(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

typedef void (*fn_ptr_t)(unsigned char *);
extern "C" fn_ptr_t g_dispatch = &target_fn;

extern "C" void indirect_caller(unsigned char *out) { g_dispatch(out); }
