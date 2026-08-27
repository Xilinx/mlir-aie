//===- stack_size_indirect_call_multi_global_kernel.cc -----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Same shape as stack_size_indirect_call_kernel.cc, plus a second, unrelated
// global (g_unrelated). Without -fdata-sections, g_dispatch and g_unrelated
// would share one .data section, making that section's owning symbol
// ambiguous and silently dropping the "target_fn's address escapes into
// g_dispatch" record the indirect-call inference depends on.

extern "C" void target_fn(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

typedef void (*fn_ptr_t)(unsigned char *);
extern "C" fn_ptr_t g_dispatch = &target_fn;
extern "C" int g_unrelated = 1;

extern "C" void indirect_caller(unsigned char *out) { g_dispatch(out); }
