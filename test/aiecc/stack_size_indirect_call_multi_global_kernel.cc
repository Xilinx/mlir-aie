//===- stack_size_indirect_call_multi_global_kernel.cc ----------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The same shape as stack_size_indirect_call_kernel.cc, plus a second,
// unrelated global, g_unrelated. Under -fdata-sections each global gets its
// own section, and the analysis attributes the escape of target_fn to
// g_dispatch. Inside one shared .data section that attribution is ambiguous,
// and the analysis drops the record that the indirect-call inference needs.

extern "C" void target_fn(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

typedef void (*fn_ptr_t)(unsigned char *);
extern "C" fn_ptr_t g_dispatch = &target_fn;
extern "C" int g_unrelated = 1;

extern "C" void indirect_caller(unsigned char *out) { g_dispatch(out); }
