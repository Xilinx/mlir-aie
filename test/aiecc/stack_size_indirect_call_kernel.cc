//===- stack_size_indirect_call_kernel.cc -----------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_indirect_call.mlir. An indirect call loads its
// target from a register, so the call site carries no relocation that names a
// callee. The analysis combines two other facts instead. The address of
// target_fn escapes into the function-pointer variable g_dispatch, which
// relocates the storage of g_dispatch against target_fn. And indirect_caller
// loads g_dispatch, which relocates the code of indirect_caller against
// g_dispatch. Together the two facts connect indirect_caller to target_fn.
// g_dispatch is a plain global, neither const nor local, so that -O0 keeps the
// call indirect inside this translation unit.

extern "C" void target_fn(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

typedef void (*fn_ptr_t)(unsigned char *);
extern "C" fn_ptr_t g_dispatch = &target_fn;

extern "C" void indirect_caller(unsigned char *out) { g_dispatch(out); }
