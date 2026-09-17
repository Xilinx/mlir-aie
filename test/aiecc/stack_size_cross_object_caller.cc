//===- stack_size_cross_object_caller.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_cross_object.mlir. entry_cross has a small frame
// and calls helper_cross, which stack_size_cross_object_callee.cc defines in
// another object. The core lists both objects in its link_files, so the linked
// ELF holds both frames and the call between them.

extern "C" void helper_cross(unsigned char *out);

extern "C" void entry_cross(unsigned char *out) { helper_cross(out); }
