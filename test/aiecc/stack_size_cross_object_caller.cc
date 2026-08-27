//===- stack_size_cross_object_caller.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_cross_object.mlir. entry_cross has a trivial
// frame of its own and calls helper_cross, defined in a *different*
// TU/object (stack_size_cross_object_callee.cc) -- the relocation this
// generates is against an undefined symbol with an unreliable NOTYPE, which
// StackSizeAnalysis can only recognize as a call once it has also scanned
// the callee's own object and found the symbol among what it defines. Both
// objects are listed in the core's link_files.

extern "C" void helper_cross(unsigned char *out);

extern "C" void entry_cross(unsigned char *out) { helper_cross(out); }
