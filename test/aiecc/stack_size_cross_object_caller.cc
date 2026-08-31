//===- stack_size_cross_object_caller.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_cross_object.mlir. entry_cross has a small frame
// and calls helper_cross, which stack_size_cross_object_callee.cc defines in
// another object. The call relocates against an undefined symbol of type
// NOTYPE, so StackSizeAnalysis counts it as a call once it scans the object of
// the callee and finds the symbol there. The core lists both objects in its
// link_files.

extern "C" void helper_cross(unsigned char *out);

extern "C" void entry_cross(unsigned char *out) { helper_cross(out); }
