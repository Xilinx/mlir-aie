//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stand-in host binary for the add_aie_design/add_aie_run_test fixture. It
// exists so add_executable() has something to compile and $<TARGET_FILE:> has
// something to resolve; helpers.lit never runs it.
//
//===----------------------------------------------------------------------===//

int main() { return 0; }
