//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Stand-in host binary for the add_aie_design/add_aie_run_test fixture: it
// gives add_executable() something to compile and $<TARGET_FILE:> something to
// resolve. helpers.lit only configures, so this is never run.
int main() { return 0; }
