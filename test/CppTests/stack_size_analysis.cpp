//===- stack_size_analysis.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StackSizeAnalysis.h"

#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: stack_size_analysis <elf>\n";
    return 1;
  }
  auto result = xilinx::aiecc::computeStackRequirement(argv[1], {});
  if (!result.bytes) {
    llvm::errs() << result.error << "\n";
    return 1;
  }
  if (!result.unmeasured.empty()) {
    llvm::errs() << "unmeasured functions in fixture\n";
    return 1;
  }
  llvm::outs() << *result.bytes << "\n";
  return 0;
}
