// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "StackSizeAnalysis.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  auto demand = xilinx::aiecc::measureDataSectionDemand(argv[1]);
  auto bytes = xilinx::aiecc::measureDataSectionBytes(argv[1]);
  if (!demand || !bytes)
    return 1;
  llvm::outs() << "size: " << demand->size << ", alignment: " << demand->align
               << ", section bytes: " << *bytes << "\n";
  auto banks = xilinx::aiecc::measureBankSectionBytes(argv[1], 4);
  for (unsigned i = 0; i < banks.size(); ++i)
    llvm::outs() << "bank " << i << ": size: " << banks[i].size
                 << ", alignment: " << banks[i].align << "\n";
  return 0;
}
