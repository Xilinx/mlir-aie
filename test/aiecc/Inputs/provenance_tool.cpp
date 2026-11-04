// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Provenance.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc == 4 && llvm::StringRef(argv[1]) == "--compiler") {
    llvm::outs() << llvm::json::Value(
        xilinx::aiecc::coreCompilerProvenance(argv[2], argv[3], "--version"));
    return 0;
  }
  if (argc != 3)
    return 2;
  auto input = llvm::MemoryBuffer::getFile(argv[1]);
  if (!input)
    return 2;
  auto stamped = xilinx::aiecc::stampElfProvenance(
      (*input)->getBuffer(), R"({"schema":1,"producer":"mlir-aie/aiecc"})");
  if (!stamped) {
    llvm::errs() << llvm::toString(stamped.takeError()) << "\n";
    return 1;
  }
  std::error_code ec;
  llvm::raw_fd_ostream output(argv[2], ec);
  if (ec)
    return 2;
  output.write(stamped->data(), stamped->size());
  return 0;
}
