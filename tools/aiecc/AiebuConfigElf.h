//===- AiebuConfigElf.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exception-safe wrapper around aiebu's in-memory full-ELF assembler. aiebu
// reports errors by throwing, and aiecc is built without exceptions, so the
// call lives in its own library (aiecc-aiebu) compiled with them. Only plain
// standard-library types cross this boundary.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_AIEBU_CONFIG_ELF_H
#define AIECC_AIEBU_CONFIG_ELF_H

#include <string>
#include <utility>
#include <vector>

namespace xilinx::aiecc {

// Assemble an aie2 full ELF from `configJson` (the `aiebu-asm -t aie2_config`
// config). A file the config names is taken from `files` (name -> contents)
// if listed there and read from disk otherwise; aiebu takes the contents
// without copying them. On success fills `elf` and returns an empty string;
// otherwise returns aiebu's error message.
std::string assembleAie2ConfigElf(
    const std::string &configJson,
    std::vector<std::pair<std::string, std::vector<char>>> &&files,
    std::vector<char> &elf);

} // namespace xilinx::aiecc

#endif // AIECC_AIEBU_CONFIG_ELF_H
