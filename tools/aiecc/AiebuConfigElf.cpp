//===- AiebuConfigElf.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AiebuConfigElf.h"

#include <aiebu/aiebu_assembler.h>

#include <exception>

std::string xilinx::aiecc::assembleAie2ConfigElf(
    const std::string &configJson,
    const std::vector<std::pair<std::string, std::vector<char>>> &files,
    std::vector<char> &elf) {
  try {
    aiebu::file_artifact artifact;
    for (const auto &[name, contents] : files)
      artifact.add_vfile(name, contents);
    aiebu::aiebu_assembler assembler(
        aiebu::aiebu_assembler::buffer_type::aie2_config,
        std::vector<char>(configJson.begin(), configJson.end()), artifact,
        /*flags=*/{});
    elf = assembler.get_elf();
    return {};
  } catch (const std::exception &e) {
    // An empty return means success, so never pass an empty message through.
    std::string msg = e.what();
    return msg.empty() ? "aiebu error" : msg;
  } catch (...) {
    return "unknown aiebu error";
  }
}
