//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

int main(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Vector Scalar Add Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  std::string elf_file = vm["elf"].as<std::string>();

  constexpr int IN_SIZE = 1024;
  constexpr int OUT_SIZE = 1024;

  // ------------------------------------------------------
  // Get device, load the elf
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the ELF (contains PDI + instructions + metadata)
  if (verbosity >= 1)
    std::cout << "Loading elf: " << elf_file << "\n";
  auto elf = xrt::elf(elf_file);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, elf);

  // Get a kernel handle
  std::string kernelName = vm["kernel"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::ext::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  xrt::bo bo_inA = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_SIZE * sizeof(int32_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint32_t *bufInA = bo_inA.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, 0, 0, bo_inA, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = i + 2;
    if (*(bufOut + i) != ref) {
      if (errors < 100) {
        std::cout << "Error in output " << *(bufOut + i) << " != " << ref
                  << std::endl;
      } else if (errors == 100) {
        std::cout << "..." << std::endl;
        std::cout << "[Errors truncated]" << std::endl;
      }
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut + i) << " == " << ref
                << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
