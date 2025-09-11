//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("add_one_objFifo");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt:elf ctx_elf{vm["xclbin"].as<std::string>()};
  xrt::hw_context context;
  try {
    context = xrt::hw_context(device, ctx_elf);
  }

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, vm["kernel"].as<std::string>());

  xrt::bo bo_inA = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
  xrt::bo bo_inB = xrt::ext::bo{device, IN_SIZE * sizeof(int32_t)};
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
  auto run = kernel(opcode, 0, 0, bo_inA, bo_inB, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < 64; i++) {
    uint32_t ref = i + 42;
    if (*(bufOut + i) != ref) {
      std::cout << "Error in output " << *(bufOut + i) << " != " << ref
                << std::endl;
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
