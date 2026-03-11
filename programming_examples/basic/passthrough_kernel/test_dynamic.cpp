//===- test_dynamic.cpp - Dynamic TXN generation test -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Demonstrates dynamic TXN binary generation at runtime. Instead of loading
// a pre-compiled insts.bin, this uses a C++ function (generated from the MLIR
// runtime sequence via aie-translate --aie-generate-txn-cpp) that calls the
// TxnEncoding library to produce the TXN instruction binary in memory.
//
// The same static XCLBIN is used - only the instruction stream is generated
// at runtime. This enables future dynamic parameterization (variable problem
// sizes) without recompilation of the XCLBIN.
//
//===----------------------------------------------------------------------===//

// Include the generated TXN function
#include "generated_txn.h"

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"

using DATATYPE = std::uint8_t;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Dynamic Passthrough Kernel Test");
  options.add_options()("x,xclbin", "XCLBIN file",
                        cxxopts::value<std::string>())(
      "k,kernel", "Kernel name",
      cxxopts::value<std::string>()->default_value("MLIR_AIE"))(
      "v,verbosity", "Verbosity level",
      cxxopts::value<int>()->default_value("0"))("h,help", "Print usage");

  auto vm = options.parse(argc, argv);
  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string xclbinPath = vm["xclbin"].as<std::string>();
  std::string kernelName = vm["kernel"].as<std::string>();
  int verbosity = vm["verbosity"].as<int>();

  // Generate TXN instructions at runtime using the generated function.
  // This produces the exact same binary as insts.bin would contain.
  std::vector<uint32_t> instr_v = generate_txn_sequence();

  if (verbosity >= 1)
    std::cout << "Generated TXN instruction count: " << instr_v.size() << "\n";

  // Initialize XRT
  xrt::device device;
  xrt::kernel kernel;
  test_utils::init_xrt_load_kernel(device, kernel, verbosity, xclbinPath,
                                   kernelName);

  constexpr int VOLUME = IN1_SIZE / sizeof(DATATYPE);

  // Set up buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, VOLUME * sizeof(DATATYPE),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, VOLUME * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_tmp = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_ctrl = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace =
      xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Copy instruction stream
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  // Initialize buffers
  DATATYPE *bufIn = bo_in.map<DATATYPE *>();
  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  for (int i = 0; i < VOLUME; i++)
    bufIn[i] = static_cast<DATATYPE>(i);
  memset(bufOut, 0, VOLUME * sizeof(DATATYPE));

  // Sync to device
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run kernel
  if (verbosity >= 1)
    std::cout << "Running kernel...\n";

  auto start = std::chrono::high_resolution_clock::now();
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out, bo_tmp,
                    bo_ctrl, bo_trace);
  run.wait();
  auto stop = std::chrono::high_resolution_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Verify results
  int errors = 0;
  for (int i = 0; i < VOLUME; i++) {
    if (bufOut[i] != bufIn[i]) {
      if (verbosity >= 1)
        std::cout << "Error at " << i << ": " << (int)bufOut[i]
                  << " != " << (int)bufIn[i] << std::endl;
      errors++;
    }
  }

  float npu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();
  std::cout << "NPU time: " << npu_time << "us\n";

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
