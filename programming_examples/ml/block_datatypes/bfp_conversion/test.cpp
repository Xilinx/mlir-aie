//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <sys/types.h>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cxxopts.hpp"
#include "test_utils.h"

#include "../helper.h"
#include "common.h"

#include <stdfloat>

// Clangd fix, remove
#ifdef _CLANGD
namespace std {
using bfloat16_t = float;
} // namespace std
#endif

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("BFP Conversion test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  // int do_verify = vm["verify"].as<bool>();
  // int n_iterations = vm["iters"].as<int>();
  // int n_warmup_iterations = vm["warmup"].as<int>();
  // int trace_size = vm["trace_sz"].as<int>();

  const int numberFloats = 64;
  const int bfpBytesSize = numberFloats * 1.125;

  // Load instruction sequence
  std::vector<uint32_t> instr =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::ranges::find_if(
      xkernels, [node, verbosity](xrt::xclbin::kernel &k) {
        auto name = k.get_name();
        if (verbosity >= 1) {
          std::cout << "Name: " << name << std::endl;
        }
        return name.rfind(node, 0) == 0;
      });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  auto boInstr = xrt::bo(device, instr.size() * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto boInA = xrt::bo(device, numberFloats * sizeof(std::bfloat16_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto boInB = xrt::bo(device, numberFloats * sizeof(std::bfloat16_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto boOut = xrt::bo(device, bfpBytesSize * sizeof(int8_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // ------------------------------------------------------
  // Generate data for buffers
  // ------------------------------------------------------
  // Generate random floats
  std::mt19937 rng(0);

  float floatA[numberFloats];
  float floatB[numberFloats];

  std::ranges::transform(floatA, floatA, [&](float _) {
    return generateRandomFloatingPoint(rng, -5, 5);
  });
  std::ranges::transform(floatB, floatB, [&](float _) {
    return generateRandomFloatingPoint(rng, -5, 5);
  });

  std::bfloat16_t bfloatA[numberFloats];
  std::bfloat16_t bfloatB[numberFloats];

  std::ranges::transform(
      floatA, bfloatA, [](float f) { return static_cast<std::bfloat16_t>(f); });
  std::ranges::transform(
      floatB, bfloatB, [](float f) { return static_cast<std::bfloat16_t>(f); });

  // ------------------------------------------------------
  // Write data into buffers
  // ------------------------------------------------------
  std::bfloat16_t *bufInA = boInA.map<std::bfloat16_t *>();
  memcpy(bufInA, bfloatA, (numberFloats * sizeof(std::bfloat16_t)));

  std::bfloat16_t *bufInB = boInB.map<std::bfloat16_t *>();
  memcpy(bufInB, bfloatB, (numberFloats * sizeof(std::bfloat16_t)));

  void *bufInstr = boInstr.map<void *>();
  memcpy(bufInstr, instr.data(), instr.size() * sizeof(int));

  boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boInA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boInB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Run kernel
  // ------------------------------------------------------
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, boInstr, instr.size(), boInA, boInB, boOut);
  run.wait();

  boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint8_t *bufOut = boOut.map<uint8_t *>();

  // ------------------------------------------------------
  // Check output
  // ------------------------------------------------------

  // Calculate the expected output with fp
  float matrixSize = std::sqrt(numberFloats);
  if (matrixSize != std::floor(matrixSize)) {
    std::cout << "Matrix size is not square, cannot run test.\n";
    return 1;
  }
  // matrixMultiply(floatA, floatB, expectedResult, matrixSize);
  std::vector<float> floatAVec(floatA, floatA + numberFloats);
  std::vector<float> floatBVec(floatB, floatB + numberFloats);
  std::vector<float> expectedResultVec(numberFloats);
  matmul_common::matmul<float, float, float>(matrixSize, matrixSize, matrixSize,
                                             floatAVec, floatBVec,
                                             expectedResultVec, false, false);

  printBfp16ebs8Array(numberFloats * 1.125,
                      std::vector(bufOut, bufOut + bfpBytesSize));
  auto outputTransformed = bfp16ebs8ToFloat(bfpBytesSize, bufOut, 0);

  int errors = 0;

  for (uint32_t i = 0; i < numberFloats; i++) {
    if (i % 8 == 0) {
      std::cout << "Block " << i / 8 << "\n";
    }
    // Note that this nearly equal function parameters are handpicked for this
    // particular example and do not reflect how the general case should be
    // handled for any bfp type.
    if (!test_utils::nearly_equal(outputTransformed[i], expectedResultVec[i],
                                  0.25, 3.5)) {
      std::cout << "Error in output " << outputTransformed[i]
                << " != " << expectedResultVec[i] << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << outputTransformed[i]
                  << " ~= " << expectedResultVec[i] << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
