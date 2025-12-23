//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cxxopts.hpp"
#include "test_utils.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using INOUT0_DATATYPE = std::bfloat16_t;
using INOUT1_DATATYPE = std::bfloat16_t;
#endif

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
template <typename T>
int verify(int size, int tile_size, std::vector<T> A, std::vector<T> B,
           int verbosity) {

  int errors = 0;
  T max_val = A[0];
  std::vector<T> RefVec(size);

  for (uint32_t i = 1; i < A.size(); i++) {
    A[i] = (T)(A[i]);
    T val = A[i];
    if (val > max_val) {
      max_val = val;
    }
  }

  for (uint32_t t = 0; t < size; t += tile_size) {
    float running = 0.0;
    for (uint32_t i = 0; i < tile_size; i++) {
      float ez = (float)(exp(A[t + i] - max_val));
      running += ez;
      RefVec[t + i] = (T)exp(A[t + i] - max_val);
    }

    for (uint32_t i = 0; i < tile_size; i++) {
      RefVec[t + i] /= (T)running;
    }
  }

  for (uint32_t i = 0; i < size; i++) {

    if (!test_utils::nearly_equal(RefVec[i], B[i], 0.04, 0.001)) {
      std::cout << "Error in output " << B[i] << " != " << RefVec[i]
                << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << B[i] << " == " << RefVec[i]
                  << std::endl;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Softmax Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  options.add_options()("npu", "Select NPU",
                        cxxopts::value<int>()->default_value("2"));

  test_utils::parse_options(argc, argv, options, vm);

  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int dev = vm["npu"].as<int>();

  int TILE_SIZE = 1024;
  int INOUT0_VOLUME = 262144;        // Input
  int INOUT1_VOLUME = INOUT0_VOLUME; // Output

  size_t INOUT0_SIZE = INOUT0_VOLUME * sizeof(INOUT0_DATATYPE);
  size_t INOUT1_SIZE = INOUT1_VOLUME * sizeof(INOUT1_DATATYPE);

  size_t OUT_SIZE = INOUT1_SIZE + trace_size;

  srand(42);

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  std::cout << "Running with device: " << device << std::endl;

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inout0 =
      xrt::bo(device, INOUT0_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inout1 =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  // Assumes trace will only be added to inout1

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize Inout buffer 0 with ascending bfloat16 raw patterns
  // All of them ...
  INOUT0_DATATYPE *bufInOut0 = bo_inout0.map<INOUT0_DATATYPE *>();
  std::vector<INOUT0_DATATYPE> AVec(INOUT0_VOLUME);
  for (int i = 0; i < INOUT0_VOLUME; i++) {
    if (dev == 1) {
      // NPU1: Use bfloat16 values in range [4.0, 4.0]
      AVec[i] = test_utils::random_bfloat16_t((std::bfloat16_t)8.0,
                                              (std::bfloat16_t)-4.0);
    } else if (dev == 2) {
      // NPU2: Use bfloat16 values in range [-512.0, 512.0]
      AVec[i] = test_utils::random_bfloat16_t((std::bfloat16_t)1024.0,
                                              (std::bfloat16_t)-512.0);
    }
  }
  memcpy(bufInOut0, AVec.data(), (AVec.size() * sizeof(INOUT0_DATATYPE)));

  // Initialize Inout buffer 1 with zeros
  char *bufInOut1 = bo_inout1.map<char *>();
  memset(bufInOut1, 0, OUT_SIZE); // Zeroes out INOUT1_VOLUME + trace_size

  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }

    // Run kernel
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inout0, bo_inout1);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_inout1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Copy output results and verify they are correct
    std::vector<INOUT1_DATATYPE> BVec(INOUT1_VOLUME);

    memcpy(BVec.data(), bufInOut1, (BVec.size() * sizeof(INOUT1_DATATYPE)));
    if (do_verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      errors = verify(INOUT0_VOLUME, TILE_SIZE, AVec, BVec, verbosity);
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0
    if (trace_size > 0) {
      test_utils::write_out_trace(((char *)bufInOut1) + INOUT1_SIZE, trace_size,
                                  vm["trace_file"].as<std::string>());
    }

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
