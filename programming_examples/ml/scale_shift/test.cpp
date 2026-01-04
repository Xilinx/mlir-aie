//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
using INOUT2_DATATYPE = std::bfloat16_t;
using INOUT3_DATATYPE = std::bfloat16_t;
#endif

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
template <typename T>
int verify(int size, std::vector<T> A, std::vector<T> B, std::vector<T> C,
           std::vector<T> D, int verbosity) {
  int errors = 0;
  for (uint32_t i = 0; i < size; i++) {
    T ref = A[i] * B[i] + C[i];
    if (!test_utils::nearly_equal(ref, D[i], 0.002)) {
      if (verbosity >= 1) {
        std::cout << "Error in output " << D[i] << " != " << ref << " from "
                  << A[i] << " * " << B[i] << " + " << C[i] << std::endl;
      }
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << D[i] << " == " << ref << std::endl;
    }
  }
  return errors;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Eltwise Mul Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  // ------------------------------------------------------
  // Configure this to match your design's buffer size
  // ------------------------------------------------------
  int INOUT0_VOLUME = 65536;
  int INOUT1_VOLUME = INOUT0_VOLUME;
  int INOUT2_VOLUME = INOUT0_VOLUME;
  int INOUT3_VOLUME = INOUT0_VOLUME;

  size_t INOUT0_SIZE = INOUT0_VOLUME * sizeof(INOUT0_DATATYPE);
  size_t INOUT1_SIZE = INOUT1_VOLUME * sizeof(INOUT1_DATATYPE);
  size_t INOUT2_SIZE = INOUT2_VOLUME * sizeof(INOUT2_DATATYPE);
  size_t INOUT3_SIZE = INOUT3_VOLUME * sizeof(INOUT3_DATATYPE);

  size_t OUT_SIZE = INOUT3_SIZE + trace_size;

  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
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
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inout0 =
      xrt::bo(device, INOUT0_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inout1 =
      xrt::bo(device, INOUT1_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_inout2 =
      xrt::bo(device, INOUT1_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  // Assumes trace will only be added to inout3
  auto bo_inout3 =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize Inout buffer 0
  INOUT0_DATATYPE *bufInOut0 = bo_inout0.map<INOUT0_DATATYPE *>();
  std::vector<INOUT0_DATATYPE> AVec(INOUT0_VOLUME);
  for (int i = 0; i < INOUT0_VOLUME; i++)
    AVec[i] = INOUT0_DATATYPE(4.0f);
  memcpy(bufInOut0, AVec.data(), (AVec.size() * sizeof(INOUT0_DATATYPE)));

  // Initialize Inout buffer 1
  INOUT1_DATATYPE *bufInOut1 = bo_inout1.map<INOUT0_DATATYPE *>();
  std::vector<INOUT1_DATATYPE> BVec(INOUT1_VOLUME);
  for (int i = 0; i < INOUT1_VOLUME; i++)
    BVec[i] = INOUT1_DATATYPE(3.35f);
  memcpy(bufInOut1, BVec.data(), (BVec.size() * sizeof(INOUT1_DATATYPE)));

  // Initialize Inout buffer 2
  INOUT1_DATATYPE *bufInOut2 = bo_inout2.map<INOUT2_DATATYPE *>();
  std::vector<INOUT2_DATATYPE> CVec(INOUT2_VOLUME);
  for (int i = 0; i < INOUT2_VOLUME; i++)
    CVec[i] = INOUT2_DATATYPE(0.77f);
  memcpy(bufInOut2, CVec.data(), (CVec.size() * sizeof(INOUT2_DATATYPE)));

  // Initialize Inout buffer 3
  char *bufInOut3 = bo_inout3.map<char *>();
  std::vector<INOUT3_DATATYPE> DVec(INOUT3_VOLUME);
  memset(bufInOut3, 0, OUT_SIZE); // Zeroes out INOUT2_VOLUME + trace_size

  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

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
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inout0, bo_inout1,
                      bo_inout2, bo_inout3);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_inout3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Copy output results and verify they are correct
    memcpy(DVec.data(), bufInOut3, (DVec.size() * sizeof(INOUT3_DATATYPE)));
    if (do_verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      errors = verify(INOUT0_VOLUME, AVec, BVec, CVec, DVec, verbosity);
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
      test_utils::write_out_trace(((char *)bufInOut3) + INOUT3_SIZE, trace_size,
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
