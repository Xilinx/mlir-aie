//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using INOUT0_DATATYPE = std::uint32_t;
using INOUT1_DATATYPE = std::uint32_t;
using INOUT2_DATATYPE = std::uint32_t;
#endif

namespace po = boost::program_options;

// Verify results
template<typename Tout>
int verify(int CSize, std::vector<Tout> C, int verbosity) {
  int errors = 0;
  for (uint32_t i = 0; i < CSize; i++) {
    uint32_t ref = i + 2;
    if (C[i] != ref) {
      std::cout << "Error in output " << C[i] << " != " << ref
                << std::endl;
      errors++;
    } else {
      if(verbosity > 1)
        std::cout << "Correct output " << C[i] << " == " << ref
                  << std::endl;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  srand(time(NULL));

  int INOUT0_VOLUME = 64; // Input only, 64
  int INOUT1_VOLUME = 64; // Not used
  int INOUT2_VOLUME = 64; // Output only, 64

  size_t INOUT0_SIZE = INOUT0_VOLUME * sizeof(INOUT0_DATATYPE); 
  size_t INOUT1_SIZE = INOUT1_VOLUME * sizeof(INOUT1_DATATYPE); 
  size_t INOUT2_SIZE = INOUT2_VOLUME * sizeof(INOUT2_DATATYPE); 

  size_t OUT_SIZE = INOUT2_SIZE + trace_size;

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
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

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // Initialize input/ output buffer sizes
  // TODO - Add your custom buffer size here
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inout0 = xrt::bo(device, INOUT0_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inout1 = xrt::bo(device, INOUT1_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  // Assumes trace will only be added to inout2
  auto bo_inout2 = xrt::bo(device, OUT_SIZE,
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Initiaalize input buffers
  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize Inout buffer 0
  INOUT0_DATATYPE *bufInOut0 = bo_inout0.map<INOUT0_DATATYPE *>();
  std::vector<INOUT0_DATATYPE> AVec(INOUT0_VOLUME);
  for (int i = 0; i < INOUT0_VOLUME; i++)
    AVec[i] = i + 1;
    //AVec.push_back(i + 1);  
  memcpy(bufInOut0, AVec.data(), (AVec.size() * sizeof(INOUT0_DATATYPE)));

  // Initialize Inout buffer 1
  // INOUT1_DATATYPE *bufInOut1 = bo_inout1.map<INOUT0_DATATYPE *>();
  // std::vector<INOUT1_DATATYPE> BVec(INOUT1_VOLUME);
  // for (int i = 0; i < INOUT1_VOLUME; i++)
  //   BVec[i] = i + 1
  //   //BVec.push_back(i + 1);
  // memcpy(bufInOut1, BVec.data(), (BVec.size() * sizeof(INOUT1_DATATYPE)));

  // Initialize Inout buffer 2
  char *bufInOut2 = bo_inout2.map<char *>();
  std::vector<INOUT2_DATATYPE> CVec(INOUT2_VOLUME);
  memset(bufInOut2, 0, OUT_SIZE); // Zeroes out INOUT2_VOLUME + trace_size

  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Initialize run configs
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // Run loop
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }

    // Run kernel
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(bo_instr, instr_v.size(), bo_inout0, bo_inout1, bo_inout2);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_inout2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Copy output results and verify they are correct
    memcpy(CVec.data(), bufInOut2, (CVec.size() * sizeof(INOUT2_DATATYPE)));
    if (do_verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      errors = verify(INOUT2_VOLUME, CVec, verbosity);
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
      test_utils::write_out_trace(((char *)bufInOut2) + INOUT2_SIZE, trace_size,
                                     vm["trace_file"].as<std::string>());
    }

    // Accumulat run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // TODO - Mac count to guide gflops 
  float macs = 0;

  // Print verification results
  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " 
              << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " 
              << macs / (1000 * npu_time_max) << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
