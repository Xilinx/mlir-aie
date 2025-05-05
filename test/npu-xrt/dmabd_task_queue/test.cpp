//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int A_VOLUME = 5;
constexpr int B_VOLUME = 96;
constexpr int C_VOLUME = 96;
constexpr int D_VOLUME = 9;

using IN_DATATYPE = int32_t;

constexpr int A_SIZE = (A_VOLUME * sizeof(IN_DATATYPE));
constexpr int B_SIZE = (B_VOLUME * sizeof(IN_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(IN_DATATYPE));
constexpr int D_SIZE = (D_VOLUME * sizeof(IN_DATATYPE));

constexpr bool VERIFY = true;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("dmabd_task_queue");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  srand(time(NULL));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
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

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_d =
      xrt::bo(device, D_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  IN_DATATYPE *bufA = bo_a.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    AVec[i] = i;
  }
  memcpy(bufA, AVec.data(), (A_SIZE));

  IN_DATATYPE *bufB = bo_b.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    BVec[i] = i + 1;
  }
  memcpy(bufB, BVec.data(), (B_SIZE));

  IN_DATATYPE *bufC = bo_c.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> CVec(C_VOLUME);
  for (int i = 0; i < C_VOLUME; i++) {
    CVec[i] = i + 2;
  }
  memcpy(bufC, CVec.data(), (C_SIZE));

  IN_DATATYPE *bufD = bo_d.map<IN_DATATYPE *>();
  memset(bufD, 0, (D_SIZE));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_d.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = 1;

  int errors = 0;

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  run.set_arg(3, bo_a);
  run.set_arg(4, bo_b);
  run.set_arg(5, bo_c);
  run.set_arg(6, bo_d);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }
    // unsigned int opcode = 3;
    // auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_c);
    run.start();
    run.wait();

    bo_d.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (VERIFY) {
      std::vector<IN_DATATYPE> DVecRef(D_VOLUME);
      if (verbosity >= 1) {
        std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      for (unsigned i = 0; i < D_VOLUME; i++)
        DVecRef[i] = 0;
      for (unsigned i = 0; i < A_VOLUME; i++)
        DVecRef[0] += AVec[i];
      for (unsigned i = 1; i < D_VOLUME; i++) {
        for (unsigned j = 12 * (i - 1); j < 12 * i; j++) {
          DVecRef[i] += BVec[j] + CVec[j];
        }
      }
      for (unsigned i = 0; i < D_VOLUME; i++) {
        if (bufD[i] != DVecRef[i]) {
          std::cout << "Getting error at: bufD[" << i << "]: expected "
                    << DVecRef[i] << " actual " << bufD[i] << std::endl;
          errors++;
        }
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }
  }

  if (VERIFY && !errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
