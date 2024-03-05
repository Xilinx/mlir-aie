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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../matrix_multiplication.h"

constexpr int M = 256;
constexpr int K = 256;
constexpr int N = 256;

constexpr int A_VOLUME = M * K;
constexpr int B_VOLUME = N * K;
constexpr int C_VOLUME = M * N;

using A_DATATYPE = std::bfloat16_t;
using B_DATATYPE = std::bfloat16_t;
using C_DATATYPE = float;

constexpr int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
constexpr int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

constexpr bool VERIFY = true;
constexpr bool ENABLE_TRACING = false;
constexpr int TRACE_SIZE = 8192;

constexpr int OUT_SIZE = C_SIZE + (ENABLE_TRACING ? TRACE_SIZE : 0);

namespace po = boost::program_options;

void write_out_trace(char *bufOut, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut =
      (uint32_t *)((char *)bufOut + sizeof(C_DATATYPE) * C_VOLUME);
  for (int i = 0; i < TRACE_SIZE / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  matmul_common::add_default_options(desc);
  if (ENABLE_TRACING) {
    desc.add_options()("trace,t",
                       po::value<std::string>()->default_value("trace.txt"),
                       "where to store trace output");
  }

  matmul_common::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();

  srand(time(NULL));

  std::vector<uint32_t> instr_v = matmul_common::load_instr_sequence(vm["instr"].as<std::string>());
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
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    AVec[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    BVec[i] = matmul_common::random_bfloat16_t();
  }
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  memset(bufOut, 0, OUT_SIZE);
  std::vector<C_DATATYPE> CVec(C_VOLUME);

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto start = std::chrono::system_clock::now();
  auto run = kernel(bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
  run.wait();
  auto stop = std::chrono::system_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reinterpret first C_VOLUME bytes of bufOut as our output C_DATATYPE C
  // matrix
  memcpy(CVec.data(), bufOut, (CVec.size() * sizeof(C_DATATYPE)));

  int errors = 0;
  int max_printable_errors = 100;

  std::vector<C_DATATYPE> CVecRef(C_VOLUME);
  if (VERIFY) {
    const float absTol = 0.5;
    const float relTol = 0.5;
    matmul_common::matmul(M, N, K, AVec, BVec, CVecRef);

    for(int row = 0; row < M; row++) {
      for(int col = 0; col < N; col++) {
        if(!matmul_common::nearly_equal(CVecRef[row*N+col], CVec[row*N+col], relTol, absTol)) {
          errors++;
          if (errors < max_printable_errors) {
            std::cout << "\nerror, row " << row << ", col " << col << ", expected "
                      << std::to_string(CVecRef[row*N+col]) << ", got "
                      << std::to_string(CVec[row*N+col]) << "\n";
          }
        }
      }
    }
  } else {
    std::cout << "WARNING: matmul results not verified." << std::endl;
  }

  if (ENABLE_TRACING) {
    write_out_trace(bufOut, vm["trace"].as<std::string>());
  }

  std::cout << std::endl
            << "NPU matmul time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                     start)
                   .count()
            << "ms." << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nerror count: " << errors << "\n\n";
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
