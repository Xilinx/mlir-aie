//===- test_dynamic.cpp - Dynamic GEMM test harness -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Test harness for dynamic single-core GEMM. Generates TXN instructions at
// runtime for the specified M/K/N, using the same XCLBIN for all sizes.
//
//===----------------------------------------------------------------------===//

#include "dynamic_gemm_txn.h"

#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdfloat>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

using A_DATATYPE = std::bfloat16_t;
using B_DATATYPE = std::bfloat16_t;
using C_DATATYPE = float;
using ACC_DATATYPE = float;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Dynamic GEMM Test");
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>()->default_value("MLIR_AIE"))(
      "verbosity,v", "the verbosity of the output",
      cxxopts::value<int>()->default_value("1"))(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "rows,M", "Matrix rows M",
      cxxopts::value<int>()->default_value("64"))(
      "inner,K", "Matrix inner dimension K",
      cxxopts::value<int>()->default_value("64"))(
      "columns,N", "Matrix columns N",
      cxxopts::value<int>()->default_value("64"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"));

  cxxopts::ParseResult vm;
  try {
    vm = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n" << options.help() << "\n";
    return 1;
  }
  if (vm.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  if (!vm.count("xclbin")) {
    std::cerr << "Error: --xclbin is required\n" << options.help() << "\n";
    return 1;
  }

  int verbosity = vm["verbosity"].as<int>();
  bool do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup = vm["warmup"].as<int>();
  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    std::cerr << "Error: M, K, N must be multiples of 32\n";
    return 1;
  }

  srand(1726250518); // fixed seed for reproducibility

  if (verbosity >= 1)
    std::cout << "Dynamic GEMM: " << M << "x" << K << "x" << N << std::endl;

  // Generate TXN instructions for this M/K/N
  std::vector<uint32_t> instr_v = dynamic_gemm::generate_gemm_txn(M, K, N);
  if (verbosity >= 1)
    std::cout << "Generated " << instr_v.size() << " instruction words\n";

  int A_VOLUME = M * K;
  int B_VOLUME = K * N;
  int C_VOLUME = M * N;
  size_t A_SIZE = A_VOLUME * sizeof(A_DATATYPE);
  size_t B_SIZE = B_VOLUME * sizeof(B_DATATYPE);
  size_t C_SIZE = C_VOLUME * sizeof(C_DATATYPE);

  // XRT setup
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [&node](xrt::xclbin::kernel &xk) {
        return xk.get_name().rfind(node, 0) == 0;
      });

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace =
      xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  // Initialize buffers
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++)
    AVec[i] = matmul_common::get_random<A_DATATYPE>();
  memcpy(bufA, AVec.data(), A_SIZE);

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++)
    BVec[i] = matmul_common::get_random<B_DATATYPE>() * i;
  memcpy(bufB, BVec.data(), B_SIZE);

  char *bufOut = bo_out.map<char *>();
  memset(bufOut, 0, C_SIZE);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup;
  float npu_time_total = 0, npu_time_min = 9999999, npu_time_max = 0;
  int errors = 0;
  float macs = 2.0f * M * K * N;

  float abs_tol = matmul_common::get_abs_tol<C_DATATYPE>();
  float rel_tol = matmul_common::get_rel_tol<C_DATATYPE>();

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(3, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp,
                      bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < (unsigned)n_warmup)
      continue;

    if (do_verify) {
      std::vector<C_DATATYPE> CVec(C_VOLUME);
      memcpy(CVec.data(), bufOut, C_SIZE);
      if (verbosity >= 1)
        std::cout << "Verifying against reference matmul ..." << std::endl;
      errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
          M, N, K, AVec, BVec, CVec, verbosity, abs_tol, rel_tol);
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    npu_time_total += npu_time;
    npu_time_min = std::min(npu_time, npu_time_min);
    npu_time_max = std::max(npu_time, npu_time_max);
  }

  std::cout << "\nAvg NPU matmul time: " << npu_time_total / n_iterations
            << "us.\n";
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << "\n";
  std::cout << "\nMin NPU matmul time: " << npu_time_min << "us.\n";
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << "\n";

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\nFailed.\n\n";
    return 1;
  }
}
