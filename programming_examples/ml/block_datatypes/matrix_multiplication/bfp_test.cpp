//===- bfp_test.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdfloat>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Clangd fix, remove
#ifdef _CLANGD
namespace std {
using bfloat16_t = double;
} // namespace std
#endif

#include "../helper.h"
#include "common.h"

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
// TODO: This might have to be adjusted for bfp
float abs_tol = matmul_common::get_abs_tol<std::bfloat16_t>();
float rel_tol = matmul_common::get_rel_tol<std::bfloat16_t>();

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);
  options.add_options()("trows,w", "Tile size m",
                        cxxopts::value<int>()->default_value("64"))(
      "tinner,y", "Tile size k", cxxopts::value<int>()->default_value("64"))(
      "tcolumns,z", "Tile size n", cxxopts::value<int>()->default_value("64"));

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int m = vm["w"].as<int>();
  int k = vm["y"].as<int>();
  int n = vm["z"].as<int>();

  bool do_verify_stochastic = (long long)M * N > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_SIZE = M * K;
  int B_SIZE = N * K;
  int C_SIZE = M * N;

  size_t A_VOLUME = (A_SIZE * sizeof(uint8_t)) * 1.125;
  size_t B_VOLUME = (B_SIZE * sizeof(uint8_t)) * 1.125;
  size_t C_VOLUME = (C_SIZE * sizeof(uint8_t)) * 1.125;

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

  // ------------------------------------------------------
  // Initialize input/output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_VOLUME, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_VOLUME, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_VOLUME, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // ------------------------------------------------------
  // Generate data for buffers
  // ------------------------------------------------------
  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  std::vector<float> AVec(A_SIZE);
  for (int i = 0; i < A_SIZE; i++) {
    // Limiting to 16 to avoid precision loss issues
    AVec[i] = (float)((rand() % 16));

    // if (i % K == i / K) {
    //   AVec[i] = 1.0;
    // } else {
    //   AVec[i] = 0.0;
    // }

    // AVec[i] = (i / 8) % 64;
  }

  std::vector<float> BVec(B_SIZE);
  for (int i = 0; i < B_SIZE; i++) {
    // Limiting to 16 to avoid precision loss issues
    BVec[i] = (float)((rand() % 16));
    // Diagonal:
    // if (i % K == i / K) {
    //   BVec[i] = 1.0;
    // } else {
    //   BVec[i] = 0.0;
    // }

    // if (i % 128 < 8 && i / 128 < 8)
    //   BVec[i] = i / 8;
    // else
    //   BVec[i] = 0.0;

    // BVec[i] = i / 8;
  }

  auto AVecBfp = floatToBfp16(8, A_SIZE, AVec.data(), 0);
  auto BVecBfp = floatToBfp16(8, B_SIZE, BVec.data(), 0);

  auto shuffleStart = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> AVecBfpShuffled =
      shuffleMatrixForBfp16ebs8(K, M, k, m, AVecBfp);
  std::vector<uint8_t> BVecBfpShuffled =
      shuffleMatrixForBfp16ebs8(K, N, k, n, BVecBfp);
  auto shuffleStop = std::chrono::high_resolution_clock::now();

  float inputShuffleTime =
      std::chrono::duration_cast<std::chrono::microseconds>(shuffleStop -
                                                            shuffleStart)
          .count();

  // std::ofstream outfile1("inputB.txt");
  // matmul_common::print_matrix(BVec, K, N, K, outfile1, " ", " ... ", 3);
  // printBfp16ebs8Array(A_VOLUME, BVecBfp, 16, 16, outfile1);
  // outfile1.close();

  // std::ofstream outfile2("inputBShuffled.txt");
  // auto temp = bfp16ebs8ToFloat(B_VOLUME, BVecBfpShuffled.data());
  // // printBfp16ebs8Array(B_VOLUME, BVecBfpShuffled, 16, 16, outfile2);
  // matmul_common::print_matrix(temp, K, N, K, outfile2, " ", " ... ", 3);
  // outfile2.close();

  // ------------------------------------------------------
  // Write data into buffers
  // ------------------------------------------------------
  uint8_t *bufA = bo_a.map<uint8_t *>();
  uint8_t *bufB = bo_b.map<uint8_t *>();
  memcpy(bufA, AVecBfpShuffled.data(), A_VOLUME);
  memcpy(bufB, BVecBfpShuffled.data(), B_VOLUME);

  // Initialize outputs; bufOut is results matrix
  char *bufOut = bo_out.map<char *>();

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Run kernel
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  float outShuffleTime = 0;

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // ------------------------------------------------------
    // Check output
    // ------------------------------------------------------
    if (do_verify) {
      std::vector<uint8_t> CVecBfp(C_VOLUME);
      memcpy(CVecBfp.data(), bufOut, C_VOLUME);

      auto outShuffleStart = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> CVecBfpShuffled =
          shuffleMatrixForBfp16ebs8(N, M, n, m, CVecBfp, true);
      auto outShuffleStop = std::chrono::high_resolution_clock::now();

      outShuffleTime += std::chrono::duration_cast<std::chrono::microseconds>(
                            outShuffleStop - outShuffleStart)
                            .count();

      auto CVec = bfp16ebs8ToFloat(C_VOLUME, CVecBfpShuffled.data(), 0);

      if (verbosity >= 1) {
        std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      if (do_verify_stochastic) {
        errors = matmul_common::verify_stochastic<float, float, float>(
            M, N, K, AVec, BVec, CVec, verify_stochastic_n_samples, verbosity,
            abs_tol, rel_tol, true);
      } else {
        errors = matmul_common::verify<float, float, float>(
            M, N, K, AVec, BVec, CVec, verbosity, abs_tol * 3, rel_tol, true);
      }
      auto vstop = std::chrono::system_clock::now();

      // std::ofstream outfile("output.txt");
      // matmul_common::print_matrix(CVec, N, M, N, outfile, " ", " ... ", 3);
      // // printBfp16ebs8Array(C_VOLUME, CVecBfp, 16, 16, outfile);
      // outfile.close();

      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Output results
  // ------------------------------------------------------
  std::cout << std::endl
            << "Avg NPU matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Avg NPU gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;

  std::cout << std::endl
            << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

  std::cout << std::endl
            << "Shuffle time: "
            << inputShuffleTime + (outShuffleTime / n_iterations) << "us."
            << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nError count: " << errors;
  if (do_verify_stochastic) {
    std::cout << " (out of " << verify_stochastic_n_samples
              << " random samples)";
  }
  std::cout << "\n\n";
  std::cout << "\nFailed.\n\n";
  return 1;
}
