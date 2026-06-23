//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

static test_utils::bfloat16_t relu_bf16(test_utils::bfloat16_t x) {
  return (test_utils::bfloat16_to_float(x) > 0.0f)
             ? x
             : test_utils::bfloat16_from_float(0.0f);
}

static test_utils::bfloat16_t silu_bf16(test_utils::bfloat16_t x) {
  const float xf = test_utils::bfloat16_to_float(x);
  const float sigmoid_approx = 0.5f * (std::tanh(0.5f * xf) + 1.0f);
  return test_utils::bfloat16_from_float(xf * sigmoid_approx);
}

static test_utils::bfloat16_t gelu_bf16(test_utils::bfloat16_t x) {
  const float xf = test_utils::bfloat16_to_float(x);
  const float sqrt_2_over_pi = 0.79788456f;
  const float beta = 0.044715f;
  const float inner = sqrt_2_over_pi * (xf + beta * xf * xf * xf);
  return test_utils::bfloat16_from_float(0.5f * xf * (1.0f + std::tanh(inner)));
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Eltwise Unary Test");
  cxxopts::ParseResult vm;

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "length,l", "the length of the transfer in bfloat16 elements",
      cxxopts::value<int>()->default_value("4096"))(
      "op,o", "Unary op the kernel computes (relu | silu | gelu)",
      cxxopts::value<std::string>()->default_value("relu"));

  try {
    vm = options.parse(argc, argv);

    if (vm.count("help")) {
      std::cout << options.help() << std::endl;
      return 1;
    }

    if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << std::endl;
      return 1;
    }
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::string op = vm["op"].as<std::string>();
  test_utils::bfloat16_t (*ref_fn)(test_utils::bfloat16_t) = nullptr;
  float rel_tol = 0.0f;
  // Larger abs_th lets the near-zero region pass when the LUT-backed
  // kernel rounds to 0 but the bf16-arith reference produces tiny
  // non-zero values (e.g., gelu near x = -3 yields ~-0.006).
  float abs_th = 0.001f;
  if (op == "relu") {
    ref_fn = relu_bf16;
    rel_tol = 0.001f;
  } else if (op == "silu") {
    ref_fn = silu_bf16;
    rel_tol = 0.04f;
  } else if (op == "gelu") {
    ref_fn = gelu_bf16;
    rel_tol = 0.1f;
    abs_th = 0.01f;
  } else {
    std::cerr << "--op must be 'relu', 'silu', or 'gelu' (got '" << op
              << "')\n";
    return 1;
  }

  int verbosity = vm["verbosity"].as<int>();

  int N = vm["length"].as<int>();
  if ((N % 1024)) {
    std::cerr << "Length must be a multiple of 1024." << std::endl;
    return 1;
  }

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>()
              << std::endl;
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>()
              << std::endl;
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  if (verbosity >= 1)
    std::cout << "Getting hardware context." << std::endl;
  xrt::hw_context context(device, xclbin.get_uuid());

  xrt::elf elf(vm["instr"].as<std::string>());
  xrt::module mod{elf};

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::ext::kernel(context, mod, kernelName);

  xrt::bo bo_inA = xrt::ext::bo{device, N * sizeof(test_utils::bfloat16_t)};
  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(test_utils::bfloat16_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  test_utils::bfloat16_t *bufInA = bo_inA.map<test_utils::bfloat16_t *>();
  std::vector<test_utils::bfloat16_t> srcVecA;
  for (int i = 0; i < N; i++)
    srcVecA.push_back(test_utils::bfloat16_from_float(i * 0.05f + -3.0f));
  memcpy(bufInA, srcVecA.data(),
         (srcVecA.size() * sizeof(test_utils::bfloat16_t)));

  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  auto cfg_run = kernel(opcode, 0, 0, bo_inA, bo_out);
  cfg_run.wait();
  auto start = std::chrono::high_resolution_clock::now();
  auto run = kernel(opcode, 0, 0, bo_inA, bo_out);
  run.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  const float npu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::cout << std::endl;
  std::cout << "Latency (us): " << npu_time << std::endl;
  std::cout << std::endl;

  double total_bytes = 2.0 * N * sizeof(test_utils::bfloat16_t);
  double bandwidth_GBps = total_bytes / (npu_time * 1e-6) / 1e9;
  std::cout << "Effective Bandwidth: " << bandwidth_GBps << " GB/s"
            << std::endl;

  test_utils::bfloat16_t *bufOut = bo_out.map<test_utils::bfloat16_t *>();

  int errors = 0;

  for (int i = 0; i < N; i++) {
    test_utils::bfloat16_t ref = ref_fn(srcVecA[i]);
    const float expected = test_utils::bfloat16_to_float(ref);
    const float actual = test_utils::bfloat16_to_float(*(bufOut + i));
    if (!test_utils::nearly_equal(actual, expected, rel_tol, abs_th)) {
      errors++;
      if (errors <= 100) {
        std::cout << "Mismatch at index " << i << ": "
                  << "Expected: " << expected << ", "
                  << "Got: " << actual << std::endl;
      }
    }
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
