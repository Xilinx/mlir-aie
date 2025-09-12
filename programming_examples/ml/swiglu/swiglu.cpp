// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <stdfloat>
#include <cassert>
#include <cmath>

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "cxxopts.hpp"
#include "test_utils.h"

#define DTYPE std::bfloat16_t

#include "invocation_plan.h"

void matmul(const std::vector<DTYPE> &a, const std::vector<DTYPE> &b, std::vector<DTYPE> &c, unsigned m, unsigned k, unsigned n) {
  assert(a.size() == m * k);
  assert(b.size() == k * n);
  assert(c.size() == m * n);
  std::cout << "Calculating reference matmul of " << m << "x" << k << " * " << k << "x" << n << "..." << std::endl;
  double iterations = m*n;
  for(unsigned row = 0; row < m; row++) {
    for(unsigned col = 0; col < n; col++) {
      float sum = 0.0f;
      for(unsigned i = 0; i < k; i++) {
        sum += static_cast<float>(a[row*k + i]) * static_cast<float>(b[i*n + col]);
      }
      std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << (100.0f * (row*n + col + 1) / iterations) << "%" << std::flush;
      c[row*n + col] = static_cast<DTYPE>(sum);
    }
  }
  std::cout << std::endl;
}

void eltwise_mul(const std::vector<DTYPE> &a, const std::vector<DTYPE> &b, std::vector<DTYPE> &c) {
  assert(a.size() == b.size());
  assert(a.size() == c.size());
  for(unsigned i = 0; i < a.size(); i++) {
    c[i] = a[i] * b[i];
  }
}

DTYPE silu(DTYPE &input) {
  DTYPE half_x = input * DTYPE(0.5f);
  DTYPE tanh_half_x = std::tanh(half_x);
  DTYPE sigmoid_approx = DTYPE(0.5f) * (tanh_half_x + DTYPE(1.0f));
  return input * sigmoid_approx;
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options(argv[0]);
  cxxopts::ParseResult vm;
  options.add_options()
     ("help,h", "produce help message")
     ("verify", "verify outputs?", cxxopts::value<unsigned>()->default_value("0"))
     ("xclbin,x", "the input xclbin path", cxxopts::value<std::string>())
     ("insts-matmul", "path to the instruction binary for the matrix-vector multiplication kernel", cxxopts::value<std::string>()->default_value("build/matmul.bin"))
     ("insts-silu", "path to the instruction binary for the silu activation kernel", cxxopts::value<std::string>()->default_value("build/silu.bin"))
     ("insts-eltwise-mul", "path to the instruction binary for the element-wise multiplication kernel", cxxopts::value<std::string>()->default_value("build/eltwise_mul.bin"))
     ("dim", "embedding dimension", cxxopts::value<unsigned>()->default_value("2048"))
     ("seq", "sequence length", cxxopts::value<unsigned>()->default_value("2048"))
     ("epsilon", "relative threshold for floating point comparsion (result must be within this percentage of magnitude of both results)", cxxopts::value<float>()->default_value("0.0202"))
     ("abs_th", "absolute threshold for floating point comparison (difference between results must either be less than this value or less than relative threshold)", cxxopts::value<float>()->default_value("0.1"));

  vm = options.parse(argc, argv);
  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }
  // Check required options
  if (!vm.count("xclbin")) {
    std::cerr << "Error: Required options missing\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::string xclbin_path = vm["xclbin"].as<std::string>();
  float epsilon = vm["epsilon"].as<float>();
  float abs_th = vm["abs_th"].as<float>();
  unsigned dim = vm["dim"].as<unsigned>();
  unsigned seq = vm["seq"].as<unsigned>();
  unsigned verify = vm["verify"].as<unsigned>();

  // set up and calculate reference values
  // This is the CPU-calculated equivalent of the computation grpah we instantiate below.
  std::vector<std::bfloat16_t> ref_inp          = std::vector<std::bfloat16_t>(dim*seq);
  std::vector<std::bfloat16_t> ref_W1           = std::vector<std::bfloat16_t>(dim*dim);
  std::vector<std::bfloat16_t> ref_W2           = std::vector<std::bfloat16_t>(dim*dim);
  std::vector<std::bfloat16_t> ref_left         = std::vector<std::bfloat16_t>(dim*seq);
  std::vector<std::bfloat16_t> ref_right        = std::vector<std::bfloat16_t>(dim*seq);
  std::vector<std::bfloat16_t> ref_left_swished = std::vector<std::bfloat16_t>(dim*seq);
  std::vector<std::bfloat16_t> ref_result       = std::vector<std::bfloat16_t>(dim*seq);

  std::bfloat16_t *ptr_ref_inp          = nullptr;
  std::bfloat16_t *ptr_ref_W1           = nullptr;
  std::bfloat16_t *ptr_ref_W2           = nullptr;
  std::bfloat16_t *ptr_ref_left         = nullptr;
  std::bfloat16_t *ptr_ref_right        = nullptr;
  std::bfloat16_t *ptr_ref_left_swished = nullptr;
  std::bfloat16_t *ptr_ref_result       = nullptr;

  if(verify) {
    std::cout << "Calculating reference values on CPU (this will take a while)..." << std::endl;
    // Use a fixed seed for repeatability
    std::srand(42);
    std::generate(ref_inp.begin(), ref_inp.end(), []() { return test_utils::random_bfloat16_t(static_cast<std::bfloat16_t>(4.0f), static_cast<std::bfloat16_t>(0.0f)); });
    std::generate(ref_W1.begin(), ref_W1.end(),   []() { return test_utils::random_bfloat16_t(static_cast<std::bfloat16_t>(4.0f), static_cast<std::bfloat16_t>(0.0f)); });
    std::generate(ref_W2.begin(), ref_W2.end(),   []() { return test_utils::random_bfloat16_t(static_cast<std::bfloat16_t>(4.0f), static_cast<std::bfloat16_t>(0.0f)); });
    matmul(ref_W1, ref_inp, ref_left, dim, dim, seq);
    std::transform(ref_left.begin(), ref_left.end(), ref_left_swished.begin(), [](DTYPE val) { return silu(val); });
    matmul(ref_W2, ref_inp, ref_right, dim, dim, seq);
    eltwise_mul(ref_left_swished, ref_right, ref_result);
    ptr_ref_inp          = ref_inp.data();
    ptr_ref_W1           = ref_W1.data();
    ptr_ref_W2           = ref_W2.data();
    ptr_ref_left         = ref_left.data();
    ptr_ref_right        = ref_right.data();
    ptr_ref_left_swished = ref_left_swished.data();
    ptr_ref_result       = ref_result.data();
  }

  // Initialize the NPU and load our design
  constexpr unsigned device_index = 0;
  xrt::device device = xrt::device(device_index);
  xrt::xclbin xclbin(xclbin_path);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  // We use the InvocationPlan API from invocation_plan.h below to instantiate and run multiple designs
  // (a simple computation graph). We will include a version of this API in future versions of the test_utils.
  std::vector<KernelInfo> kernels = {
    {"matmul",      vm["insts-matmul"].as<std::string>(),      "matmul"},
    {"silu",        vm["insts-silu"].as<std::string>(),        "silu"},
    {"eltwise_mul", vm["insts-eltwise-mul"].as<std::string>(), "eltwise_mul"}
  };

  std::vector<KernelBufferInfo> buffers = {
    {"inp",          dim*seq, KernelBufferInfo::Direction::IN,     ptr_ref_inp},
    {"W1",           dim*dim, KernelBufferInfo::Direction::IN,     ptr_ref_W1},
    {"W2",           dim*dim, KernelBufferInfo::Direction::IN,     ptr_ref_W2},
    {"left",         dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_left},
    {"right",        dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_right},
    {"left_swished", dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_left_swished},
    {"result",       dim*seq, KernelBufferInfo::Direction::OUT,    ptr_ref_result}
  };

  std::vector<KernelInvocationInfo> runlist = {
    {"matmul",      {"W1",           "inp",          "left"  }},
    {"matmul",      {"W2",           "inp",          "right" }},
    {"silu",        {"left",         "left_swished"          }},
    {"eltwise_mul", {"left_swished", "right",       "result"}}
  };

  InvocationPlanInfo plan_info = {
      .xclbin = xclbin_path,
      .kernels = kernels,
      .buffers = buffers,
      .runlist = runlist
  };

  InvocationPlan plan = InvocationPlan::fromInfo(plan_info, device, xclbin, context);
  auto [success, t_elapsed] = plan.invoke();
  
  std::cout << "Elapsed time: " << t_elapsed << " μs" << std::endl;

  std::vector<std::pair<std::string, unsigned>> errors = plan.verifyOutputBuffers(epsilon, abs_th);
  
  if(errors.size()) {
    for (const auto &[buffer_name, i] : errors) {
      const KernelBuffer &buffer = plan.buffers[buffer_name];
    std::cout << buffer_name << ": Mismatch at index " << i << ": " << std::fixed << std::setprecision(3) << std::setw(8) << buffer.buf[i] << " != " << std::setw(8) << buffer.reference[i] << std::endl;
      if ("result" == buffer_name) {
        // result is supposed to be the result left_swished[i] * right[i]
        std::cout << "  Reference: " << std::fixed << std::setprecision(3) << std::setw(8) << plan.buffers["left_swished"].reference[i] << " * " << std::setw(8) << plan.buffers["right"].reference[i] << " = " << std::setw(8) << buffer.reference[i] << std::endl;
        std::cout << "  Computed:  " << std::fixed << std::setprecision(3) << std::setw(8) << plan.buffers["left_swished"].buf[i] << " * " << std::setw(8) << plan.buffers["right"].buf[i] << " = " << std::setw(8) << buffer.buf[i] << std::endl;
      }
    }
    std::cout << "FAIL." << std::endl;
    return 1;
  } else {
    std::cout << "PASS!" << std::endl;
  }

  return 0;
}
