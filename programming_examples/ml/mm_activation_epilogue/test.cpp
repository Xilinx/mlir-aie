//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#include "cxxopts.hpp"
#include "test_utils.h"

using DATATYPE = float;

// ----------------------------------------------------------------------------
// Reference math (matches mm_activation_epilogue_row's three modes).
// ----------------------------------------------------------------------------
static float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

static float gelu_tanh_ref(float x) {
  const float c0 = 0.7978845608f; // sqrt(2/pi)
  const float c1 = 0.044715f;
  return 0.5f * x * (1.0f + std::tanh(c0 * (x + c1 * x * x * x)));
}

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
int verify(int size, std::vector<DATATYPE> &A, std::vector<DATATYPE> &Identity,
           std::vector<DATATYPE> &Silu, std::vector<DATATYPE> &Gelu,
           int verbosity) {
  int errors = 0;
  for (int i = 0; i < size; i++) {
    const float a = A[i];

    if (Identity[i] != a) {
      if (verbosity >= 1)
        std::cout << "identity mismatch at " << i << ": " << Identity[i]
                  << " != " << a << std::endl;
      errors++;
    }

    // NOTE: nearly_equal's 3rd arg is a RELATIVE tolerance (epsilon); the
    // absolute threshold is the 4th (abs_th), and it defaults to FLT_MIN.
    // Writing `nearly_equal(x, y, 0.05f)` therefore asks for a strict ~5%
    // RELATIVE gate with no absolute floor, which fails on every near-zero
    // SiLU/GELU output (actual=0 vs expected=-0.0002 is a 0.0002 absolute
    // error but an infinite relative one) even though an atol=0.05 gate is
    // plainly satisfied. Pass 0.05 as abs_th to get the intended gate.
    const float silu_expected = silu_ref(a);
    if (!test_utils::nearly_equal(silu_expected, Silu[i], 1e-6f, 0.05f)) {
      if (verbosity >= 1)
        std::cout << "silu mismatch at " << i << ": " << Silu[i]
                  << " != " << silu_expected << std::endl;
      errors++;
    }

    const float gelu_expected = gelu_tanh_ref(a);
    if (!test_utils::nearly_equal(gelu_expected, Gelu[i], 1e-6f, 0.05f)) {
      if (verbosity >= 1)
        std::cout << "gelu mismatch at " << i << ": " << Gelu[i]
                  << " != " << gelu_expected << std::endl;
      errors++;
    }
  }
  return errors;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
  cxxopts::Options options("mm_activation_epilogue Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);
  options.add_options()("length,l", "the number of float32 elements per tensor",
                        cxxopts::value<int>()->default_value("65536"));

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  // Must match the -l the design was compiled with, or the host buffers will
  // not match the xclbin's. The design splits `length` over 1024-element tiles
  // across 2 cores, so it rejects anything that is not a multiple of 2048.
  int VOLUME = vm["length"].as<int>();
  if (VOLUME <= 0 || VOLUME % 2048) {
    std::cerr << "--length must be a positive multiple of 2048." << std::endl;
    return 1;
  }
  size_t SIZE = VOLUME * sizeof(DATATYPE);
  size_t OUT_SIZE =
      SIZE + trace_size; // trace, if any, rides on the last output

  // Fixed seed: the GELU path's measured device error (max 0.0375) leaves only
  // ~1.3x headroom under the 0.05 gate, so a time-seeded draw could flake.
  srand(42);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_id =
      xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_silu =
      xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  // Assumes trace will only be added to the last output.
  auto bo_gelu =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  DATATYPE *bufA = bo_a.map<DATATYPE *>();
  std::vector<DATATYPE> AVec(VOLUME);
  for (int i = 0; i < VOLUME; i++)
    AVec[i] = 8.0f * (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f);
  memcpy(bufA, AVec.data(), AVec.size() * sizeof(DATATYPE));

  char *bufId = bo_id.map<char *>();
  char *bufSilu = bo_silu.map<char *>();
  char *bufGelu = bo_gelu.map<char *>();
  memset(bufId, 0, SIZE);
  memset(bufSilu, 0, SIZE);
  memset(bufGelu, 0, OUT_SIZE);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_id.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_silu.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_gelu.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  int errors = 0;

  std::vector<DATATYPE> IdVec(VOLUME), SiluVec(VOLUME), GeluVec(VOLUME);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_id, bo_silu, bo_gelu);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_id.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_silu.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_gelu.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations)
      continue;

    memcpy(IdVec.data(), bufId, IdVec.size() * sizeof(DATATYPE));
    memcpy(SiluVec.data(), bufSilu, SiluVec.size() * sizeof(DATATYPE));
    memcpy(GeluVec.data(), bufGelu, GeluVec.size() * sizeof(DATATYPE));

    if (do_verify) {
      errors = verify(VOLUME, AVec, IdVec, SiluVec, GeluVec, verbosity);
    } else if (verbosity >= 1) {
      std::cout << "WARNING: results not verified." << std::endl;
    }

    if (trace_size > 0) {
      test_utils::write_out_trace(((char *)bufGelu) + SIZE, trace_size,
                                  vm["trace_file"].as<std::string>());
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  std::cout << "Min NPU time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU time: " << npu_time_max << "us." << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
