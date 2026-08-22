//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the program-memory overlay example. One core runs three real
// aie_kernels in turn, each written into its program memory just before it
// runs, so the output rows can only all be right if all three were really
// loaded and executed.
//
//   row 0   silu     x * sigmoid(x)
//   row 1   gelu     tanh approximation
//   row 2   softmax  over the whole row
//
// The kernels use tanh/exp approximations in bfloat16, so the references here
// are checked to a tolerance rather than exactly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int N_ELEMS = 1024; // baked into silu_bf16 / gelu_bf16
constexpr int N_PHASES = 3;
// bfloat16 carries about three decimal digits, and both kernels approximate
// tanh, so compare on a relative tolerance.
constexpr float TOL = 2e-2f;

using test_utils::bfloat16_t;
// bfloat16_t is std::bfloat16_t only where the toolchain has it; otherwise it is
// a bare uint16_t holding the bits, and a plain float() cast silently converts
// the bit pattern instead of the value. Always go through the helper.
using test_utils::bfloat16_to_float;

static void reference(int phase, const std::vector<float> &in,
                      std::vector<float> &out) {
  switch (phase) {
  case 0: // silu
    for (int i = 0; i < N_ELEMS; i++)
      out[i] = in[i] / (1.0f + std::exp(-in[i]));
    break;
  case 1: // gelu, tanh approximation
    for (int i = 0; i < N_ELEMS; i++) {
      float x = in[i];
      float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
      out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    break;
  default: { // softmax
    float mx = *std::max_element(in.begin(), in.end());
    float sum = 0.0f;
    for (int i = 0; i < N_ELEMS; i++)
      sum += std::exp(in[i] - mx);
    for (int i = 0; i < N_ELEMS; i++)
      out[i] = std::exp(in[i] - mx) / sum;
    break;
  }
  }
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("program_memory_overlay");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernelName = vm["kernel"].as<std::string>();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N_ELEMS * sizeof(bfloat16_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N_PHASES * N_ELEMS * sizeof(bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  bfloat16_t *bufIn = bo_in.map<bfloat16_t *>();
  std::vector<float> inF(N_ELEMS);
  for (int i = 0; i < N_ELEMS; i++) {
    // Straddle zero and stay in a range where the approximations are valid.
    inF[i] = -4.0f + 8.0f * (float(i) / float(N_ELEMS - 1));
    bufIn[i] = test_utils::bfloat16_from_float(inF[i]);
    inF[i] = bfloat16_to_float(bufIn[i]); // what the device actually got
  }
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bfloat16_t *bufOut = bo_out.map<bfloat16_t *>();

  static const char *kNames[] = {"silu", "gelu", "softmax"};
  int errors = 0;
  std::vector<float> want(N_ELEMS);
  for (int phase = 0; phase < N_PHASES; phase++) {
    reference(phase, inF, want);
    for (int i = 0; i < N_ELEMS; i++) {
      float got = bfloat16_to_float(bufOut[phase * N_ELEMS + i]);
      float tol = TOL * std::max(1.0f, std::fabs(want[i]));
      if (std::fabs(got - want[i]) <= tol)
        continue;
      if (errors < 8)
        std::cout << "phase " << phase << " (" << kNames[phase] << ") [" << i
                  << "]: got " << got << ", expected " << want[i] << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\n" << errors << " errors -- an overlay did not take effect\n";
  std::cout << "failed.\n\n";
  return 1;
}
