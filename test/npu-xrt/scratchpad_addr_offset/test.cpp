// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for DMA address offset patching via offset_parameter.
//
// Setup:
//   - Input buffer: 32 i32 values [0, 1, 2, ..., 31]
//   - Core: passthrough of 8 elements
//   - offset_parameter @input_offset controls the DMA read start position
//
// We run three times with different offsets and verify the output each time.
//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

int main(int argc, const char *argv[]) {
  constexpr int N_INPUT = 32;
  constexpr int N_OUTPUT = 8;

  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  // Input buffer: [0, 1, 2, ..., 31] as i32
  xrt::bo bo_in = xrt::ext::bo{device, N_INPUT * sizeof(int32_t)};
  auto *buf_in = bo_in.map<int32_t *>();
  for (int i = 0; i < N_INPUT; ++i)
    buf_in[i] = i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Output buffer: 8 x i32
  xrt::bo bo_out = xrt::ext::bo{device, N_OUTPUT * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  struct TestCase {
    int32_t offset;
    std::vector<int32_t> expected;
  };
  std::vector<TestCase> test_cases = {
      {0, {0, 1, 2, 3, 4, 5, 6, 7}},
      {8, {8, 9, 10, 11, 12, 13, 14, 15}},
      {16, {16, 17, 18, 19, 20, 21, 22, 23}},
  };

  bool all_pass = true;
  int run_idx = 0;
  for (auto &tc : test_cases) {
    ++run_idx;

    // Clear output
    memset(buf_out, 0, N_OUTPUT * sizeof(int32_t));
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Write offset parameter (in elements)
    params.write("input_offset", tc.offset);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::vector<int32_t> result(buf_out, buf_out + N_OUTPUT);
    bool pass = (result == tc.expected);
    if (!pass)
      all_pass = false;

    std::cout << "Run " << run_idx << " — offset=" << tc.offset
              << "  expected=[";
    for (size_t i = 0; i < tc.expected.size(); ++i)
      std::cout << tc.expected[i] << (i + 1 < tc.expected.size() ? ", " : "");
    std::cout << "]  got=[";
    for (size_t i = 0; i < result.size(); ++i)
      std::cout << result[i] << (i + 1 < result.size() ? ", " : "");
    std::cout << "]  " << (pass ? "PASS" : "FAIL") << std::endl;
  }

  if (all_pass) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
