// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for a DMA transfer length set at runtime via length_parameter.
//
// Setup:
//   - Input buffer: 16 rows of 16 i32 values [0, 1, ..., 255]
//   - The DMAs move the first 8 values of each of `rows` rows into `out`, and
//     the last 8 into `out2`
//   - A third pair writes the first 8 * `rows` values twice into `out3`, as
//     rows of 8 at a stride of 16, the second pass 256 values in
//   - Output buffers: 256 i32 values each (512 for `out3`), prefilled with -1
//
// We run with several row counts and check both the values moved and that
// nothing past them was written.
//

#include <cstdint>
#include <iostream>
#include <stdexcept>
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
  constexpr int N = 256;
  constexpr int ROW = 16;
  constexpr int ROW_READ = 8;
  constexpr int N3 = 512;
  constexpr int PASS3 = 256;

  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *buf_in = bo_in.map<int32_t *>();
  for (int i = 0; i < N; ++i)
    buf_in[i] = i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();
  xrt::bo bo_out2 = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *buf_out2 = bo_out2.map<int32_t *>();
  xrt::bo bo_out3 = xrt::ext::bo{device, N3 * sizeof(int32_t)};
  auto *buf_out3 = bo_out3.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);
  run.set_arg(2, bo_out2);
  run.set_arg(3, bo_out3);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  bool all_pass = true;
  for (int32_t rows : {0, 1, 5, 0, 16}) {
    for (int i = 0; i < N; ++i) {
      buf_out[i] = -1;
      buf_out2[i] = -1;
    }
    for (int i = 0; i < N3; ++i)
      buf_out3[i] = -1;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    params.write("rows", rows);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int moved = rows * ROW_READ;
    int errors = 0;
    for (int i = 0; i < N; ++i) {
      int32_t expected =
          i < moved ? (i / ROW_READ) * ROW + i % ROW_READ : int32_t(-1);
      if (buf_out[i] != expected) {
        if (errors < 8)
          std::cout << "  out[" << i << "] = " << buf_out[i] << ", expected "
                    << expected << std::endl;
        ++errors;
      }
      int32_t expected2 = i < moved ? expected + ROW_READ : int32_t(-1);
      if (buf_out2[i] != expected2) {
        if (errors < 8)
          std::cout << "  out2[" << i << "] = " << buf_out2[i] << ", expected "
                    << expected2 << std::endl;
        ++errors;
      }
    }
    for (int i = 0; i < N3; ++i) {
      int row = (i % PASS3) / ROW;
      int col = i % ROW;
      int32_t expected3 =
          row < rows && col < ROW_READ ? row * ROW_READ + col : int32_t(-1);
      if (buf_out3[i] != expected3) {
        if (errors < 8)
          std::cout << "  out3[" << i << "] = " << buf_out3[i] << ", expected "
                    << expected3 << std::endl;
        ++errors;
      }
    }
    bool pass = errors == 0;
    if (!pass)
      all_pass = false;
    std::cout << "rows=" << rows << "  moved " << moved << " values  "
              << (pass ? "PASS" : "FAIL") << std::endl;
  }

  // 16 rows reach the end of the input, so the compiler bounds rows to
  // [0, 16] and the host library rejects anything else before it is written.
  for (int32_t rows : {17, -1}) {
    bool rejected = false;
    try {
      params.write("rows", rows);
    } catch (const std::invalid_argument &e) {
      rejected = true;
      std::cout << "rows=" << rows << "  rejected: " << e.what() << std::endl;
    }
    if (!rejected || params.read("rows") != 16u) {
      std::cout << "rows=" << rows << "  not rejected  FAIL" << std::endl;
      all_pass = false;
    }
  }

  if (all_pass) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
