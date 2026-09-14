// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define DTYPE int32_t

constexpr size_t DATA_COUNT = 20;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(DTYPE);

// The sequence alternates col0/col1/col0/col1, so a stale configuration left
// by one column shows up as a wrong slice rather than a hang.
int main(int argc, const char *argv[]) {
  int reps = (argc > 1) ? std::atoi(argv[1]) : 1;

  std::vector<DTYPE> vec_in(DATA_COUNT);
  for (size_t i = 0; i < vec_in.size(); i++)
    vec_in[i] = DTYPE(i);

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);
  xrt::bo bo_inout = xrt::ext::bo{device, BUF_SIZE};
  char *buf_inout = bo_inout.map<char *>();

  std::vector<DTYPE> vec_ref(DATA_COUNT);
  for (size_t i = 0; i < 4; i++)
    vec_ref[i] = vec_in[i] + 2;
  for (size_t i = 4; i < 8; i++)
    vec_ref[i] = vec_in[i] + 3;
  for (size_t i = 8; i < 12; i++)
    vec_ref[i] = vec_in[i] + 2;
  for (size_t i = 12; i < 16; i++)
    vec_ref[i] = vec_in[i] + 3;
  for (size_t i = 16; i < DATA_COUNT; i++)
    vec_ref[i] = vec_in[i];

  int exact = 0;
  std::vector<DTYPE> vec_out(DATA_COUNT);
  for (int r = 0; r < reps; r++) {
    memcpy(buf_inout, vec_in.data(), BUF_SIZE);
    bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_inout);
    run.start();
    run.wait2();
    bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(vec_out.data(), buf_inout, BUF_SIZE);
    if (vec_out == vec_ref) {
      exact++;
    } else if (r == 0 || exact == r) {
      // Print the first divergence only, so a run that fails every dispatch
      // does not bury the summary.
      for (size_t i = 0; i < DATA_COUNT; i++)
        std::cout << "  [" << std::setw(2) << i << "] in " << std::setw(4)
                  << vec_in[i] << "  out " << std::setw(4) << vec_out[i]
                  << "  ref " << std::setw(4) << vec_ref[i]
                  << (vec_out[i] == vec_ref[i] ? "" : "   <-- mismatch")
                  << std::endl;
    }
  }

  if (exact == reps) {
    std::cout << "PASS: " << exact << "/" << reps
              << " exact alternations across disjoint columns" << std::endl;
    return 0;
  }
  std::cout << "Failed: " << exact << "/" << reps << " exact" << std::endl;
  return 1;
}
