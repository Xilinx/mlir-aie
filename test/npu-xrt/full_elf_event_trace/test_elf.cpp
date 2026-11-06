//===- test_elf.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host runner for the full-ELF event-trace design.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr size_t N = 64;
constexpr size_t OUT_BYTES = N * sizeof(int32_t);
constexpr size_t TRACE_SIZE = 8192;

int main() {
  auto device = xrt::device(0);
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_in = xrt::ext::bo{device, OUT_BYTES};
  xrt::bo bo_out = xrt::ext::bo{device, OUT_BYTES};
  xrt::bo bo_trace = xrt::ext::bo{device, TRACE_SIZE};

  auto *in = bo_in.map<int32_t *>();
  for (size_t i = 0; i < N; i++)
    in[i] = static_cast<int32_t>(i);
  std::memset(bo_out.map<char *>(), 0, OUT_BYTES);
  std::memset(bo_trace.map<char *>(), 0, TRACE_SIZE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);
  run.set_arg(2, bo_trace);
  run.start();
  run.wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto *out = bo_out.map<int32_t *>();
  for (size_t i = 0; i < N; i++)
    if (out[i] != static_cast<int32_t>(i)) {
      std::cerr << "FAIL: out[" << i << "] = " << out[i] << " != " << i << "\n";
      return 1;
    }

  uint32_t *words = bo_trace.map<uint32_t *>();
  std::ofstream fout("trace.txt");
  for (size_t i = 0; i < TRACE_SIZE / sizeof(uint32_t); i++)
    fout << std::setfill('0') << std::setw(8) << std::hex << words[i] << "\n";
  fout.close();

  std::cout << "passthrough OK; trace written to trace.txt\n";
  return 0;
}
