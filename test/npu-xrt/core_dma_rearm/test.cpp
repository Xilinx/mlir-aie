//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Send two buffers to a channel armed for one. The core arms it again between
// them, and only then sends back the token this reads.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include GEN_HDR

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

constexpr int SRC_LEN = 16;
constexpr int TOK_LEN = 8;
constexpr int TOKEN = 42;
using DTYPE = int32_t;

int main() {
  std::optional<std::vector<uint32_t>> instr_opt = generate_txn_main_sequence();
  if (!instr_opt || instr_opt->empty()) {
    std::cout << "empty instruction sequence\n";
    return 1;
  }
  std::vector<uint32_t> instr_v = std::move(*instr_opt);

  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_src = xrt::bo(device, SRC_LEN * sizeof(DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_tok = xrt::bo(device, TOK_LEN * sizeof(DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  DTYPE *buf_src = bo_src.map<DTYPE *>();
  for (int i = 0; i < SRC_LEN; i++)
    buf_src[i] = i + 1;
  DTYPE *buf_tok = bo_tok.map<DTYPE *>();
  std::fill_n(buf_tok, TOK_LEN, 0);

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_tok.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_src, bo_tok);
  // The run finishes only if the core got past its second wait, which only an
  // armed channel lets it do. One that never finishes ends either as a
  // timeout here or as the command state XRT throws on once the hardware has
  // given up on it.
  bool completed = false;
  try {
    completed =
        run.wait2(std::chrono::seconds(5)) == std::cv_status::no_timeout;
  } catch (const std::exception &e) {
    std::cout << "the run ended badly: " << e.what() << "\n";
  }
  if (!completed) {
    std::cout << "FAIL: the run did not complete; the channel was never "
                 "armed again"
              << std::endl;
    // Leaving by the usual way would take the still-pending run's destructor
    // with it, and that throws too, turning this into an abort.
    std::_Exit(1);
  }

  bo_tok.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (buf_tok[0] != TOKEN) {
    std::cout << "FAIL: token " << buf_tok[0] << ", expected " << TOKEN << "\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
