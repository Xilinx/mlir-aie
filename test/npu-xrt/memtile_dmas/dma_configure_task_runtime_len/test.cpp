// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the dispatch-time-sized mem tile round trip. The instruction stream
// is not read from an insts.bin: it is built at runtime by the generated C++
// TXN builder called with the tile count `n` (argv[1]), so one xclbin serves
// every n. A nullopt return means one of the BD-field guards tripped.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#include GEN_HDR

#ifndef XCLBIN
#define XCLBIN std::string("final.xclbin")
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define DTYPE int32_t
#define TILE_LEN 512
#define MAX_LEN 4096

int main(int argc, const char *argv[]) {
  int64_t n = (argc > 1) ? std::atoll(argv[1]) : 8;
  const int64_t len = TILE_LEN * n;
  if (len > MAX_LEN) {
    std::cout << "n=" << n << " exceeds the " << (MAX_LEN / TILE_LEN)
              << "-tile mem tile buffer\n";
    return 1;
  }

  std::optional<std::vector<uint32_t>> instr_opt =
      generate_txn_main_sequence(n);
  if (!instr_opt) {
    std::cout << "builder returned nullopt for n=" << n
              << " (a BD field guard tripped)\n";
    return 1;
  }
  std::vector<uint32_t> instr_v = std::move(*instr_opt);
  assert(instr_v.size() > 0);

  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);

  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), KERNEL_NAME) == 0);

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, MAX_LEN * sizeof(DTYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, MAX_LEN * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < MAX_LEN; i++)
    buf_input[i] = i + 1;

  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, MAX_LEN * sizeof(DTYPE));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  bool pass = true;
  for (int64_t i = 0; i < len; i++) {
    if (buf_output[i] != buf_input[i]) {
      std::cout << "MISMATCH at " << i << ": got " << buf_output[i]
                << " expected " << buf_input[i] << "\n";
      pass = false;
      break;
    }
  }
  // The tail is what proves the BD actually shrank. Without it the test passes
  // just as well when the runtime length is ignored and a max-sized transfer
  // runs every time.
  for (int64_t i = len; i < MAX_LEN; i++) {
    if (buf_output[i] != 0) {
      std::cout << "WROTE PAST len at " << i << ": got " << buf_output[i]
                << " expected 0 (len=" << len << ")\n";
      pass = false;
      break;
    }
  }

  std::cout << (pass ? "PASS!" : "FAIL.") << " (n=" << n << ", len=" << len
            << ", " << instr_v.size() << " insts)\n";
  return pass ? 0 : 1;
}
