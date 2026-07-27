// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the dynamic CONDITIONAL passthrough. The instruction stream is built
// at runtime by the generated C++ TXN builder (generate_txn_main_sequence, from
// GEN_HDR) called with the tile count `n` and a predicate `do_copy`. When
// do_copy is true the design copies n input tiles to the output; when false it
// issues no transfer and the output stays zero. One xclbin serves both.
// argv: [n] [do_copy]

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
#define TILE_LEN 256

int main(int argc, const char *argv[]) {
  int64_t n = (argc > 1) ? std::atoll(argv[1]) : 8;
  bool do_copy = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;
  const int in_len = TILE_LEN;
  const int64_t out_len = TILE_LEN * n;

  // Build the instruction stream at runtime for this (n, do_copy).
  std::optional<std::vector<uint32_t>> instr_opt =
      generate_txn_main_sequence(n, do_copy);
  if (!instr_opt) {
    std::cout << "builder returned nullopt for n=" << n
              << " (exceeds BD pool)\n";
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
  auto bo_input = xrt::bo(device, in_len * sizeof(DTYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, out_len * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < in_len; i++)
    buf_input[i] = i + 1;

  // Sentinel: an untouched output element must stay this when do_copy is false.
  const DTYPE kSentinel = -12345;
  DTYPE *buf_output = bo_output.map<DTYPE *>();
  for (int64_t i = 0; i < out_len; i++)
    buf_output[i] = kSentinel;

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

  // Golden: do_copy true -> n copies of the input tile; false -> untouched.
  bool pass = true;
  for (int64_t tile = 0; tile < n; tile++) {
    for (int i = 0; i < TILE_LEN; i++) {
      DTYPE expected = do_copy ? buf_input[i] : kSentinel;
      DTYPE got = buf_output[tile * TILE_LEN + i];
      if (got != expected) {
        std::cout << "MISMATCH at tile=" << tile << " elem=" << i << ": got "
                  << got << " expected " << expected << "\n";
        pass = false;
      }
    }
  }

  std::cout << (pass ? "PASS!" : "FAIL.") << " (n=" << n
            << " do_copy=" << do_copy << ", " << instr_v.size() << " insts)\n";
  return pass ? 0 : 1;
}
