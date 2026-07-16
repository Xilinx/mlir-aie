// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the dynamic (runtime tile-count) ping-pong passthrough. Unlike a
// static design, the instruction stream is NOT read from an insts.bin: it is
// built AT RUNTIME by the generated C++ TXN builder (generate_txn_main_sequence,
// from GEN_HDR) called with the tile count `n` chosen at runtime (argv[1]). That
// is the whole point of the dynamic BD free-list pool -- one compiled design
// serves any n. The builder returns std::nullopt if n exceeds the tile's BD
// pool; we treat that as a build failure.

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
  // Runtime tile count: the single parameter the compiled design is
  // parameterized on. Same xclbin, same builder, any n.
  int64_t n = (argc > 1) ? std::atoll(argv[1]) : 8;
  const int in_len = TILE_LEN;
  const int64_t out_len = TILE_LEN * n;

  // Build the instruction stream at runtime for this n.
  std::optional<std::vector<uint32_t>> instr_opt =
      generate_txn_main_sequence(n);
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
  auto bo_input = xrt::bo(device, in_len * sizeof(DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(3));
  auto bo_output = xrt::bo(device, out_len * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < in_len; i++)
    buf_input[i] = i + 1;

  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, out_len * sizeof(DTYPE));

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

  // Golden: output is n copies of the input tile.
  bool pass = true;
  for (int64_t tile = 0; tile < n; tile++) {
    for (int i = 0; i < TILE_LEN; i++) {
      DTYPE expected = buf_input[i];
      DTYPE got = buf_output[tile * TILE_LEN + i];
      if (got != expected) {
        std::cout << "MISMATCH at tile=" << tile << " elem=" << i << ": got "
                  << got << " expected " << expected << "\n";
        pass = false;
      }
    }
  }

  std::cout << (pass ? "PASS!" : "FAIL.") << " (n=" << n
            << ", " << instr_v.size() << " insts)\n";
  return pass ? 0 : 1;
}
