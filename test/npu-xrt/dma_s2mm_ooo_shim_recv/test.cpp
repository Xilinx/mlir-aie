// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the two-slot rotation proof of the out-of-order S2MM shim receiver.
// Sender s2 stamps out_of_order id 0 and sends [1..8]; sender s3 stamps id 1
// and sends [101..108]. The receive BD with runtime bd_id 0 writes slot 1 and
// the BD with id 1 writes slot 0, so a correct result (slot 0 = s3, slot 1 =
// s2) is only possible if placement follows the header id, not arrival or
// config order.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include GEN_HDR

#ifndef XCLBIN
#define XCLBIN std::string("final.xclbin")
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define DTYPE int32_t
#define SLOT 8
#define OUT_LEN 16

int main() {
  std::optional<std::vector<uint32_t>> instr_opt = generate_txn_main_sequence();
  if (!instr_opt) {
    std::cout << "builder returned nullopt (exceeds BD pool)\n";
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

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, OUT_LEN * sizeof(DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_tok = xrt::bo(device, SLOT * sizeof(DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  DTYPE *buf_out = bo_out.map<DTYPE *>();
  memset(buf_out, 0, OUT_LEN * sizeof(DTYPE));
  DTYPE *buf_tok = bo_tok.map<DTYPE *>();
  memset(buf_tok, 0, SLOT * sizeof(DTYPE));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_tok.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_out, bo_tok);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Slot 0 (offset 0) must hold sender s3's [101..108]; slot 1 (offset 8) must
  // hold sender s2's [1..8].
  bool pass = true;
  for (int i = 0; i < SLOT; i++) {
    DTYPE exp0 = 101 + i;
    DTYPE exp1 = 1 + i;
    if (buf_out[i] != exp0) {
      std::cout << "MISMATCH slot0 elem=" << i << ": got " << buf_out[i]
                << " expected " << exp0 << "\n";
      pass = false;
    }
    if (buf_out[SLOT + i] != exp1) {
      std::cout << "MISMATCH slot1 elem=" << i << ": got " << buf_out[SLOT + i]
                << " expected " << exp1 << "\n";
      pass = false;
    }
  }

  std::cout << (pass ? "PASS!" : "FAIL.") << " (" << instr_v.size()
            << " insts)\n";
  return pass ? 0 : 1;
}
