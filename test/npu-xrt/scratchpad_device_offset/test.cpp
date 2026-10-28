// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A DMA offset the device computes within the same command (aie.mlir).
//
// The table holds 0..63. The host writes row 7's offset (56) into the
// scratchpad before each run; the core overwrites it with selector * 8. The
// output is the selector's row only if the device's write is copied in again
// before the fill is patched.
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
  constexpr int N_TABLE = 64;
  constexpr int N_ROW = 8;
  constexpr int32_t HOST_OFFSET = 56;

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "test:sequence");

  xrt::bo bo_sel = xrt::ext::bo{device, sizeof(int32_t)};
  auto *buf_sel = bo_sel.map<int32_t *>();

  xrt::bo bo_table = xrt::ext::bo{device, N_TABLE * sizeof(int32_t)};
  auto *buf_table = bo_table.map<int32_t *>();
  for (int i = 0; i < N_TABLE; ++i)
    buf_table[i] = i;
  bo_table.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::bo bo_out = xrt::ext::bo{device, N_ROW * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_sel);
  run.set_arg(1, bo_table);
  run.set_arg(2, bo_out);
  auto params = test_utils::ParameterScratchpad(run, "params.txt");
  // The core's offset lands in the run's own scratchpad, through a
  // user-pointer buffer over its host memory: the scratchpad's own buffer
  // is on the device heap, which the shim DMA does not reach.
  auto scratchpad = run.get_ctrl_scratchpad_bo();
  xrt::bo alias =
      xrt::ext::bo{device, scratchpad.map<void *>(), scratchpad.size()};
  run.set_arg(3, alias);

  bool all_pass = true;
  for (int32_t sel : {3, 5, 0}) {
    buf_sel[0] = sel;
    bo_sel.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(buf_out, 0, N_ROW * sizeof(int32_t));
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    params.write("row_offset", HOST_OFFSET);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto pad = run.get_ctrl_scratchpad_bo();
    pad.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::cout << "selector " << sel << ": scratchpad after the run holds "
              << pad.map<int32_t *>()[0] << std::endl;
    std::vector<int32_t> got(buf_out, buf_out + N_ROW), want, host;
    for (int i = 0; i < N_ROW; ++i) {
      want.push_back(sel * N_ROW + i);
      host.push_back(HOST_OFFSET + i);
    }
    bool pass = got == want;
    all_pass &= pass;
    std::cout << "selector " << sel << ": got [";
    for (int i = 0; i < N_ROW; ++i)
      std::cout << got[i] << (i + 1 < N_ROW ? ", " : "");
    std::cout << "] "
              << (pass          ? "PASS"
                  : got == host ? "FAIL (the host's row)"
                                : "FAIL")
              << std::endl;
  }
  std::cout << (all_pass ? "PASS!" : "FAIL.") << std::endl;
  return all_pass ? 0 : 1;
}
