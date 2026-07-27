// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the dynamic ping-pong passthrough WITH a separate trace buffer. The
// instruction stream is built at runtime by the generated C++ TXN builder
// (generate_txn_main_sequence(n), from GEN_HDR). -aie-insert-trace-flows
// appended a dedicated i8 trace-buffer argument (default reuse_output_buffer=
// false), so the kernel takes an extra trailing operand for the trace buffer.
// We verify the data output is still correct (n copies of the input, proving
// the trace didn't perturb the runtime-sized transfer) AND that the trace
// buffer is non-empty (tracing actually wrote something).

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
#define TRACE_SIZE 8192

int main(int argc, const char *argv[]) {
  int64_t n = (argc > 1) ? std::atoll(argv[1]) : 8;
  const int in_len = TILE_LEN;
  const int64_t out_len = TILE_LEN * n;

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
  auto xkernel = std::find_if(xkernels.begin(), xkernels.end(),
                              [](xrt::xclbin::kernel &k) {
                                return k.get_name().rfind(KERNEL_NAME, 0) == 0;
                              });
  if (xkernel == xkernels.end()) {
    std::cout << "no kernel matching '" << KERNEL_NAME << "' in the xclbin\n";
    return 1;
  }
  std::string kernel_name = xkernel->get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  // Data args are group_id 3 (in) and 4 (out); the appended trace buffer is the
  // next kernel operand, group_id 5.
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(instr_v[0]),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, in_len * sizeof(DTYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, out_len * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_trace =
      xrt::bo(device, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < in_len; i++)
    buf_input[i] = i + 1;

  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, out_len * sizeof(DTYPE));

  char *buf_trace = bo_trace.map<char *>();
  memset(buf_trace, 0, TRACE_SIZE);

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(instr_v[0]));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output, bo_trace);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Data golden: output is n copies of the input tile -- proves the separate
  // trace buffer did not perturb the runtime-sized output transfer.
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

  // Trace sanity: the buffer must contain at least one non-zero word (tracing
  // wrote something into its dedicated buffer).
  bool trace_nonempty = false;
  for (int i = 0; i < TRACE_SIZE; i++)
    if (buf_trace[i] != 0) {
      trace_nonempty = true;
      break;
    }
  if (!trace_nonempty) {
    std::cout << "trace buffer is empty (no trace data written)\n";
    pass = false;
  }

  std::cout << (pass ? "PASS!" : "FAIL.") << " (n=" << n << ", "
            << instr_v.size() << " insts, trace "
            << (trace_nonempty ? "non-empty" : "empty") << ")\n";
  return pass ? 0 : 1;
}
