//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host check for a same-packet-id fan-in.
//
// N source tiles each send one CHUNK-byte packet carrying packet id 0, all to
// the same shim S2MM channel. The arbiter interleaves same-id packets in an
// order the compiler does not control, so this test deliberately does NOT check
// positions -- it histograms the output by sentinel value. Source i fills its
// buffer with the byte (i + 1).
//
// Correct behaviour: exactly CHUNK bytes of each sentinel 1..N, nothing else.
// A router that merges two same-id streams onto one amsel and then fans that
// amsel out to two master ports delivers some source's payload twice, which
// shows up here as a doubled histogram bucket.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#ifndef NUM_SRC
#define NUM_SRC 2
#endif

constexpr int CHUNK = 256;
constexpr int OUT_SIZE = CHUNK * NUM_SRC;

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string xclbin_name = (argc > 1) ? argv[1] : "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int8_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  // Poison the output so that "not written" is distinguishable from "wrote 0".
  int8_t *bufOut = bo_out.map<int8_t *>();
  std::fill(bufOut, bufOut + OUT_SIZE, (int8_t)-1);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  // Bounded wait, and catch the XRT exception: a merge-then-fanout misroute
  // duplicates packets that no destination BD ever consumes, which backs the
  // stream up and deadlocks rather than returning wrong data. Report that as a
  // diagnosis instead of letting the exception abort the process.
  try {
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_out);
    if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
      std::cout << "Kernel TIMED OUT after 10s -- stream stalled.\n"
                << "This is the signature of duplicated packets with no "
                   "matching destination BD.\n\nfailed.\n\n";
      return 1;
    }
    ert_cmd_state r = run.state();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r
                << "\n\nfailed.\n\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cout << "Kernel dispatch failed: " << e.what() << "\n"
              << "ERT_CMD_STATE_TIMEOUT here means the stream deadlocked, the "
                 "documented\nhardware symptom of the merge-then-fanout "
                 "misroute.\n\nfailed.\n\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::map<int, int> hist;
  for (int i = 0; i < OUT_SIZE; i++)
    hist[(int)bufOut[i]]++;

  std::cout << "output histogram (value: count):\n";
  for (auto &[v, c] : hist)
    std::cout << "  " << v << ": " << c << "\n";

  int errors = 0;
  for (int s = 1; s <= NUM_SRC; s++) {
    int got = hist.count(s) ? hist[s] : 0;
    if (got != CHUNK) {
      std::cout << "Error: sentinel " << s << " appeared " << got
                << " times, expected " << CHUNK;
      if (got == 2 * CHUNK)
        std::cout << "  <-- DUPLICATED (merge-then-fanout)";
      if (got == 0)
        std::cout << "  <-- MISSING";
      std::cout << "\n";
      errors++;
    }
  }
  for (auto &[v, c] : hist) {
    if (v < 1 || v > NUM_SRC) {
      std::cout << "Error: unexpected value " << v << " x" << c << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed. errors=" << errors << "\n\n";
  return 1;
}
