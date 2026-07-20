// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host for the dynamic (runtime M/N) whole-array matmul. The instruction stream
// is NOT read from an insts.bin: it is built AT RUNTIME by the generated C++
// TXN builder (generate_txn_main_sequence, from GEN_HDR) called with the M/K/N
// chosen at runtime (argv). One compiled xclbin serves any M/N that is a
// multiple of the tile granularity and within the compiled maximum -- the
// compute cores run a while_true loop and consume exactly as many tiles as the
// host DMA feeds. K is fixed at the compiled value (the core's reduction depth
// is baked in). The builder returns std::nullopt if the shape overflows the
// tile's BD pool; we treat that as a build failure.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#include GEN_HDR

#ifndef XCLBIN
#define XCLBIN std::string("final.xclbin")
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

using A_DTYPE = std::int16_t;
using B_DTYPE = std::int16_t;
using C_DTYPE = std::int16_t;
using ACC_DTYPE = std::int16_t;

int main(int argc, const char *argv[]) {
  // Runtime problem dimensions. K must equal the compiled value.
  int M = (argc > 1) ? std::atoi(argv[1]) : 512;
  int K = (argc > 2) ? std::atoi(argv[2]) : 512;
  int N = (argc > 3) ? std::atoi(argv[3]) : 512;

  // Build the instruction stream at runtime for this M/K/N.
  std::optional<std::vector<uint32_t>> instr_opt =
      generate_txn_main_sequence(M, K, N);
  if (!instr_opt) {
    std::cout << "builder returned nullopt for M=" << M << " K=" << K
              << " N=" << N << " (exceeds BD pool / compiled max)\n";
    return 1;
  }
  std::vector<uint32_t> instr_v = std::move(*instr_opt);
  if (instr_v.empty()) {
    std::cout << "builder produced an empty instruction stream\n";
    return 1;
  }

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

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(instr_v[0]),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, M * K * sizeof(A_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_b = xrt::bo(device, K * N * sizeof(B_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));
  auto bo_c = xrt::bo(device, M * N * sizeof(C_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

  // Host-side inputs and golden reference.
  std::vector<A_DTYPE> A_vec(M * K);
  std::vector<B_DTYPE> B_vec(K * N);
  for (auto &v : A_vec)
    v = matmul_common::get_random<A_DTYPE>();
  for (auto &v : B_vec)
    v = matmul_common::get_random<B_DTYPE>();

  A_DTYPE *buf_a = bo_a.map<A_DTYPE *>();
  std::memcpy(buf_a, A_vec.data(), M * K * sizeof(A_DTYPE));
  B_DTYPE *buf_b = bo_b.map<B_DTYPE *>();
  std::memcpy(buf_b, B_vec.data(), K * N * sizeof(B_DTYPE));
  C_DTYPE *buf_c = bo_c.map<C_DTYPE *>();
  std::memset(buf_c, 0, M * N * sizeof(C_DTYPE));

  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(instr_v[0]));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::vector<C_DTYPE> C_vec(M * N);
  std::memcpy(C_vec.data(), buf_c, M * N * sizeof(C_DTYPE));

  int n_errors = matmul_common::verify<A_DTYPE, C_DTYPE, ACC_DTYPE>(
      M, N, K, A_vec, B_vec, C_vec, /*verbosity=*/0);

  std::cout << (n_errors == 0 ? "PASS!" : "FAIL.") << " (M=" << M << " K=" << K
            << " N=" << N << ", " << instr_v.size() << " insts, " << n_errors
            << " errors)\n";
  return n_errors == 0 ? 0 : 1;
}
