// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Reads a 64-word table back from runtime offsets on a mem tile (8 words at
// @m_off, a multiple of 4) and a core tile (13 words at @c_off), checks every
// word against the table, and that the host refuses a mem tile offset off a
// 4-word step.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

int main() {
  constexpr int N_IN = 64, N_OUT = 16, M_LEN = 8, C_LEN = 13;
  constexpr int32_t SENTINEL = -1;

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "test:sequence");

  xrt::bo bo_in = xrt::ext::bo{device, N_IN * sizeof(int32_t)};
  auto *in = bo_in.map<int32_t *>();
  for (int i = 0; i < N_IN; ++i)
    in[i] = 1000 + i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo bo_m = xrt::ext::bo{device, N_OUT * sizeof(int32_t)};
  xrt::bo bo_c = xrt::ext::bo{device, N_OUT * sizeof(int32_t)};
  auto *m_out = bo_m.map<int32_t *>();
  auto *c_out = bo_c.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_m);
  run.set_arg(2, bo_c);
  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  bool pass = true;
  try {
    params.write("m_off", 1);
    std::cout << "m_off = 1 was accepted, but the mem tile moves in 4 words"
              << std::endl;
    pass = false;
  } catch (const std::invalid_argument &e) {
    std::cout << "m_off = 1 refused: " << e.what() << std::endl;
  }

  for (auto [m_off, c_off] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 1}, {4, 2}, {8, 3}, {52, 51}}) {
    for (int i = 0; i < N_OUT; ++i)
      m_out[i] = c_out[i] = SENTINEL;
    bo_m.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    params.write("m_off", m_off);
    params.write("c_off", c_off);
    params.sync();
    run.start();
    run.wait2();
    bo_m.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bool ok = true;
    for (int i = 0; i < N_OUT; ++i) {
      ok &= m_out[i] == (i < M_LEN ? in[m_off + i] : SENTINEL);
      ok &= c_out[i] == (i < C_LEN ? in[c_off + i] : SENTINEL);
    }
    std::cout << "m_off=" << m_off << " c_off=" << c_off << " mem=[";
    for (int i = 0; i < N_OUT; ++i)
      std::cout << m_out[i] << (i + 1 < N_OUT ? " " : "");
    std::cout << "] core=[";
    for (int i = 0; i < N_OUT; ++i)
      std::cout << c_out[i] << (i + 1 < N_OUT ? " " : "");
    std::cout << "] " << (ok ? "PASS" : "FAIL") << std::endl;
    pass &= ok;
  }
  std::cout << (pass ? "PASS!" : "FAIL.") << std::endl;
  return pass ? 0 : 1;
}
