//===- test.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host testbench for the reconfiguration example, selected at compile time:
//
//   (default)   xclbin flow, single run.
//                 argv: <xclbin> <insts.bin> <cores>
//   -DRUNLIST   xclbin flow, <runs> runs chained in one xrt::runlist.
//                 argv: <xclbin> <insts.bin> <cores> <runs>
//   -DFULL_ELF  full-ELF reconfig flow (kernel main:sequence).
//                 argv: <aie.elf> <cores>
//
// Every mode checks the output buffer equals [0, 1, ..., cores-1] and prints
// the elapsed device time so the flows can be compared apples-to-apples.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#if defined(FULL_ELF)
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"
#else
#include <fstream>
#if defined(RUNLIST)
#include "xrt/experimental/xrt_kernel.h" // xrt::runlist
#endif
#endif

using clk = std::chrono::high_resolution_clock;

static int check(const int32_t *out, int cores, int run_idx = -1) {
  int errors = 0;
  for (int i = 0; i < cores; i++) {
    if (out[i] != i) {
      std::cout << "Error:";
      if (run_idx >= 0)
        std::cout << " run " << run_idx;
      std::cout << " out[" << i << "] = " << out[i] << " != " << i << "\n";
      errors++;
    }
  }
  return errors;
}

#if !defined(FULL_ELF)
static std::vector<uint32_t> load_instr_binary(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint32_t> v(n / sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(v.data()), n);
  return v;
}
#endif

int main(int argc, char **argv) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  int errors = 0;

#if defined(FULL_ELF)
  std::string elf_path = argv[1];
  int cores = std::stoi(argv[2]);

  xrt::elf ctx_elf{elf_path};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");

  xrt::bo bo_out = xrt::ext::bo{device, (size_t)cores * sizeof(int32_t)};
  int32_t *out = bo_out.map<int32_t *>();
  for (int i = 0; i < cores; i++)
    out[i] = -1;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_out);

  auto t0 = clk::now();
  run.start();
  run.wait2();
  auto t1 = clk::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  errors = check(out, cores);
  std::cout << "full-ELF reconfig: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                   .count()
            << " us\n";
#else
  std::string xclbin_path = argv[1];
  std::string insts_path = argv[2];
  int cores = std::stoi(argv[3]);

  auto instr_v = load_instr_binary(insts_path);
  xrt::xclbin xclbin(xclbin_path);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, "MLIR_AIE");

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  unsigned int opcode = 3;

#if defined(RUNLIST)
  int n_runs = std::stoi(argv[4]);
  xrt::runlist runlist(context);
  std::vector<xrt::bo> bos;
  std::vector<xrt::run> runs;
  for (int r = 0; r < n_runs; r++) {
    xrt::bo bo_out(device, cores * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                   kernel.group_id(3));
    int32_t *out = bo_out.map<int32_t *>();
    for (int i = 0; i < cores; i++)
      out[i] = -1;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run(kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (uint32_t)instr_v.size());
    run.set_arg(3, bo_out);

    runlist.add(run);
    bos.push_back(std::move(bo_out));
    runs.push_back(std::move(run));
  }

  auto t0 = clk::now();
  runlist.execute();
  runlist.wait();
  auto t1 = clk::now();

  for (int r = 0; r < n_runs; r++) {
    bos[r].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    errors += check(bos[r].map<int32_t *>(), cores, r);
  }
  std::cout << "xclbin runlist (" << n_runs << " runs): "
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                   .count()
            << " us\n";
#else
  xrt::bo bo_out(device, cores * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                 kernel.group_id(3));
  int32_t *out = bo_out.map<int32_t *>();
  for (int i = 0; i < cores; i++)
    out[i] = -1;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto t0 = clk::now();
  auto run = kernel(opcode, bo_instr, (uint32_t)instr_v.size(), bo_out);
  run.wait();
  auto t1 = clk::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  errors = check(out, cores);
  std::cout << "xclbin single run: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                   .count()
            << " us\n";
#endif
#endif

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\n" << errors << " mismatches.\nfail.\n\n";
  return 1;
}
