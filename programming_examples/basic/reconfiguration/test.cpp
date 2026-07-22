//===- test.cpp ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host testbench / micro-benchmark for the reconfiguration example, selected at
// compile time. Every mode runs `iters` timed iterations, checks the output is
// [0, 1, ..., cols*rows-1], and prints:
//
//   runtimes_us: t0,t1,...            (per-iteration device time)
//   stats_us: mean,min,max
//
// Modes:
//   (default)   "separate xclbins": worker + empty are separate xclbins, each
//               iteration runs the worker then the empty reset (two contexts).
//                 argv: <worker.xclbin> <worker.bin> <empty.xclbin> <empty.bin>
//                       <cols> <rows> <iters>
//   -DRUNLIST   worker + empty runs chained in one xrt::runlist per iteration.
//                 argv: same as default
//   -DFULL_ELF  full-ELF reconfig flow (kernel main:sequence); the ELF resets
//               itself (loads @empty then @worker), so no host-side reset.
//                 argv: <aie.elf> <cols> <rows> <iters>
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
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
#include "xrt/experimental/xrt_kernel.h" // xrt::runlist
#endif

using clk = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

static int check(const int32_t *out, int n) {
  int errors = 0;
  for (int i = 0; i < n; i++)
    if (out[i] != i) {
      std::cout << "Error: out[" << i << "] = " << out[i] << " != " << i << "\n";
      errors++;
    }
  return errors;
}

static void report(const std::vector<double> &t) {
  std::cout << "runtimes_us: ";
  for (size_t i = 0; i < t.size(); i++)
    std::cout << t[i] << (i + 1 < t.size() ? "," : "");
  std::cout << "\n";
  double mean = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
  double mn = *std::min_element(t.begin(), t.end());
  double mx = *std::max_element(t.begin(), t.end());
  std::cout << "stats_us: " << mean << "," << mn << "," << mx << "\n";
}

#if !defined(FULL_ELF)
static std::vector<uint32_t> load_bin(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint32_t> v(n / sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(v.data()), n);
  return v;
}

// A registered xclbin + its instruction stream, ready to spawn runs.
struct Kernel {
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  uint32_t n_instr;

  Kernel(xrt::device &device, const std::string &xclbin_path,
         const std::string &insts_path) {
    auto instr = load_bin(insts_path);
    xrt::xclbin xclbin(xclbin_path);
    device.register_xclbin(xclbin);
    context = xrt::hw_context(device, xclbin.get_uuid());
    kernel = xrt::kernel(context, "MLIR_AIE");
    n_instr = (uint32_t)instr.size();
    bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bo_instr.map<void *>(), instr.data(),
           instr.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run make_run(xrt::bo &out) {
    xrt::run r(kernel);
    r.set_arg(0, (unsigned)3); // opcode
    r.set_arg(1, bo_instr);
    r.set_arg(2, n_instr);
    r.set_arg(3, out);
    return r;
  }
};
#endif

int main(int argc, char **argv) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  int errors = 0;
  std::vector<double> times;

#if defined(FULL_ELF)
  std::string elf_path = argv[1];
  int n = std::stoi(argv[2]) * std::stoi(argv[3]);
  int iters = std::stoi(argv[4]);

  xrt::elf ctx_elf{elf_path};
  xrt::hw_context context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo bo_out = xrt::ext::bo{device, (size_t)n * sizeof(int32_t)};
  int32_t *out = bo_out.map<int32_t *>();

  for (int it = 0; it < iters; it++) {
    std::fill(out, out + n, -1);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = xrt::run(kernel);
    run.set_arg(0, bo_out);
    auto t0 = clk::now();
    run.start();
    run.wait2();
    auto t1 = clk::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    errors += check(out, n);
    times.push_back(us(t1 - t0).count());
  }
#else
  std::string worker_xclbin = argv[1], worker_bin = argv[2];
  std::string empty_xclbin = argv[3], empty_bin = argv[4];
  int n = std::stoi(argv[5]) * std::stoi(argv[6]);
  int iters = std::stoi(argv[7]);

  Kernel worker(device, worker_xclbin, worker_bin);
  Kernel empty(device, empty_xclbin, empty_bin);

  xrt::bo bo_out(device, n * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                 worker.kernel.group_id(3));
  int32_t *out = bo_out.map<int32_t *>();
  xrt::bo bo_empty(device, sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                   empty.kernel.group_id(3));

  for (int it = 0; it < iters; it++) {
    std::fill(out, out + n, -1);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

#if defined(RUNLIST)
    xrt::runlist runlist(worker.context);
    auto rw = worker.make_run(bo_out);
    auto re = empty.make_run(bo_empty);
    runlist.add(rw);
    runlist.add(re);
    auto t0 = clk::now();
    runlist.execute();
    runlist.wait();
    auto t1 = clk::now();
#else
    auto rw = worker.make_run(bo_out);
    auto re = empty.make_run(bo_empty);
    auto t0 = clk::now();
    rw.start();
    rw.wait();
    re.start();
    re.wait();
    auto t1 = clk::now();
#endif

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    errors += check(out, n);
    times.push_back(us(t1 - t0).count());
  }
#endif

  report(times);
  if (errors) {
    std::cout << "\n" << errors << " mismatches.\nfail.\n";
    return 1;
  }
  std::cout << "\nPASS!\n";
  return 0;
}
