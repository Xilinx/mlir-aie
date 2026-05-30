//===- test.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// Host runner for the AIE2P dispatch-overhead bisector example.
//
// Drives a TRIVIAL passthrough xclbin many times and reports a
// per-iteration wall-time histogram, plus the instruction-stream
// byte count and per-launch chunk count actually used. A driver
// script can sweep variants (different N_CHUNKS, different
// DENSE_BYTES) and regress to attribute per-launch wall to the
// dispatch-floor suspects described in dispatch_overhead_bisector.py.
//
// Reports (printed as KEY=VALUE lines on stdout for machine parsing,
// in addition to a human-readable summary):
//
//   dispatch_iters=<N>
//   dispatch_warmup=<W>
//   dispatch_dense_bytes=<B>
//   dispatch_n_chunks=<C>
//   dispatch_instr_bytes=<I>
//   dispatch_total_bytes=<B*C>
//   dispatch_wall_us_total=<sum>
//   dispatch_wall_us_avg=<sum/N>
//   dispatch_wall_us_min=<min>
//   dispatch_wall_us_max=<max>
//   dispatch_wall_us_p50=<p50>
//   dispatch_wall_us_p90=<p90>
//
//===----------------------------------------------------------------------===//

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> read_instr_binary(const std::string &path) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh)
    throw std::runtime_error("failed to open instructions file: " + path);
  std::vector<uint32_t> out;
  uint32_t word = 0;
  while (fh.read(reinterpret_cast<char *>(&word), sizeof(word)))
    out.push_back(word);
  return out;
}

struct Args {
  std::string xclbin;
  std::string instr;
  std::string kernel = "MLIR_AIE";
  size_t dense_bytes = 4096;
  int n_chunks = 1;
  int iters = 100;
  int warmup = 5;
  // Optional: emit a single CSV row per iter to this path. Off by
  // default to keep stdout clean.
  std::string csv_out;
};

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "-x" || k == "--xclbin") a.xclbin = next();
    else if (k == "-i" || k == "--instr") a.instr = next();
    else if (k == "-k" || k == "--kernel") a.kernel = next();
    else if (k == "--dense-bytes") a.dense_bytes = std::stoull(next());
    else if (k == "--n-chunks") a.n_chunks = std::stoi(next());
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--csv-out") a.csv_out = next();
    else throw std::runtime_error("unknown arg: " + k);
  }
  if (a.xclbin.empty() || a.instr.empty())
    throw std::runtime_error("required: -x <xclbin> -i <instr>");
  if (a.iters < 1)
    throw std::runtime_error("--iters must be >= 1");
  if (a.warmup < 0)
    throw std::runtime_error("--warmup must be >= 0");
  if (a.n_chunks < 1)
    throw std::runtime_error("--n-chunks must be >= 1");
  return a;
}

double percentile(std::vector<double> sorted, double pct) {
  if (sorted.empty())
    return 0.0;
  size_t idx = static_cast<size_t>((pct / 100.0) * (sorted.size() - 1));
  if (idx >= sorted.size())
    idx = sorted.size() - 1;
  return sorted[idx];
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse(argc, argv);

  auto instr_v = read_instr_binary(args.instr);
  size_t instr_bytes = instr_v.size() * sizeof(uint32_t);
  size_t total_bytes = args.dense_bytes * static_cast<size_t>(args.n_chunks);

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(args.xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, args.kernel);

  // BO ordering matches the aie.runtime_sequence args:
  //   arg0 (bo_instr = group_id(1)) -> instructions buffer
  //   arg1 (bo_input = group_id(3)) -> input  (bisector_in shim alloc)
  //   arg2 (bo_output = group_id(4)) -> output (bisector_out shim alloc)
  //   arg3 (bo_unused = group_id(5)) -> unused (third sequence arg)
  auto bo_instr = xrt::bo(device, instr_bytes,
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, total_bytes,
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, total_bytes,
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_unused = xrt::bo(device, 1,
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_bytes);

  uint8_t *bufInput = bo_input.map<uint8_t *>();
  // Deterministic non-zero pattern so silent corruption would be
  // visible if checked; the bisector's verdict is timing — it does
  // not gate on byte-equality, only on the runner not crashing.
  for (size_t i = 0; i < total_bytes; ++i)
    bufInput[i] = static_cast<uint8_t>(i & 0xff);

  uint8_t *bufOutput = bo_output.map<uint8_t *>();
  std::memset(bufOutput, 0, total_bytes);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const unsigned int opcode = 3;
  unsigned num_iter = args.iters + args.warmup;
  std::vector<double> per_iter_us;
  per_iter_us.reserve(args.iters);

  for (unsigned it = 0; it < num_iter; ++it) {
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output,
                      bo_unused);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    if (it < (unsigned)args.warmup)
      continue;

    double npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    per_iter_us.push_back(npu_time);
  }

  // Sync once at the end (we don't need the output for the verdict,
  // but sync once so XRT teardown doesn't trip on a dirty cache).
  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  double sum = 0.0;
  double minv = per_iter_us.empty() ? 0.0 : per_iter_us[0];
  double maxv = per_iter_us.empty() ? 0.0 : per_iter_us[0];
  for (double v : per_iter_us) {
    sum += v;
    if (v < minv) minv = v;
    if (v > maxv) maxv = v;
  }
  double avg = per_iter_us.empty() ? 0.0 : sum / per_iter_us.size();

  std::vector<double> sorted = per_iter_us;
  std::sort(sorted.begin(), sorted.end());
  double p50 = percentile(sorted, 50.0);
  double p90 = percentile(sorted, 90.0);

  if (!args.csv_out.empty()) {
    std::ofstream csv(args.csv_out);
    if (!csv)
      throw std::runtime_error("failed to open csv-out: " + args.csv_out);
    csv << "iter,wall_us\n";
    for (size_t i = 0; i < per_iter_us.size(); ++i)
      csv << i << "," << per_iter_us[i] << "\n";
  }

  // KEY=VALUE form for machine parsing.
  std::cout << "dispatch_iters=" << args.iters << "\n";
  std::cout << "dispatch_warmup=" << args.warmup << "\n";
  std::cout << "dispatch_dense_bytes=" << args.dense_bytes << "\n";
  std::cout << "dispatch_n_chunks=" << args.n_chunks << "\n";
  std::cout << "dispatch_instr_bytes=" << instr_bytes << "\n";
  std::cout << "dispatch_total_bytes=" << total_bytes << "\n";
  std::cout << "dispatch_wall_us_total=" << sum << "\n";
  std::cout << "dispatch_wall_us_avg=" << avg << "\n";
  std::cout << "dispatch_wall_us_min=" << minv << "\n";
  std::cout << "dispatch_wall_us_max=" << maxv << "\n";
  std::cout << "dispatch_wall_us_p50=" << p50 << "\n";
  std::cout << "dispatch_wall_us_p90=" << p90 << "\n";
  std::cout << "PASS!" << std::endl;
  return 0;
}
