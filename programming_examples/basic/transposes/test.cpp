//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Strategy-agnostic transpose verifier — operates on bo_in / bo_out as
// raw byte buffers and checks that the on-device output is the full
// (M x K)^T = (K x M) transpose of the input.  --dtype-bytes selects
// element granularity (1, 2, or 4).
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Transpose Test",
                           "Strategy-agnostic transpose verifier");

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance MLIR_AIE)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i", "path of file containing userspace instructions",
      cxxopts::value<std::string>())(
      "rows,M", "M, number of rows in the input matrix",
      cxxopts::value<int>()->default_value("64"))(
      "cols,K", "K, number of columns in the input matrix",
      cxxopts::value<int>()->default_value("64"))(
      "dtype-bytes,b", "element size in bytes (1, 2, or 4)",
      cxxopts::value<int>()->default_value("4"));

  auto vm = options.parse(argc, argv);
  if (vm.count("help")) {
    std::cout << options.help() << std::endl;
    return 1;
  }
  if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
    std::cerr << "Error: Required options missing\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  const uint32_t M = vm["M"].as<int>();
  const uint32_t K = vm["K"].as<int>();
  const uint32_t Nel = M * K;
  const uint32_t bpe = vm["dtype-bytes"].as<int>();
  if (bpe != 1 && bpe != 2 && bpe != 4) {
    std::cerr << "--dtype-bytes must be 1, 2, or 4 (got " << bpe << ")\n";
    return 1;
  }

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                   std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA =
      xrt::bo(device, Nel * bpe, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB =
      xrt::bo(device, Nel * bpe, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, Nel * bpe, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Fill input with a deterministic byte pattern.  We only ever compare
  // bytes against bytes, so the per-element type doesn't matter — what
  // matters is that the same input bytes show up at the transposed
  // output positions.
  uint8_t *bufInA = bo_inA.map<uint8_t *>();
  for (uint32_t i = 0; i < Nel * bpe; i++)
    bufInA[i] = static_cast<uint8_t>((i + 1) & 0xFF);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  run.wait();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint8_t *bufOut = bo_out.map<uint8_t *>();
  int errors = 0;
  // Output is (K x M) with the same element size.  For each (i, j) of
  // the K x M output, compare bpe bytes against the (j, i) input element.
  for (uint32_t i = 0; i < K; i++) {
    for (uint32_t j = 0; j < M; j++) {
      const uint8_t *src = bufInA + (j * K + i) * bpe;
      const uint8_t *dst = bufOut + (i * M + j) * bpe;
      if (memcmp(src, dst, bpe) != 0) {
        if (errors < 16) {
          std::cout << "mismatch at (i=" << i << ", j=" << j << ")\n";
        } else if (errors == 16) {
          std::cout << "...\n[further mismatches truncated]\n";
        }
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\n" << errors << " mismatches.\nfail.\n\n";
  return 1;
}
