//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/experimental/xrt_kernel.h" // for xrt::runlist
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Two-run runlist where each run's kernel has 6 host buffer arguments (3 in /
// 3 out). Exercises the firmware command-chain walker together with host buffer
// arguments beyond the first 5. run0 (ADDONE) adds 1 to each of three inputs;
// run1 (ADDTWO) adds 2, taking run0's three outputs as its three inputs. Final
// result for a correctly executed in-order chain is input + 3.

constexpr int SIZE = 64;
constexpr int LANES = 3;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("add_multi_arg_runlist");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  auto xkernels = xclbin.get_kernels();
  auto xkernel0 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  return k.get_name() == "ADDONE";
                                });
  auto xkernel1 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  return k.get_name() == "ADDTWO";
                                });

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  auto kernel0 = xrt::kernel(context, xkernel0.get_name());
  auto kernel1 = xrt::kernel(context, xkernel1.get_name());

  auto bo_instr0 = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));
  auto bo_instr1 = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));

  // Data buffers: 3 inputs to run0, 3 shared (run0 out -> run1 in), 3 outputs.
  std::vector<xrt::bo> bo_in0, bo_mid, bo_out1;
  for (int l = 0; l < LANES; l++) {
    bo_in0.push_back(xrt::bo(device, SIZE * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3)));
    bo_mid.push_back(xrt::bo(device, SIZE * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(4)));
    bo_out1.push_back(xrt::bo(device, SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4)));
  }

  // Initialize inputs: lane l holds i + l.
  for (int l = 0; l < LANES; l++) {
    uint32_t *p = bo_in0[l].map<uint32_t *>();
    for (int i = 0; i < SIZE; i++)
      p[i] = i + l;
    bo_in0[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  void *bufInstr0 = bo_instr0.map<void *>();
  void *bufInstr1 = bo_instr1.map<void *>();
  memcpy(bufInstr0, instr_v.data(), instr_v.size() * sizeof(int));
  memcpy(bufInstr1, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  // run0 = ADDONE: args opcode, instr, ninstr, then (in, out) per lane.
  xrt::run run0(kernel0);
  run0.set_arg(0, opcode);
  run0.set_arg(1, bo_instr0);
  run0.set_arg(2, instr_v.size());
  for (int l = 0; l < LANES; l++) {
    run0.set_arg(3 + 2 * l, bo_in0[l]);
    run0.set_arg(4 + 2 * l, bo_mid[l]);
  }

  // run1 = ADDTWO: takes run0's outputs (bo_mid) as inputs.
  xrt::run run1(kernel1);
  run1.set_arg(0, opcode);
  run1.set_arg(1, bo_instr1);
  run1.set_arg(2, instr_v.size());
  for (int l = 0; l < LANES; l++) {
    run1.set_arg(3 + 2 * l, bo_mid[l]);
    run1.set_arg(4 + 2 * l, bo_out1[l]);
  }

  xrt::runlist runlist(context);
  runlist.add(run0);
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  int errors = 0;
  for (int l = 0; l < LANES; l++) {
    bo_out1[l].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint32_t *p = bo_out1[l].map<uint32_t *>();
    for (int i = 0; i < SIZE; i++) {
      uint32_t ref = (i + l) + 1 + 2;
      if (p[i] != ref) {
        if (errors < 16)
          std::cout << "Error lane " << l << " idx " << i << ": " << p[i]
                    << " != " << ref << std::endl;
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed with " << errors << " errors\n\n";
  return 1;
}
