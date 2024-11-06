//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int CTRL_IN_SIZE = 64;
constexpr int CTRL_OUT_SIZE = 64;
constexpr int OUT_SIZE = 64;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  test_utils::check_arg_file_exists(vm, "xclbin");
  test_utils::check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_ctrlOut = xrt::bo(device, CTRL_OUT_SIZE * sizeof(int32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_ctrlIn = xrt::bo(device, CTRL_IN_SIZE * sizeof(int32_t),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint32_t beats = 1 - 1;
  uint32_t operation = 0;
  uint32_t stream_id = 0;
  auto parity = [](uint32_t n) {
    uint32_t p = 0;
    while (n) {
      p += n & 1;
      n >>= 1;
    }
    return (p % 2) == 0;
  };

  // Lock0_value
  uint32_t address = 0x0001F000;
  uint32_t header0 = stream_id << 24 | operation << 22 | beats << 20 | address;
  header0 |= (0x1 & parity(header0)) << 31;

  // Lock2_value
  address += 0x20;
  uint32_t header1 = stream_id << 24 | operation << 22 | beats << 20 | address;
  header1 |= (0x1 & parity(header1)) << 31;

  // set lock values to 2
  uint32_t data = 2;
  std::vector<uint32_t> ctrlPackets = {
      // Ctrl pkt for tile (0, 2)
      header0,
      data,
      header1,
      data,
      // Ctrl pkt for tile (0, 3)
      header0,
      data,
      header1,
      data,
      // Ctrl pkt for tile (0, 4)
      header0,
      data,
      header1,
      data,
      // Ctrl pkt for tile (0, 5)
      header0,
      data,
      header1,
      data,
  };
  void *bufctrlIn = bo_ctrlIn.map<void *>();
  memcpy(bufctrlIn, ctrlPackets.data(), ctrlPackets.size() * sizeof(int));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlIn.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_ctrlOut, bo_ctrlIn, bo_out);

  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < 32; i++) {
    uint32_t ref = 7 + i % 8;
    if (*(bufOut + i) != ref) {
      std::cout << "Error in dma output " << *(bufOut + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct dma output " << *(bufOut + i) << " == " << ref
                << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}