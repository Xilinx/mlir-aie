//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;
constexpr std::uint64_t DDR_AIE_ADDR_OFFSET = std::uint64_t{0x80000000};

void check_arg_file_exists(po::variables_map &vm_in, std::string name) {
  if (!vm_in.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  } else {
    std::ifstream test(vm_in[name].as<std::string>());
    if (!test) {
      throw std::runtime_error("The " + name + " file " +
                               vm_in[name].as<std::string>() +
                               " does not exist.\n");
    }
  }
}

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

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
      "path of file containing userspace instructions to be sent to the LX6")(
      "length,l", po::value<int>()->default_value(4096),
      "the length of the transfer in int32_t");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << std::endl;
    return 1;
  }

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  int N = vm["length"].as<int>();
  if ((N % 1024)) {
    std::cerr << "Length must be a multiple of 1024." << std::endl;
    return 1;
  }

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>()
              << std::endl;
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>()
              << std::endl;
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
    std::cout << "Getting hardware context." << std::endl;
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  std::vector<uint32_t> outVec;
  int32_t *bufOut = bo_out.map<int32_t *>();
  for (int i = 0; i < N; i++)
    outVec.push_back(i + 0x1011);
  memcpy(bufOut, outVec.data(), (outVec.size() * sizeof(uint32_t)));

  std::cout << "DDR AIE Address Offset: " << std::hex << DDR_AIE_ADDR_OFFSET
            << std::endl;
  std::cout << "bo_out.address() + DDR_AIE_ADDR_OFFSET: " << std::hex
            << bo_out.address() + DDR_AIE_ADDR_OFFSET << std::endl;

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  std::cout << std::dec;
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;

  uint64_t opcode = 3;
  xrt::run aie_run(kernel);
  aie_run.set_arg(0, opcode);
  aie_run.set_arg(1, bo_instr);
  aie_run.set_arg(2, uint32_t(bo_instr.size() / sizeof(int)));
  aie_run.set_arg(3, bo_out);
  aie_run.start();
  std::cv_status wait_ret = aie_run.wait2(2000 * std::chrono::milliseconds{1});
  if (wait_ret == std::cv_status::timeout) {
    std::cerr << "Timeout waiting for kernel to finish." << std::endl;
  }

  if (verbosity >= 1)
    std::cout << "Kernel Done. return code: " << aie_run.return_code()
              << std::endl;

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  bufOut = bo_out.map<int32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < N; i++) {
    uint32_t ref = 42;
    if (*(bufOut + i) != ref) {
      if (errors < 10)
        std::cout << "Mismatch at " << i << ": " << *(bufOut + i)
                  << " != " << ref << std::endl;
      errors++;
    }
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " / " << N << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
