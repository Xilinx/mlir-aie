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

#include "experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

namespace po = boost::program_options;

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

std::vector<uint32_t> load_instr_binary(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  // read size of file, reserve space in  instr_v, then read the file into
  // instr_v
  instr_file.seekg(0, instr_file.end);
  int size = instr_file.tellg();
  instr_file.seekg(0, instr_file.beg);
  std::vector<uint32_t> instr_v(size / 4);
  instr_file.read(reinterpret_cast<char *>(instr_v.data()), size);
  return instr_v;
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")("verbosity,v",
                               po::value<int>()->default_value(0),
                               "the verbosity of the output")(
      "instr0,i", po::value<std::string>()->required(),
      "path to instructions for kernel0")("instr1,j",
                                          po::value<std::string>()->required(),
                                          "path to instructions for kernel1")(
      "cfg,c", po::value<std::string>()->required(), "txn binary path");
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

  std::vector<uint32_t> instr_0_v =
      load_instr_sequence(vm["instr0"].as<std::string>());

  std::vector<uint32_t> instr_1_v =
      load_instr_sequence(vm["instr1"].as<std::string>());

  std::vector<uint32_t> cfg_1_v =
      load_instr_binary(vm["cfg"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1) {
    std::cout << "Sequence instr 0 count: " << instr_0_v.size() << "\n";
    std::cout << "Sequence instr 1 count: " << instr_1_v.size() << "\n";
    std::cout << "Sequence cfg count: " << cfg_1_v.size() << "\n";
  }

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel0 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  std::cout << "Name: " << name << std::endl;
                                  return name == "ADDONE";
                                });
  auto kernelName0 = xkernel0.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  auto kernel0 = xrt::kernel(context, kernelName0);

  auto bo_instr_0 = xrt::bo(device, instr_0_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));
  auto bo_inA_0 = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3));
  auto bo_out_0 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(4));

  auto bo_instr_1 = xrt::bo(device, instr_1_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));
  auto bo_cfg_1 = xrt::bo(device, cfg_1_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));
  auto bo_inA_1 = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3));
  auto bo_out_1 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initializing the input vectors
  std::vector<uint32_t> srcVecA;
  std::vector<uint32_t> srcVecA_1;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);

  for (int i = 0; i < IN_SIZE; i++)
    srcVecA_1.push_back(i + 2);

  // Getting handles to the input data BOs and copying input data to them
  uint32_t *bufInA_0 = bo_inA_0.map<uint32_t *>();
  uint32_t *bufInA_1 = bo_inA_1.map<uint32_t *>();
  memcpy(bufInA_0, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));
  memcpy(bufInA_1, srcVecA_1.data(), (srcVecA_1.size() * sizeof(uint32_t)));

  // Getting handles to the instruction sequence BOs and copy data to them
  void *bufInstr_0 = bo_instr_0.map<void *>();
  void *bufInstr_1 = bo_instr_1.map<void *>();
  void *bufCfg_1 = bo_cfg_1.map<void *>();
  memcpy(bufInstr_0, instr_0_v.data(), instr_0_v.size() * sizeof(int));
  memcpy(bufInstr_1, instr_1_v.data(), instr_1_v.size() * sizeof(int));
  memcpy(bufCfg_1, cfg_1_v.data(), cfg_1_v.size() * sizeof(int));

  // Synchronizing BOs
  bo_instr_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_cfg_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  // Creating a runlist to contain two seperate runs
  xrt::runlist runlist = xrt::runlist(context);

  // Creating the first run
  xrt::run run0 = xrt::run(kernel0);
  run0.set_arg(0, opcode);
  run0.set_arg(1, bo_instr_0);
  run0.set_arg(2, instr_0_v.size());
  run0.set_arg(3, bo_inA_0);
  run0.set_arg(4, bo_out_0);
  run0.set_arg(5, 0);
  run0.set_arg(6, 0);
  run0.set_arg(7, 0);

  xrt::run run1_cfg = xrt::run(kernel0);
  run1_cfg.set_arg(0, opcode);
  run1_cfg.set_arg(1, bo_cfg_1);
  run1_cfg.set_arg(2, cfg_1_v.size());
  run1_cfg.set_arg(3, 0);
  run1_cfg.set_arg(4, 0);
  run1_cfg.set_arg(5, 0);
  run1_cfg.set_arg(6, 0);
  run1_cfg.set_arg(7, 0);

  // Creating the second run
  xrt::run run1 = xrt::run(kernel0);
  run1.set_arg(0, opcode);
  run1.set_arg(1, bo_instr_1);
  run1.set_arg(2, instr_1_v.size());
  run1.set_arg(3, bo_inA_1);
  run1.set_arg(4, bo_out_1);
  run1.set_arg(5, 0);
  run1.set_arg(6, 0);
  run1.set_arg(7, 0);

  // Executing and waiting on the runlist
  runlist.add(run0);
  runlist.add(run1_cfg);
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  // Synchronizing the output BOs
  bo_out_0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut_0 = bo_out_0.map<uint32_t *>();
  uint32_t *bufOut_1 = bo_out_1.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < 64; i++) {
    uint32_t ref = (i + 1) + 1;
    if (*(bufOut_0 + i) != ref) {
      std::cout << "Error in output " << *(bufOut_0 + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut_0 + i) << " == " << ref
                << std::endl;
    }
  }

  for (uint32_t i = 0; i < 64; i++) {
    uint32_t ref = (i + 2) + 102;
    if (*(bufOut_1 + i) != ref) {
      std::cout << "Error in output " << *(bufOut_1 + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut_1 + i) << " == " << ref
                << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed with " << errors << " errors \n\n";
    return 1;
  }
}
