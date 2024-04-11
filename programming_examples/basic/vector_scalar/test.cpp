//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr bool VERIFY = true;
constexpr bool ENABLE_TRACING = false;
constexpr int TRACE_SIZE = 8192;

constexpr int IN_SIZE = 4096;
constexpr int OUT_SIZE = ENABLE_TRACING ? IN_SIZE + TRACE_SIZE / 4 : IN_SIZE;
// constexpr int OUT_SIZE = 4096;

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

void write_out_trace(uint32_t *bufOut, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut =
      (uint32_t *)((char *)bufOut + sizeof(uint32_t) * IN_SIZE);
  for (int i = 0; i < TRACE_SIZE / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  if (ENABLE_TRACING) {
    desc.add_options()("trace,t",
                       po::value<std::string>()->default_value("trace.txt"),
                       "where to store trace output");
  }

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
  if (ENABLE_TRACING) {
    desc.add_options()("trace,t",
                       po::value<std::string>()->default_value("trace.txt"),
                       "where to store trace output");
  }

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

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

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
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inB = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  int32_t *bufInA = bo_inA.map<int32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  if (VERIFY) {
    if (verbosity >= 1) {
      std::cout << "Verifying results ..." << std::endl;
    }
    for (uint32_t i = 0; i < IN_SIZE; i++) {
      uint32_t ref = (i + 1) * 3;
      if (*(bufOut + i) != ref) {
        std::cout << "Error in output " << *(bufOut + i) << " != " << ref
                  << std::endl;
        errors++;
      } else {
        std::cout << "Correct output " << *(bufOut + i) << " == " << ref
                  << std::endl;
      }
    }
  } else {
    if (verbosity >= 1)
      std::cout << "WARNING: vector-scalar results not verified." << std::endl;
  }

  if (ENABLE_TRACING) {
    write_out_trace(bufOut, vm["trace"].as<std::string>());
  }

  if (VERIFY && !errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
