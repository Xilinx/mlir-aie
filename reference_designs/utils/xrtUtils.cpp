//===- xrtUtils.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


#include "xrtUtils.h"

#include <fstream>
#include <sstream>
#include <iostream>

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

void initXrtLoadKernel(xrt::device &device, xrt::kernel &kernel, int verbosity, std::string xclbinFileName, std::string kernelNameInXclbin)
{
    // Start the XRT test code
    // Get a device handle
    unsigned int device_index = 0;
    device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin: " << xclbinFileName << "\n";
    auto xclbin = xrt::xclbin(xclbinFileName);

    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << kernelNameInXclbin << "\n";

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(), 
                               [kernelNameInXclbin](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(kernelNameInXclbin, 0) == 0;
                               });
    auto kernelName = xkernel.get_name();

    if (verbosity >= 1)
        std::cout << "Registering xclbin: " << xclbinFileName << "\n";

    device.register_xclbin(xclbin);

    // get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context.\n";
    xrt::hw_context context(device, xclbin.get_uuid());

    // get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << "\n";
    kernel = xrt::kernel(context, kernelName);

    return;
}