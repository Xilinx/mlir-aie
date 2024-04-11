//===- test_utils.cpp ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code


#include "test_utils.h"

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void test_utils::check_arg_file_exists(po::variables_map &vm_in, std::string name) {
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

void test_utils::add_default_options(po::options_description &desc) {
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions sent to the NPU")(
      "verify", po::value<bool>()->default_value(true),
      "whether to verify the AIE computed output")(
      "iters", po::value<int>()->default_value(1))(
      "warmup", po::value<int>()->default_value(0))(
      "trace_sz,t", po::value<int>()->default_value(0))(
      "trace_file", po::value<std::string>()->default_value("trace.txt"),
      "where to store trace output");
}

void test_utils::parse_options(int argc, const char *argv[], po::options_description &desc,
                   po::variables_map &vm) {
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      std::exit(1);
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    std::exit(1);
  }

  try {
    check_arg_file_exists(vm, "xclbin");
    check_arg_file_exists(vm, "instr");
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
  }
}

// --------------------------------------------------------------------------
// AIE Specifics
// --------------------------------------------------------------------------

std::vector<uint32_t> test_utils::load_instr_sequence(std::string instr_path) {
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


// --------------------------------------------------------------------------
// XRT 
// --------------------------------------------------------------------------
void test_utils::init_xrt_load_kernel(xrt::device &device, 
                                      xrt::kernel &kernel, int verbosity,
                                      std::string xclbinFileName,
                                      std::string kernelNameInXclbin) {
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
                               [kernelNameInXclbin, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(kernelNameInXclbin, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << xclbinFileName << "\n";

  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  kernel = xrt::kernel(context, kernelName);

  return;
}

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

static inline std::int16_t test_utils::random_int16_t() {
  return (std::int16_t)rand() % 0x10000;
}


// --------------------------------------------------------------------------
// Tracing
// --------------------------------------------------------------------------
void test_utils::write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

