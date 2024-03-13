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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

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
      "length,l", po::value<long long>()->default_value(4096),
      "the length of the transfer in int32_t")(
      "iters", po::value<int>()->default_value(10))(
      "warmup", po::value<int>()->default_value(1));
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

  long long N = vm["length"].as<long long>();
  if ((N % 1024)) {
    std::cerr << "Length must be a multiple of 1024, but got " << N << "."
              << std::endl;
    return 1;
  }

  int iters = vm["iters"].as<int>();
  int warmup = vm["warmup"].as<int>();

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
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
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
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inA = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(2));
  auto bo_inB = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  int32_t *bufInA = bo_inA.map<int32_t *>();
  std::vector<uint32_t> srcVecA(N);
  for (int i = 0; i < N; i++) {
    srcVecA[i] = i + 1;
  }
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  typedef std::chrono::system_clock::duration duration_t;

  std::vector<duration_t> runtimes(iters);
  int errors = 0;

  for (int i = 0; i < iters + warmup; i++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel." << std::endl;
    auto start = std::chrono::system_clock::now();
    auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
    run.wait();
    auto stop = std::chrono::system_clock::now();
    if (i < warmup) {
      continue;
    }
    runtimes[i - warmup] = stop - start;

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint32_t *bufOut = bo_out.map<uint32_t *>();

    for (uint32_t i = 0; i < N; i++) {
      uint32_t ref = (i + 1);
      if (*(bufOut + i) != ref) {
        errors++;
      }
    }
  }

  duration_t sum =
      std::accumulate(runtimes.begin(), runtimes.end(), duration_t(0));
  duration_t mean = sum / runtimes.size();
  std::pair<std::vector<duration_t>::iterator,
            std::vector<duration_t>::iterator>
      minmax = std::minmax_element(runtimes.begin(), runtimes.end());
  duration_t min = *minmax.first;
  duration_t max = *minmax.second;

  long long n_bytes = srcVecA.size() * sizeof(srcVecA[0]);
  double mean_bps =
      (double)n_bytes /
      std::chrono::duration_cast<std::chrono::duration<double>>(mean).count();
  double max_bps =
      (double)n_bytes /
      std::chrono::duration_cast<std::chrono::duration<double>>(min).count();
  double min_bps =
      (double)n_bytes /
      std::chrono::duration_cast<std::chrono::duration<double>>(max).count();

  std::cout << iters << " runs, each pushed "
            << srcVecA.size() * sizeof(srcVecA[0]) << " bytes of data."
            << std::endl
            << "Mean: " << std::setw(8) << mean.count() << " us  / " << mean_bps
            << " bytes/s" << std::endl
            << "Min:  " << std::setw(8) << min.count() << " us  / " << max_bps
            << " bytes/s" << std::endl
            << "Max:  " << std::setw(8) << max.count() << " us  / " << min_bps
            << " bytes/s" << std::endl;

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl;
    return 1;
  }
}
