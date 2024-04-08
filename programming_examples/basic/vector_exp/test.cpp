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
#include <math.h>
#include <sstream>
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr bool VERIFY = false;

constexpr int IN_SIZE = 65536;
constexpr int OUT_SIZE = IN_SIZE;

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

static inline std::bfloat16_t random_bfloat16_t() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

bool nearly_equal(std::bfloat16_t a, std::bfloat16_t b) {
  std::bfloat16_t diff = fabs(a - b);
  if ((diff / a) < 0.01)
    return true;
  else
    return false;
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
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(std::bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(std::bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  std::bfloat16_t *bufA = bo_inA.map<std::bfloat16_t *>();
  std::vector<std::bfloat16_t> AVec(IN_SIZE);
  for (int i = 0; i < IN_SIZE; i++)
    AVec[i] = random_bfloat16_t();
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(std::bfloat16_t)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int sticky_errors = 0;

  unsigned num_iter = 256;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_out);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::bfloat16_t *bufOut = bo_out.map<std::bfloat16_t *>();

    int errors = 0;

    if (VERIFY) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      for (uint32_t i = 0; i < IN_SIZE; i++) {
        std::bfloat16_t ref = exp(AVec[i]);
        if (!nearly_equal(*(bufOut + i), ref)) {
          std::cout << "Error in " << i << " output " << *(bufOut + i)
                    << " != " << ref << " actual e^" << AVec[i] << " : "
                    << exp(AVec[i]) << std::endl;
          errors++;
          sticky_errors++;
        } else {
          if (verbosity >= 2)
            std::cout << "Correct " << i << " output " << *(bufOut + i)
                      << " == " << ref << std::endl;
        }
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: vector-scalar results not verified."
                  << std::endl;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;

    if (VERIFY) {
      if (!errors) {
        std::cout << iter << ": pass!\n";
      } else {
        std::cout << iter << ": fail! " << errors << " errors\n";
      }
    }
  }

  std::cout << "Avg NPU exec time: " << npu_time_total / num_iter << "us."
            << std::endl;
  std::cout << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;

  // Let's figure out how many cycles it takes a core to do a single e^x
  // There are 4 cores, so the total number of e^x's it does is one quarter of
  // the test size

  int per_core_calcs = IN_SIZE / 4;
  float avg_npu_time = npu_time_total / num_iter;
  float avg_npu_clocks =
      avg_npu_time / 1.0E-3; // Time is in uS, but the AIE is clocked in nS
  float clocks_per_calc = avg_npu_clocks / per_core_calcs;
  std::cout << "Clocks per calc " << clocks_per_calc << std::endl;

  // Lets benchmark the CPU
  float cpu_time_total = 0;
  float cpu_time_min = 9999999;
  float cpu_time_max = 0;
  for (unsigned iter = 0; iter < num_iter; iter++) {

    std::vector<std::bfloat16_t> AVec(IN_SIZE);
    std::vector<std::bfloat16_t> ResVec(IN_SIZE);
    for (int i = 0; i < IN_SIZE; i++) {
      AVec[i] = random_bfloat16_t();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < IN_SIZE; i++) {
      ResVec[i] = exp(AVec[i]);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    float cpu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    cpu_time_total += cpu_time;
    cpu_time_min = (cpu_time < cpu_time_min) ? cpu_time : cpu_time_min;
    cpu_time_max = (cpu_time > cpu_time_max) ? cpu_time : cpu_time_max;
  }
  std::cout << "Avg CPU exec time: " << cpu_time_total / num_iter << "us."
            << std::endl;
  std::cout << "Min CPU matmul time: " << cpu_time_min << "us." << std::endl;
  std::cout << "Max CPU matmul time: " << cpu_time_max << "us." << std::endl;

  if (VERIFY) {
    if (!sticky_errors) {
      std::cout << std::endl << "PASS!" << std::endl << std::endl;
      return 0;
    } else {
      std::cout << std::endl << "FAIL." << std::endl << std::endl;
      return 1;
    }
  } else {
    std::cout << "Verification skipped, but I'm sure it worked.  I trust in you"
              << std::endl;
  }
  return 0;
}
