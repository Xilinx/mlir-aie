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
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr bool VERIFY = true;

constexpr int TEST_SIZE = 65536;
constexpr int TILE_SIZE = 1024;
constexpr int NUM_TILES = TEST_SIZE / TILE_SIZE;

constexpr int SF_BLOCK_SIZE = 32;

constexpr int TOTAL_TILE_SIZE =
    (TILE_SIZE / 2) + (TILE_SIZE / SF_BLOCK_SIZE) * 2;

constexpr int IN_SIZE = (TEST_SIZE / 2) + (TEST_SIZE / SF_BLOCK_SIZE) * 2;
constexpr int OUT_SIZE = TEST_SIZE * 2;

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
  auto bo_in = xrt::bo(device, IN_SIZE * sizeof(char), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(2));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(char), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  char *bufA = bo_in.map<char *>();
  std::vector<char> AVec(IN_SIZE);

  std::vector<std::int8_t> A_private;
  std::vector<std::bfloat16_t> A_sf;

  for (int t = 0; t < NUM_TILES; t++) {
    for (int pr = 0; pr < TILE_SIZE / 2; pr++) {
      std::int8_t lower = (rand()) & 0xf;
      std::int8_t upper = (rand()) & 0xf;
      AVec[t * TOTAL_TILE_SIZE + pr] = ((upper) << 4) + lower;
      A_private.push_back(lower);
      A_private.push_back(upper);
      if (verbosity >= 2) {
        if (t == 0)
          std::cout << std::hex << (t * TOTAL_TILE_SIZE + pr) << " : "
                    << ((upper << 4) + lower) << std::dec << std::endl;
      }
    }
    for (int isf = 0; isf < TILE_SIZE / SF_BLOCK_SIZE; isf++) {
      std::bfloat16_t sf =
          std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
      std::uint16_t bits = *((std::uint16_t *)&sf);
      std::int8_t upper = (std::int8_t)(bits >> 8);
      std::int8_t lower = (std::int8_t)(bits & 0x00ff);
      AVec[t * TOTAL_TILE_SIZE + TILE_SIZE / 2 + isf * 2] = lower;
      AVec[t * TOTAL_TILE_SIZE + TILE_SIZE / 2 + isf * 2 + 1] = upper;
      A_sf.push_back(sf);
      if (verbosity >= 2) {
        if (t == 0)
          std::cout << std::hex
                    << (t * TOTAL_TILE_SIZE + TILE_SIZE / 2 + isf * 2)
                    << " and +1 :" << sf << std::dec << std::endl;
      }
    }
  }

  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(char)));

  if (verbosity >= 2)
    std::cout << "Pre run values in  " << std::hex << int(bufA[0]) << ", "
              << int(bufA[1]) << ", " << int(bufA[2]) << std::dec << std::endl;

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int sticky_errors = 0;

  unsigned num_iter = 16;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto run = kernel(bo_instr, instr_v.size(), bo_in, bo_out);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::bfloat16_t *bufOut = bo_out.map<std::bfloat16_t *>();

    int errors = 0;

    if (verbosity >= 2) {
      std::cout << "First values in  " << std::hex << int(bufA[0]) << ", "
                << int(bufA[1]) << ", " << int(bufA[2]) << std::dec
                << std::endl;
      std::cout << "First sf values in  " << std::hex << int(bufA[512]) << ", "
                << int(bufA[513]) << ", " << int(bufA[514]) << ", "
                << int(bufA[515]) << std::dec << std::endl;
      std::cout << "First values out " << std::hex << bufOut[0] << ", "
                << bufOut[1] << ", " << bufOut[2] << ", " << bufOut[3]
                << std::dec << std::endl;
      std::cout << "Second values out " << std::hex << bufOut[32] << ", "
                << bufOut[33] << ", " << bufOut[34] << ", " << bufOut[35]
                << std::dec << std::endl;

      std::cout << "Reference values " << std::hex << A_private[0] << ", "
                << A_private[1] << std::dec << std::endl;
      std::cout << "Reference sfs    " << A_sf[0] << ", " << A_sf[1]
                << std::endl;
    }
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;

    if (VERIFY) {
      for (int t = 0; t < NUM_TILES; t++) {
        for (int pr = 0; pr < TILE_SIZE; pr++) {
          std::bfloat16_t sf = A_sf[t * SF_BLOCK_SIZE + pr / SF_BLOCK_SIZE];
          int val = (int)(A_private[t * TILE_SIZE + pr]);
          if (val >= 8)
            val = (val & 0x7) - 8; // Two's complement, but threes a crowd
          std::bfloat16_t scaled = sf * val;

          std::bfloat16_t from_AIE = bufOut[(t * TILE_SIZE) + pr];

          // These will not exactly match
          // The default rounding mode in AIE2 is to truncate, so we will get
          // off by one errors.
          std::uint16_t from_AIE_raw =
              *reinterpret_cast<std::uint16_t *>(&from_AIE);
          std::uint16_t scaled_raw =
              *reinterpret_cast<std::uint16_t *>(&scaled);

          std::bfloat16_t abs_diff = fabs(from_AIE - scaled);
          if ((abs_diff / fabs(from_AIE)) > 0.01) {
            std::cout << "Tile " << t << ":" << pr << " From AIE "
                      << std::setprecision(12) << from_AIE << " ref "
                      << std::setprecision(12) << scaled << " from "
                      << std::setprecision(12) << sf << "*" << std::hex << val
                      << std::dec << std::endl;
            std::cout << "Tile " << t << ":" << pr << " From AIE " << std::hex
                      << from_AIE_raw << " ref " << scaled_raw << " from "
                      << *reinterpret_cast<std::uint16_t *>(&sf) << "*" << val
                      << std::dec << std::endl;
            errors++;
          }
        }
      }
    }

    if (VERIFY && !errors) {
      std::cout << iter << ": pass!\n";
    } else {
      std::cout << iter << ": fail! " << errors << " errors\n";
    }
  }

  std::cout << "Avg NPU exec time: " << npu_time_total / num_iter << "us."
            << std::endl;
  std::cout << "Min NPU matmul time: " << npu_time_min << "us." << std::endl;
  std::cout << "Max NPU matmul time: " << npu_time_max << "us." << std::endl;
  if (VERIFY && !sticky_errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nFAIL.\n\n";
    return 1;
  }
}
