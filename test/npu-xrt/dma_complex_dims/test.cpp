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
#include <iostream>
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
      "m", po::value<int>()->default_value(32),
      "m, number of rows in the small tile")(
      "k", po::value<int>()->default_value(64),
      "k, number of columns in the small tile")(
      "K", po::value<int>()->default_value(256),
      "K, number of columns in the large tile")(
      "r", po::value<int>()->default_value(8),
      "r, number of columns in the large tile")(
      "s", po::value<int>()->default_value(8),
      "s, number of columns in the large tile");

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

  // input arguments m, k, K, r, s
  int m = vm["m"].as<int>();
  int k = vm["k"].as<int>();
  int K = vm["K"].as<int>();
  int r = vm["r"].as<int>();
  int s = vm["s"].as<int>();

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
  auto bo_inA = xrt::bo(device, m * K * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_inB = xrt::bo(device, m * K * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, m * K * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  // xrt input buffer mapped to input buffer A
  int32_t *bufInA = bo_inA.map<int32_t *>();

  // input source vector A
  std::vector<int32_t> srcVecA(m * K);

  // <<<<<<<<< The code below emulates the "of_in_shim_to_mem" OBjFifo in line
  // 57 >>>>>>> calculate the number of tiles in the 'K' dimension
  int K_div_k = K / k;

  int index = 0;

  // write the input data to the input vector in pre-tiled format
  for (int tile_k = 0; tile_k < K_div_k; tile_k++) {
    for (int ii = 0; ii < m; ii++) {
      for (int jj = 0; jj < k; jj++) {
        // here just copy of index for easy debug
        // later replace with random data
        srcVecA[index] = index;
        index++;
      }
    }
  }

  // copy the input data to the input buffer A
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(int32_t)));

  // copy instructions
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // sync instructions and input buffer
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // run kernel
  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  run.wait();

  // sync output buffer after kernel running
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // map xrt output buffer to output buffer
  int32_t *bufOut = bo_out.map<int32_t *>();

  // create a vector for the output data
  std::vector<int32_t> OutVec(m * K);

  int errors = 0;

  // create a reference vector to verify the data
  std::vector<int32_t> refVecA(m * K);

  // Final transformation you want to have for verification
  // assumes data are pre-tiled (in m*k tiles).
  // See line 176 in this file, where data are send to the npu pre-tiled.

  int m_div_r = m / r;
  int k_div_s = k / s;

  // because assumed that data are pre-tiled,
  // the input index just needs to be increased in every iteration
  int in_index = 0;

  for (int tile_k = 0; tile_k < K_div_k; tile_k++) {
    for (int ii = 0; ii < m_div_r; ii++) {
      for (int jj = 0; jj < k_div_s; jj++) {
        for (int r_ii = 0; r_ii < r; r_ii++) {
          for (int s_jj = 0; s_jj < s; s_jj++) {
            // transformation index
            int transf_index =
                (tile_k * (m * k)) + // offset for each m*k tile in (big) 'K'
                                     // dimension (MemTile stores multiple m*k
                                     // tiles, each one processed in CompTile)
                (ii * (r * k) + jj * (s) + r_ii * (k) +
                 s_jj * (1)); // transformation index for r*s tiled acccess
                              // pattern of m*k tile

            // store the access pattern to the reference vector for verification
            refVecA[in_index] = srcVecA[transf_index];

            in_index++;
          }
        }
      }
    }
  }

  // copy output data to the output vector
  memcpy(OutVec.data(), bufOut, (OutVec.size() * sizeof(int32_t)));

  // verification
  for (int i = 0; i < m * K; i++) {
    if (OutVec[i] != refVecA[i]) {
      std::cout << "ref = " << (int)refVecA[i]
                << " NPU output = " << (int)OutVec[i] << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
