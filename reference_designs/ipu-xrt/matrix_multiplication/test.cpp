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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int M = 256;
constexpr int K = 256;
constexpr int N = 256;

constexpr int A_VOLUME = M * K;
constexpr int B_VOLUME = N * K;
constexpr int C_VOLUME = M * N;

using A_DATATYPE = std::bfloat16_t;
using B_DATATYPE = std::bfloat16_t;
using C_DATATYPE = float;

constexpr int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
constexpr int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

constexpr bool VERIFY = true;
constexpr bool ENABLE_TRACING = false;
constexpr int TRACE_SIZE = 4096; 

constexpr int OUT_SIZE = C_SIZE + (ENABLE_TRACING ? TRACE_SIZE : 0);

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

static inline std::int16_t random_int16_t() {
  return ((std::int16_t)rand() % 0x10000);
}

static inline std::bfloat16_t random_bfloat16_t() {
  return ((std::bfloat16_t)rand() / (std::bfloat16_t)INT_MAX);
}

template <typename Tin, typename Tout>
void matmul(std::vector<Tin> a, std::vector<Tin> b, std::vector<Tout> &c) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      Tout running_sum = 0;
      for (int i = 0; i < K; i++) {
        running_sum += Tout(a[row * K + i] * b[i * N + col]);
      }
      c[row * N + col] += running_sum;
    }
  }
}

void write_out_trace(char *bufOut, std::string path)
{
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)((char *)bufOut + sizeof(C_DATATYPE)*C_VOLUME);
  for(int i = 0; i < TRACE_SIZE/sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "produce help message")
      ("xclbin,x", po::value<std::string>()->required(), "the input xclbin path")
      ("kernel,k", po::value<std::string>()->required(), "the kernel name in the XCLBIN (for instance PP_PRE_FD)")
      ("verbosity,v", po::value<int>()->default_value(0), "the verbosity of the output")
      ("instr,i", po::value<std::string>()->required(), "path of file containing userspace instructions to be sent to the LX6");
  if(ENABLE_TRACING) {
    desc.add_options()
      ("trace,t", po::value<std::string>()->default_value("trace.txt"), "where to store trace output");
  }
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
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if(verbosity >= 1) {
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
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  srand(static_cast<unsigned>(time(0)));
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec;
  for (int i = 0; i < A_VOLUME; i++)
    AVec.push_back(random_bfloat16_t());
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec;
  for (int i = 0; i < B_VOLUME; i++)
    BVec.push_back(random_bfloat16_t());
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  memset(bufOut, 0, OUT_SIZE);

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto start = std::chrono::system_clock::now();
  auto run = kernel(bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
  run.wait();
  auto stop = std::chrono::system_clock::now();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reinterpret first C_VOLUME bytes of bufOut as our output C_DATATYPE C matrix
  C_DATATYPE *COut = (C_DATATYPE *)bufOut;

  int errors = 0;
  int max_errors = 100;

  if (VERIFY) {
    std::vector<C_DATATYPE> output_ref0;
    for (uint32_t i = 0; i < C_VOLUME; i++)
      output_ref0.push_back(0);
    matmul(AVec, BVec, output_ref0);

    const C_DATATYPE absTol = std::abs(0.1);
    for (uint32_t i = 0; i < C_VOLUME; i++) {
      if (std::abs(COut[i] - output_ref0[i]) > absTol) {
        errors++;
        if (errors < max_errors) {
          std::cout << "\nerror, id " << i << " expected "
                    << std::to_string(output_ref0[i]) << ", got "
                    << std::to_string(bufOut[i]) << "\n";
        }
      }
    }
  }

  if (ENABLE_TRACING) {
    write_out_trace(bufOut, vm["trace"].as<std::string>());
  }

  std::cout << std::endl
            << "NPU matmul time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                     start)
                   .count()
            << "ms." << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nerror count: " << errors << "\n\n";
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
